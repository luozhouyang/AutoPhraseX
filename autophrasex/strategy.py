import abc
import logging
import os
import random

from LAC import LAC

from . import utils
from .callbacks import (CallbackWrapper, EntropyCallback, IDFCallback,
                        NgramsCallback)


def read_input_files(input_files, callback=None, log_steps=1000):
    if isinstance(input_files, str):
        input_files = [input_files]
    num = 0
    for f in input_files:
        if not os.path.exists(f):
            logging.warning('File %s does not exist.', f)
            continue
        with open(f, mode='rt', encoding='utf8') as fin:
            for line in fin:
                line = line.rstrip('\n')
                if not line:
                    continue
                if callback:
                    callback(line)
                num += 1
                if num % log_steps == 0:
                    logging.info('Finished to process %d lines.', num)
        logging.info('Finished to process %d lines in total of file: %s', num, f)
    logging.info('Finished to process %d lines in total of all files.', num)
    logging.info('Done!')


class AbstractStrategy:

    def __init__(self, tokenizer, callbacks=None, **kwargs):
        self.tokenizer = tokenizer
        self.callback = CallbackWrapper(callbacks=callbacks)

    def fit(self, input_doc_files, N=4, **kwargs):

        def read_line(line):
            # callbacks process doc begin
            self.callback.on_process_doc_begin()
            tokens = self.tokenizer.tokenize(line, **kwargs)
            # callbacks process tokens
            self.callback.update_tokens(tokens, **kwargs)
            # callbacks process ngrams
            for n in range(1, N + 1):
                for (start, end), window in utils.ngrams(tokens, n=n):
                    self.callback.update_ngrams(start, end, window, n, **kwargs)
            # callbacks process doc end
            self.callback.on_process_doc_end()

        read_input_files(input_doc_files, callback=read_line, log_steps=kwargs.get('log_steps', 1000))

    def select_frequent_phrases(self, **kwargs):
        raise NotImplementedError()

    def build_phrase_pool(self, quality_phrases, frequent_phrases, **kwargs):
        raise NotImplementedError()

    def compose_training_data(self, pos_pool, neg_pool, **kwargs):
        raise NotImplementedError()

    def adjust_phrase_pool(self, pos_pool, neg_pool, classifier, epoch, **kwargs):
        raise NotImplementedError()

    def build_input_features(self, phrase, **kwargs):
        raise NotImplementedError()


def default_phrase_filter_fn(phrase, count):
    if any(phrase.endswith(x) for x in ['仅', '中']):
        return True
    return False


class Strategy(AbstractStrategy):

    def __init__(self, tokenizer, N=4, epsilon=0.0, **kwargs):
        self.ngrams_callback = NgramsCallback(n=N, epsilon=epsilon)
        self.idf_callback = IDFCallback(epsilon=epsilon)
        self.entropy_callback = EntropyCallback(epsilon=epsilon)
        callbacks = [self.ngrams_callback, self.idf_callback, self.entropy_callback]
        super().__init__(tokenizer=tokenizer, callbacks=callbacks, **kwargs)

        self.phrase_filter_fn = kwargs.get('phrase_filter_fn', default_phrase_filter_fn)
        self.phrase_max_count = kwargs.get('phrase_max_count', 1000)
        self.phrase_min_length = kwargs.get('phrase_min_length', 3)
        self.phrase_min_unigram_length = kwargs.get('phrase_min_unigram_length', 3)
        self.phrase_min_freq = kwargs.get('phrase_min_freq', 5)
        self.phrase_drop_stopwords = kwargs.get('phrase_drop_stopwords', True)
        self.phrase_drop_verbs = kwargs.get('phrase_drop_verbs', True)
        if self.phrase_drop_verbs:
            self.lac = LAC()
        self.prob_threshold = kwargs.get('prob_threshold', 0.45)
        self.prob_threshold_schedule_factor = kwargs.get('prob_threshold_schedule_factor', 1.0)
        self.prob_topk = kwargs.get('prob_topk', 10)
        self.min_delta = kwargs.get('min_delta', 3)

        self.features_cache = {}

    def select_frequent_phrases(self, **kwargs):
        candidates = []
        for n in range(1, self.ngrams_callback.N + 1):
            counter = self.ngrams_callback.ngrams_freq[n]
            for phrase, count in counter.items():
                _phrase = ''.join(phrase.split(' '))
                if self._filter_phrase(_phrase, count, **kwargs):
                    continue
                candidates.append((phrase, count))

        if self.phrase_drop_verbs:
            candidates = self._drop_verbs(candidates)
        candidates = sorted(candidates, key=lambda x: x[1], reverse=True)
        return [x[0] for x in candidates[:self.phrase_max_count]]

    def _filter_phrase(self, phrase, count, **kwargs):
        if len(phrase) < self.phrase_min_length:
            return True
        if count < self.phrase_min_freq:
            return True
        if self.phrase_drop_stopwords and any(utils.STOPWORDS.contains(x) for x in phrase):
            return True
        if self.phrase_filter_fn(phrase, count):
            return True
        return False

    def _drop_verbs(self, candidates):
        predictions = []
        for i in range(0, len(candidates), 100):
            # batch_count = [x[1] for x in candidates[i:i+100]]
            batch_texts = [x[0] for x in candidates[i:i+100]]
            batch_preds = self.lac.run(batch_texts)
            predictions.extend(batch_preds)
        filtered_candidates = []
        for i in range(len(predictions)):
            _, pos_tags = predictions[i]
            if any(pos in ['v', 'vn', 'vd'] for pos in pos_tags):
                continue
            filtered_candidates.append(candidates[i])
        return filtered_candidates

    def build_phrase_pool(self, quality_phrases, frequent_phrases, **kwargs):
        pos_pool, neg_pool = [], []
        for p in frequent_phrases:
            _p = ''.join(p.split(' '))
            # unigrams are positve phrase
            if _p in self.ngrams_callback.ngrams_freq[1] and len(_p) > self.phrase_min_unigram_length:
                pos_pool.append(p)
                continue
            if _p in quality_phrases:
                if len(p) > self.phrase_min_unigram_length:
                    pos_pool.append(p)
            else:
                neg_pool.append(p)
        return pos_pool, neg_pool

    def compose_training_data(self, pos_pool, neg_pool, **kwargs):
        x, y = [], []
        examples = []
        for p in pos_pool:
            examples.append((self.build_input_features(p), 1))
        for p in neg_pool:
            examples.append((self.build_input_features(p), 0))
        # shuffle
        random.shuffle(examples)
        for _x, _y in examples:
            x.append(_x)
            y.append(_y)
        return x, y

    def adjust_phrase_pool(self, pos_pool, neg_pool, classifier, epoch, **kwargs):
        new_pos_pool, new_neg_pool = [], []
        new_pos_pool.extend(pos_pool)

        input_features = [self.build_input_features(x) for x in neg_pool]
        pos_probs = [prob[1] for prob in classifier.predict_proba(input_features)]
        pairs = [(p, prob) for p, prob in zip(neg_pool, pos_probs)]
        pairs = sorted(pairs, key=lambda x: x[1], reverse=True)
        print(pairs[:10])

        threshold = self._schedule_threshold(epoch, **kwargs)
        logging.info('\t using threshold: %f', threshold)

        for idx, (p, prob) in enumerate(pairs):
            if prob > threshold:
                new_pos_pool.append(p)
                continue
            new_neg_pool.append(p)

        stop = True if len(new_pos_pool) - len(pos_pool) < self.min_delta else False
        return new_pos_pool, new_neg_pool, stop

    def _schedule_threshold(self, epoch, **kwargs):
        threshold = min(self.prob_threshold * self.prob_threshold_schedule_factor**epoch, 0.95)
        return threshold

    def build_input_features(self, phrase, **kwargs):

        def _convert_to_inputs(features):
            example = []
            for k in ['unigram', 'freq', 'doc_freq', 'idf', 'pmi', 'le', 're']:
                example.append(features[k])
            return example

        if phrase in self.features_cache:
            features = self.features_cache[phrase]
            return _convert_to_inputs(features)

        ngrams = phrase.split(' ')
        counter = self.ngrams_callback.ngrams_freq[len(ngrams)]
        freq = counter[''.join(ngrams)] / sum(counter.values())
        doc_freq = self.idf_callback.doc_freq_of(phrase)
        idf = self.idf_callback.idf_of(phrase)
        pmi = self.ngrams_callback.pmi_of(phrase)
        left_entropy = self.entropy_callback.left_entropy_of(phrase)
        right_entropy = self.entropy_callback.right_entropy_of(phrase)
        features = {
            'unigram': 1 if len(ngrams) == 1 else 0,
            'freq': freq,
            'doc_freq': doc_freq,
            'idf': idf,
            'pmi': pmi,
            'le': left_entropy,
            're': right_entropy,
        }
        self.features_cache[phrase] = features
        return _convert_to_inputs(features)
