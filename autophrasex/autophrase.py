import json
import logging
import os
import random
import re

from sklearn.ensemble import RandomForestClassifier

from autophrasex import utils

from . import utils
from .docinfo import DocInfo
from .tokenizer import BaiduLacTokenizer

CHINESE_PATTERN = re.compile(r'^[0-9a-zA-Z\u4E00-\u9FA5]+$')


def load_quality_phrase_files(input_files):
    pharses = set()

    def collect_fn(line, lino):
        pharses.add(line.strip())

    utils.load_input_files(input_files, callback=collect_fn)
    logging.info('Load quality phrases finished. There are %d quality phrases in total.', len(pharses))
    return pharses


def load_doc_info_file(input_file):
    info = {}

    def collect_fn(line, lino):
        instance = json.loads(line)
        k = list(instance.keys())[0]
        k = ''.join(x.strip() for x in k.split(' '))
        info[k] = instance[k]

    utils.load_input_files(input_file, callback=collect_fn)
    return info


def default_doc_process_fn(doc):
    return utils.uniform_chinese_text(doc)


def default_ngram_filter_fn(ngrams):
    if CHINESE_PATTERN.match(''.join(ngrams)):
        return False
    return True


class AutoPhraseX:

    def __init__(self, quality_files, **kwargs):
        self.quality_phrases = load_quality_phrase_files(quality_files)
        # TODO: use XGBoost instead of scikit
        self.classifier = RandomForestClassifier(max_depth=6, max_samples=0.8, n_jobs=6)

    def mine(self,
             input_doc_files,
             tokenize_fn,
             threshold=0.95,
             min_delta=3,
             doc_process_fn=default_doc_process_fn,
             ngram_filter_fn=default_ngram_filter_fn,
             phrase_filter_fn=None,
             min_freq=5,
             max_phrase_num=10000,
             **kwargs):

        doc_info = DocInfo.from_corpus(
            input_doc_files,
            tokenize_fn=tokenize_fn,
            doc_process_fn=doc_process_fn,
            ngram_filter_fn=ngram_filter_fn)
        logging.info('Calculate doc info finished.')

        frequent_phrases = self._select_frequent_phrases(
            doc_info, phrase_filter_fn=phrase_filter_fn, min_freq=min_freq, max_num=max_phrase_num)
        logging.info('The size of frequent phrases: %d', len(frequent_phrases))

        initial_pos_pool, initial_neg_pool = self._build_phrase_pool(frequent_phrases)
        logging.info('The size of initial positive pool: %d', len(initial_pos_pool))
        logging.info('The size of initial negative pool: %d', len(initial_neg_pool))

        pos_pool, neg_pool = initial_pos_pool, initial_neg_pool
        for epoch in range(kwargs.get('epochs', 5)):
            logging.info('Start to train model on epoch: %d', epoch + 1)
            x, y = self._compose_training_data(pos_pool, neg_pool, doc_info)
            logging.info('Compose training data finished.')
            self.classifier.fit(x, y)
            logging.info('Finished No.%d epoch.', epoch)
            pos_pool, neg_pool, stop = self._reorganize_phrase_pool(doc_info, pos_pool, neg_pool, threshold, min_delta)
            logging.info('Reorganized pool finished. ')
            logging.info('size of positive pool: %d, size of negative pool: %d', len(pos_pool), len(neg_pool))
            if stop:
                logging.info('Early stopped.')
                break
            logging.info('size of positive pool: %d, size of negative pool: %d', len(pos_pool), len(neg_pool))

        logging.info('Training done!')

        input_features = [self._build_input_features(p, doc_info) for p in initial_neg_pool]
        pos_probs = [prob[1] for prob in self.classifier.predict_proba(input_features)]
        predictions = [(p, prob) for p, prob in zip(initial_neg_pool, pos_probs)]
        predictions = sorted(predictions, key=lambda x: x[1], reverse=True)
        return predictions

    def _reorganize_phrase_pool(self, doc_info, pos_pool, neg_pool, threshold=0.95, min_delta=3):
        new_pos_pool, new_neg_pool = [], []
        new_pos_pool.extend(pos_pool)
        input_features = [self._build_input_features(x, doc_info) for x in neg_pool]
        neg_probs = [prob[0] for prob in self.classifier.predict_proba(
            input_features)]  # prob[0]: prob of class negative
        for p, prob in zip(pos_pool, neg_probs):
            if prob <= 1.0 - threshold:
                new_pos_pool.append(p)
            else:
                new_neg_pool.append(p)
        stop = True if len(new_pos_pool) - len(pos_pool) < min_delta else False
        return new_pos_pool, new_neg_pool, stop

    def _compose_training_data(self, pos_pool, neg_pool, doc_info):
        x, y = [], []
        examples = []
        for p in pos_pool:
            examples.append((self._build_input_features(p, doc_info), 1))
        for p in neg_pool:
            examples.append((self._build_input_features(p, doc_info), 0))
        # shuffle
        random.shuffle(examples)
        for _x, _y in examples:
            x.append(_x)
            y.append(_y)
        return x, y

    def _build_input_features(self, phrase, doc_info):
        # feature names: ngram_freq, doc_freq, pmi, le, re
        info = doc_info.inspect_of(phrase)
        return [info['freq'], info['doc_freq'], info['idf'], info['pmi'], info['le'], info['re']]

    def _select_frequent_phrases(self, doc_info, phrase_filter_fn=None, min_freq=5, max_num=10000):
        phrases = []
        for phrase, freq in doc_info.ngrams(with_freq=True, with_n=False, min_freq=min_freq):
            if phrase_filter_fn and phrase_filter_fn(phrase):
                continue
            phrases.append((phrase, freq))
        phrases = sorted(phrases, key=lambda x: x[1], reverse=True)
        phrases = [x for x, _ in phrases[:max_num]]
        return phrases

    def _build_phrase_pool(self, frequent_phrases):
        pos_pool, neg_pool = [], []
        for p in frequent_phrases:
            if p in self.quality_phrases:
                pos_pool.append(p)
            else:
                neg_pool.append(p)
        return pos_pool, neg_pool

