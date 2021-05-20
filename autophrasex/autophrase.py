import json
import logging
import os
import random
from copy import deepcopy

from sklearn.ensemble import RandomForestClassifier

from autophrasex.extractors import FeatureExtractorWrapper
from autophrasex.reader import AbstractCorpusReader

from . import utils
from .callbacks import CallbackWrapper
from .selector import AbstractPhraseSelector


def load_quality_phrase_files(input_files):
    pharses = set()

    def collect_fn(line, lino):
        pharses.add(line.strip())

    utils.load_input_files(input_files, callback=collect_fn)
    return pharses


class AutoPhrase:

    def __init__(self,
                 reader: AbstractCorpusReader,
                 selector: AbstractPhraseSelector,
                 extractors=None,
                 classifier=None,
                 threshold=0.4,
                 **kwargs):
        """Constractor

        Args:
            reader: Instance of AbstractCorpusReader, used to read corpus files
            selector: Instance of AbstractPhraseSelector, used to select frequent phrases
            extractors: List of AbstractFeatureExtractor, used to extract features for classifier
            threshold: Python float, negative phrase whose prob greater than this will be moved to positive pool
        """
        self.selector = selector
        self.extractors = extractors or []
        self.extractor_wrapper = FeatureExtractorWrapper(extractors=self.extractors)
        self.corpus_reader = reader

        if classifier is None:
            classifier = RandomForestClassifier(**kwargs)
        self.classifier = classifier

        # used by ThresholdSchedule
        self.threshold = threshold
        # used by EarlyStopping
        self.early_stop = False

    def mine(self,
             corpus_files,
             quality_phrase_files,
             N=4,
             epochs=10,
             callbacks=None,
             topk=300,
             filter_fn=None,
             **kwargs):
        """Mining phrase from corpus.

        Args:
            corpus_files: Files of corpus
            quality_phrase_files: File path(s) of quality phrases, one phrase each line
            epochs: Python integer, Number of training epoch
            callbacks: List of Callback, used to listen lifecycles
            topk: Python integer, Number of frequent phrases selected from Selector
            filter_fn: Python callable, with signature fn(phrase, freq), used to filter phrases

        Return:
            predictions: List of tuple (phrase, prob), predict from initial negative phrase pool
        """
        callback = CallbackWrapper(callbacks=callbacks)
        callback.begin()

        callback.on_read_corpus_begin()
        self.corpus_reader.read(
            corpus_files=corpus_files,
            extractor=self.extractor_wrapper,
            N=N,
            verbose=kwargs.get('verbose', True),
            logsteps=kwargs.get('logsteps', 1000))
        callback.on_read_corpus_end()

        callback.on_build_quality_phrases_begin(quality_phrase_files)
        quality_phrases = load_quality_phrase_files(quality_phrase_files)
        callback.on_build_quality_phrases_end(quality_phrases)

        callback.on_select_frequent_phrases_begin()
        frequent_phrases = self.selector.select(
            extractors=self.extractors,
            topk=topk,
            filter_fn=filter_fn,
            **kwargs)
        callback.on_select_frequent_phrases_end(frequent_phrases)

        callback.on_organize_phrase_pools_begin(quality_phrases, frequent_phrases)
        initial_pos_pool, initial_neg_pool = self._organize_phrase_pools(quality_phrases, frequent_phrases, **kwargs)
        callback.on_organize_phrase_pools_end(initial_pos_pool, initial_neg_pool)

        pos_pool, neg_pool = initial_pos_pool, initial_neg_pool
        for epoch in range(epochs):
            callback.on_epoch_begin(epoch)

            callback.on_epoch_prepare_training_data_begin(epoch)
            x, y = self._prepare_training_data(pos_pool, neg_pool, **kwargs)
            callback.on_epoch_prepare_training_data_end(epoch, x, y)

            self.classifier.fit(x, y)

            callback.on_epoch_reorganize_phrase_pools_begin(epoch, pos_pool, neg_pool)
            pos_pool, neg_pool = self._reorganize_phrase_pools(pos_pool, neg_pool, **kwargs)
            callback.on_epoch_reorganize_phrase_pools_end(epoch, pos_pool, neg_pool)

            if self.early_stop:
                logging.info('    early stop!')
                break

            callback.on_epoch_end(epoch)

        callback.on_predict_neg_pool_begin(neg_pool)
        predictions = self._predict_proba(initial_neg_pool)
        predictions = sorted(predictions, key=lambda x: x[1], reverse=True)
        callback.on_predict_neg_pool_end(predictions)

        callback.end()
        return predictions

    def _prepare_training_data(self, pos_pool, neg_pool, **kwargs):
        x, y = [], []
        examples = []
        for p in pos_pool:
            examples.append((self._compose_feature(p), 1))
        for p in neg_pool:
            examples.append((self._compose_feature(p), 0))
        # shuffle
        random.shuffle(examples)
        for _x, _y in examples:
            x.append(_x)
            y.append(_y)
        return x, y

    def _reorganize_phrase_pools(self, pos_pool, neg_pool, **kwargs):
        new_pos_pool, new_neg_pool = [], []
        new_pos_pool.extend(deepcopy(pos_pool))

        pairs = self._predict_proba(neg_pool)
        pairs = sorted(pairs, key=lambda x: x[1], reverse=True)
        # print(pairs[:10])

        for idx, (p, prob) in enumerate(pairs):
            if prob > self.threshold:
                new_pos_pool.append(p)
                continue
            new_neg_pool.append(p)

        return new_pos_pool, new_neg_pool

    def _organize_phrase_pools(self, quality_phrases, frequent_phrases, **kwargs):
        pos_pool, neg_pool = [], []
        for p in frequent_phrases:
            if p in quality_phrases:
                pos_pool.append(p)
                continue
            _p = ''.join(p.split(' '))
            if _p in quality_phrases:
                pos_pool.append(p)
                continue
            neg_pool.append(p)
        return pos_pool, neg_pool

    def _predict_proba(self, phrases):
        features = [self._compose_feature(phrase) for phrase in phrases]
        pos_probs = [prob[1] for prob in self.classifier.predict_proba(features)]
        pairs = [(phrase, prob) for phrase, prob in zip(phrases, pos_probs)]
        return pairs

    def _compose_feature(self, phrase):
        features = self.extractor_wrapper.extract(phrase)
        features = sorted(features.items(), key=lambda x: x[0])
        features = [x[1] for x in features]
        return features
