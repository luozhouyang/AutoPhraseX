import json
import logging
import os
import random

from sklearn.ensemble import RandomForestClassifier

from . import utils
from .callbacks import EntropyCallback, IDFCallback, NgramsCallback
from .composer import AbstractFeatureComposer, DefaultFeatureComposer
from .reader import AbstractCorpusReader, DefaultCorpusReader
from .selector import AbstractPhraseSelector, DefaultPhraseSelector
from .tokenizer import BaiduLacTokenizer


def load_quality_phrase_files(input_files):
    pharses = set()

    def collect_fn(line, lino):
        pharses.add(line.strip())

    utils.load_input_files(input_files, callback=collect_fn)
    return pharses


class AutoPhrase:

    def __init__(self, selector: AbstractPhraseSelector, composer: AbstractFeatureComposer, **kwargs):
        self.selector = selector
        self.composer = composer
        self.classifier = RandomForestClassifier(**kwargs)
        self.threshold = 0.4
        self.early_stop = None

    def mine(self, quality_phrase_files, **kwargs):

        quality_phrases = load_quality_phrase_files(quality_phrase_files)
        logging.info('Load quality phrases finished. There are %d quality phrases in total.', len(quality_phrases))

        frequent_phrases = self.selector.select(**kwargs)

        logging.info('Selected %d frequent phrases.', len(frequent_phrases))

        initial_pos_pool, initial_neg_pool = self._organize_phrase_pools(quality_phrases, frequent_phrases, **kwargs)
        logging.info('Size of initial positive pool: %d', len(initial_pos_pool))
        logging.info('Size of initial negative pool: %d', len(initial_neg_pool))

        pos_pool, neg_pool = initial_pos_pool, initial_neg_pool
        for epoch in range(kwargs.pop('epochs', 5)):
            logging.info('Starting to train model at epoch %d ...', epoch + 1)
            x, y = self._prepare_training_data(pos_pool, neg_pool, **kwargs)
            self.classifier.fit(x, y)
            logging.info('Finished to train model at epoch %d', epoch + 1)

            logging.info('Starting to adjust phrase pool...')
            pos_pool, neg_pool, stop = self._reorganize_phrase_pools(pos_pool, neg_pool, **kwargs)
            logging.info('Finished to djusted phrase pools. ')
            logging.info('\t size of positive pool: %d, size of ngeative pool: %d', len(pos_pool), len(neg_pool))
            if stop:
                logging.info('Early stopped.')
                break

        logging.info('Finished to train model!')
        predictions = self._predict_proba(initial_neg_pool)
        predictions = sorted(predictions, key=lambda x: x[1], reverse=True)
        return predictions

    def _prepare_training_data(self, pos_pool, neg_pool, **kwargs):
        x, y = [], []
        examples = []
        for p in pos_pool:
            examples.append((self.composer.compose(p), 1))
        for p in neg_pool:
            examples.append((self.composer.compose(p), 0))
        # shuffle
        random.shuffle(examples)
        for _x, _y in examples:
            x.append(_x)
            y.append(_y)
        return x, y

    def _reorganize_phrase_pools(self, pos_pool, neg_pool, **kwargs):
        new_pos_pool, new_neg_pool = [], []
        new_pos_pool.extend(pos_pool.clone())

        pairs = self._predict_proba(neg_pool)
        pairs = sorted(pairs, key=lambda x: x[1], reverse=True)
        print(pairs[:10])

        for idx, (p, prob) in enumerate(pairs):
            if prob > self.threshold:
                new_pos_pool.append(p)
                continue
            new_neg_pool.append(p)

        return new_pos_pool, new_neg_pool

    def _organize_phrase_pools(self, quality_phrases, frequent_phrases, **kwargs):
        pos_pool, neg_pool = [], []
        for p in frequent_phrases:
            _p = ''.join(p.split(' '))
            if _p in quality_phrases:
                pos_pool.append(p)
            else:
                neg_pool.append(p)
        return pos_pool, neg_pool

    def _predict_proba(self, phrases):
        features = [self.composer.compose(x) for x in phrases]
        pos_probs = [prob[1] for prob in self.classifier.predict_proba(features)]
        pairs = [(phrase, prob) for phrase, prob in zip(phrases, pos_probs)]
        return pairs


if __name__ == "__main__":
    pass
