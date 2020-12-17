import json
import logging
import os
import random
import re

from sklearn.ensemble import RandomForestClassifier

from . import utils
from .strategy import AbstractStrategy
from .tokenizer import BaiduLacTokenizer

CHINESE_PATTERN = re.compile(r'^[0-9a-zA-Z\u4E00-\u9FA5]+$')


def load_quality_phrase_files(input_files):
    pharses = set()

    def collect_fn(line, lino):
        pharses.add(line.strip())

    utils.load_input_files(input_files, callback=collect_fn)
    return pharses


class AutoPhrase:

    def __init__(self, **kwargs):
        max_depth = kwargs.pop('max_depth', 6)
        n_jobs = kwargs.pop('n_jobs', 6)
        self.classifier = RandomForestClassifier(max_depth=max_depth, n_jobs=n_jobs)

    def mine(self, input_doc_files, quality_phrase_files, strategy: AbstractStrategy, N=4, **kwargs):
        strategy.fit(input_doc_files, N=N, **kwargs)

        quality_phrases = load_quality_phrase_files(quality_phrase_files)
        logging.info('Load quality phrases finished. There are %d quality phrases in total.', len(quality_phrases))

        frequent_phrases = strategy.select_frequent_phrases(**kwargs)
        logging.info('Selected %d frequent phrases.', len(frequent_phrases))

        initial_pos_pool, initial_neg_pool = strategy.build_phrase_pool(quality_phrases, frequent_phrases, **kwargs)
        logging.info('Size of initial positive pool: %d', len(initial_pos_pool))
        logging.info('Size of initial negative pool: %d', len(initial_neg_pool))
        # TODO: 第一次构建正负池的时候，把unigrams作为正样本
        # TODO: 观察训练过程，调整threshold_schedule_factor的值，有可能需要小于1

        pos_pool, neg_pool = initial_pos_pool, initial_neg_pool
        for epoch in range(kwargs.pop('epochs', 5)):
            logging.info('Starting to train model at epoch %d ...', epoch + 1)
            x, y = strategy.compose_training_data(pos_pool, neg_pool, **kwargs)
            self.classifier.fit(x, y)
            logging.info('Finished to train model at epoch %d', epoch + 1)

            logging.info('Starting to adjust phrase pool...')
            pos_pool, neg_pool, stop = strategy.adjust_phrase_pool(pos_pool, neg_pool, self.classifier, epoch, **kwargs)
            logging.info('Finished to djusted phrase pools. ')
            logging.info('\t size of positive pool: %d, size of ngeative pool: %d', len(pos_pool), len(neg_pool))
            if stop:
                logging.info('Early stopped.')
                break

        logging.info('Finished to train model!')
        features = [strategy.build_input_features(p) for p in initial_neg_pool]
        pos_probas = [prob[1] for prob in self.classifier.predict_proba(features)]
        predictions = [(p, prob) for p, prob in zip(initial_neg_pool, pos_probas)]
        predictions = sorted(predictions, key=lambda x: x[1], reverse=True)
        return predictions
