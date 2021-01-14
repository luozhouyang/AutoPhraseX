import os
import unittest

import jieba
from autophrasex import utils
from autophrasex.autophrase import AutoPhrase
from autophrasex.strategy import Strategy
from autophrasex.tokenizer import BaiduLacTokenizer, JiebaTokenizer


class AutoPhraseTest(unittest.TestCase):

    def test_autophrase_small(self):
        tokenizer = BaiduLacTokenizer()
        # tokenizer = JiebaTokenizer()
        strategy = Strategy(
            tokenizer=tokenizer,
            N=4,
            epsilon=1e-6,
            prob_threshold=0.25,
            prob_threshold_schedule_factor=1.0)
        ap = AutoPhrase()
        predictions = ap.mine(
            input_doc_files=['data/sogou_news_tensite_content.100.txt'],
            quality_phrase_files='data/wiki_quality.txt',
            strategy=strategy,
            N=4,
            epochs=10,
        )
        for p in predictions:
            print(p)

    def test_autophrase_large(self):
        # tokenizer = BaiduLacTokenizer()
        tokenizer = JiebaTokenizer()
        strategy = Strategy(
            tokenizer=tokenizer,
            N=4,
            epsilon=1e-6,
            prob_threshold=0.45,
            prob_threshold_schedule_factor=1.05,
            phrase_min_freq=2,
            phrase_max_count=1000)

        ap = AutoPhrase()
        predictions = ap.mine(
            input_doc_files=['data/sogou_news_tensite_content.10000.txt'],
            quality_phrase_files=['data/wiki_quality.txt', 'data/quality_pred.txt'],
            strategy=strategy,
            N=4,
            epochs=10,
        )
        with open('data/predictions.txt', mode='wt', encoding='utf8') as fout:
            for n, p in predictions:
                fout.write('{}, {}\n'.format(n, p))


if __name__ == "__main__":
    unittest.main()
