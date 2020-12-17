import os
import unittest

import jieba
from autophrasex import utils
from autophrasex.autophrase import AutoPhrase
from autophrasex.strategy import Strategy
from autophrasex.tokenizer import BaiduLacTokenizer, JiebaTokenizer


class AutoPhraseTest(unittest.TestCase):

    def test_autophrase(self):
        tokenizer = BaiduLacTokenizer()
        # tokenizer = JiebaTokenizer()
        strategy = Strategy(
            tokenizer=tokenizer,
            N=4,
            epsilon=1e-6,
            threshold=0.45,
            threshold_schedule_factor=1.0)

        # files = [os.path.join('data/clean_text', f) for f in os.listdir('data/clean_text')]
        ap = AutoPhrase()
        predictions = ap.mine(
            input_doc_files=['data/sogou_news_tensite_content.10000.txt'],
            # input_doc_files=[os.path.join('data/medical', f) for f in os.listdir('data/medical')],
            # input_doc_files=['data/medical/data_2.txt'],
            # input_doc_files=files,
            quality_phrase_files='data/wiki_quality.txt',
            strategy=strategy,
            N=4,
            epochs=10,
        )
        with open('data/predictions.txt', mode='wt', encoding='utf8') as fout:
            for n, p in predictions:
                fout.write('{}, {}\n'.format(n, p))


if __name__ == "__main__":
    unittest.main()
