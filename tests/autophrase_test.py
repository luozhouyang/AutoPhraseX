import unittest

import jieba
from autophrasex import utils
from autophrasex.autophrase import AutoPhraseX
from autophrasex.tokenizer import BaiduLacTokenizer

tokenizer = BaiduLacTokenizer()


def tokenize(x):
    return tokenizer.tokenize(x)

def jieba_tokenize(x):
    return jieba.lcut(x)


class AutoPhraseTest(unittest.TestCase):

    def test_autophrase(self):
        ap = AutoPhraseX(quality_files='data/wiki_quality.txt')
        predictions = ap.mine(input_doc_files=['data/sogou_news_tensite_content.10000.txt'],
                              tokenize_fn=jieba_tokenize,
                              )
        print(predictions[:10])


if __name__ == "__main__":
    unittest.main()
