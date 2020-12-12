import unittest

import jieba
from autophrasex.tokenizer import BaiduLacTokenizer


class TokenizerTest(unittest.TestCase):

    def testLacTokenizer(self):
        tokenizer = BaiduLacTokenizer()
        results = tokenizer.tokenize('测试百度LAC分刺效果')
        print(results)

        results = tokenizer.tokenize(['测试百度LAC分刺效果,LAC是北京百度有限公司的产品'])
        print(results)

        print(tokenizer.tokenize('资产证券化'))
        print(tokenizer.tokenize('分级债基'))
        print(tokenizer.tokenize('英菲尼迪'))

    def testJiebaTokenizer(self):
        print(jieba.lcut('可口可乐公司'))
        print(jieba.lcut('英菲尼迪'))

if __name__ == "__main__":
    unittest.main()
