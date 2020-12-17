import unittest

import jieba
from autophrasex.tokenizer import BaiduLacTokenizer


class TokenizerTest(unittest.TestCase):

    def testLacTokenizer(self):
        tokenizer = BaiduLacTokenizer()
        results = tokenizer.tokenize('测试百度LAC分刺效果')
        print(results)

        print(tokenizer.tokenize('资产证券化'))
        print(tokenizer.tokenize('分级债基'))
        print(tokenizer.tokenize('英菲尼迪'))
        for text in ['大学生村官', '上交所', '李女士', '内转载', '中国雅虎', '河南频道', '权利人', '氩氦刀']:
            print(tokenizer.tokenize(text))

    def testJiebaTokenizer(self):
        print(jieba.lcut('可口可乐公司'))
        print(jieba.lcut('英菲尼迪'))

        for text in ['大学生村官', '上交所', '李女士', '内转载', '中国雅虎', '河南频道', '权利人', '氩氦刀']:
            print(jieba.lcut(text))


if __name__ == "__main__":
    unittest.main()
