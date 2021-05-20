import unittest

import jieba
from autophrasex.tokenizer import JiebaTokenizer


class TokenizerTest(unittest.TestCase):

    def testJiebaTokenizer(self):
        print(jieba.lcut('可口可乐公司'))
        print(jieba.lcut('英菲尼迪'))

        for text in ['大学生村官', '上交所', '李女士', '内转载', '中国雅虎', '河南频道', '权利人', '氩氦刀']:
            print(jieba.lcut(text))


if __name__ == "__main__":
    unittest.main()
