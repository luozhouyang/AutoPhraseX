import unittest

from autophrasex.docinfo import DocInfo


class DocInfoTest(unittest.TestCase):

    def testDocInfo(self):
        docinfo = DocInfo(epsilon=0.0)
        docinfo.update(['hello', 'world'])
        print(docinfo.inspect_of('hello'))
        docinfo.inspect()


if __name__ == "__main__":
    unittest.main()
