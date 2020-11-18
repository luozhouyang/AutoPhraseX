import unittest 

from autophrasex.docinfo import DocInfo


class DocInfoTest(unittest.TestCase):

    def testDocInfo(self):
        docinfo = DocInfo()
        docinfo.update(['hello', 'world'])
        print(docinfo.inspect_of('hello'))


if __name__ == "__main__":
    unittest.main()
