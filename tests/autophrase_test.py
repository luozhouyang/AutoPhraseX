import os
import unittest

import jieba
from autophrasex import utils
from autophrasex.autophrase import AutoPhrase
from autophrasex.callbacks import (ConstantThresholdScheduler, EarlyStopping,
                                   LoggingCallback)
from autophrasex.composer import DefaultFeatureComposer
from autophrasex.extractors import *
from autophrasex.reader import DefaultCorpusReader
from autophrasex.selector import DefaultPhraseSelector
from autophrasex.tokenizer import BaiduLacTokenizer, JiebaTokenizer


class AutoPhraseTest(unittest.TestCase):

    def test_autophrase_small(self):
        N = 4
        ngrams_extractor = NgramsExtractor(n=N)
        idf_extractor = IDFExtractor()
        entropy_extractor = EntropyExtractor()

        reader = DefaultCorpusReader(
            tokenizer=BaiduLacTokenizer(),
            extractors=[ngrams_extractor, idf_extractor, entropy_extractor])
        reader.read(corpus_files=['data/DBLP.5K.txt'], N=N, verbose=True, logsteps=500)

        autophrase = AutoPhrase(
            selector=DefaultPhraseSelector(ngrams_extractor=ngrams_extractor),
            composer=DefaultFeatureComposer(idf_extractor, ngrams_extractor, entropy_extractor),
        )

        predictions = autophrase.mine(
            quality_phrase_files='data/wiki_quality.txt',
            callbacks=[
                LoggingCallback(),
                ConstantThresholdScheduler(autophrase),
                EarlyStopping(autophrase, patience=1, min_delta=3)
            ])
        for pred in predictions:
            print(pred)


if __name__ == "__main__":
    unittest.main()
