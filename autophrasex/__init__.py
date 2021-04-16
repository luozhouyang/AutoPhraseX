import logging

from .autophrase import AutoPhrase
from .callbacks import (Callback, ConstantThresholdScheduler, EarlyStopping,
                        LoggingCallback, StateCallback)
from .composer import AbstractFeatureComposer, DefaultFeatureComposer
from .extractors import (AbstractExtractorCallback, EntropyExtractor,
                         IDFExtractor, NgramsExtractor)
from .filters import (AbstractPhraseFilter, PhraseFilterWrapper,
                      VerbPhraseFilter)
from .reader import AbstractCorpusReader, DefaultCorpusReader
from .selector import AbstractPhraseSelector, DefaultPhraseSelector
from .tokenizer import BaiduLacTokenizer, JiebaTokenizer

__name__ = 'autophrasex'
__version__ = '0.2.0'

logging.basicConfig(
    format="%(asctime)s %(levelname)7s %(filename)20s %(lineno)4d] %(message)s",
    level=logging.INFO
)
