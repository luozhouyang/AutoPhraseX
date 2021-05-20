import logging

from .autophrase import AutoPhrase
from .callbacks import (Callback, ConstantThresholdScheduler, EarlyStopping,
                        LoggingCallback, StateCallback)
from .extractors import (AbstractFeatureExtractor, AbstractNgramFiter,
                         EntropyExtractor, IDFExtractor, NgramsExtractor)
from .reader import (AbstractCorpusReadCallback, AbstractCorpusReader,
                     DefaultCorpusReader)
from .selector import (AbstractPhraseFilter, AbstractPhraseSelector,
                       DefaultPhraseSelector)
from .tokenizer import JiebaTokenizer

__name__ = 'autophrasex'
__version__ = '0.3.0'

logging.basicConfig(
    format="%(asctime)s %(levelname)7s %(filename)20s %(lineno)4d] %(message)s",
    level=logging.INFO
)
