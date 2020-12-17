import logging

from .autophrase import AutoPhrase
from .callbacks import (AbstractCallback, EntropyCallback, IDFCallback,
                        NgramsCallback)
from .strategy import AbstractStrategy, Strategy
from .tokenizer import BaiduLacTokenizer, JiebaTokenizer

__name__ = 'autophrasex'
__version__ = '0.1.0'

logging.basicConfig(
    format="%(asctime)s %(levelname)7s %(filename)20s %(lineno)4d] %(message)s",
    level=logging.INFO
)
