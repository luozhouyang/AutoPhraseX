import abc

from . import utils
from .extractors import NgramsExtractor
from .filters import PhraseFilterWrapper


class AbstractPhraseSelector(abc.ABC):

    @abc.abstractmethod
    def select(self, **kwargs):
        raise NotImplementedError()


class DefaultPhraseSelector(AbstractPhraseSelector):
    """Frequent phrases selector."""

    def __init__(self,
                 ngrams_extractor: NgramsExtractor,
                 drop_stopwords=True,
                 min_freq=3,
                 min_len=2,
                 filters=None):
        """Init.
        Args:
            ngrams_extractor: Instance of NgramsExtractor
            drop_stopwords: Python boolean, filter stopwords or not.
            min_freq: Python int, min frequence of phrase occur in corpus.
            min_len: Python int, filter shot phrase whose length is less than this.
            filters: List of AbstractPhraseFilter, used to filter phrases

        """
        super().__init__()
        self.ngrams_extractor = ngrams_extractor
        self.drop_stopwords = drop_stopwords
        self.min_freq = min_freq
        self.min_len = min_len
        self.filter = PhraseFilterWrapper(filters=filters)

    def select(self, topk=300, filter_fn=None, **kwargs):
        """Select topk frequent phrases.

        Args:
            topk: Python int, max number of phrases to select.
            filter_fn: Python callable, use custom filters to select phrases, signature is filter_fn(phrase, freq) 

        Returns:
            phrases: Python list, selected frequent phrases from NgramsExtractor
        """
        candidates = []
        for n in range(1, self.ngrams_extractor.N + 1):
            counter = self.ngrams_extractor.ngrams_freq[n]
            for phrase, count in counter.items():
                # filter low freq phrase
                if count < self.min_freq:
                    continue
                # filter short phrase
                if len(phrase) < self.min_len:
                    continue
                # filter stopwords
                if self.drop_stopwords and utils.STOPWORDS.contains(''.join(phrase.split(' '))):
                    continue
                if filter_fn and filter_fn(phrase, count):
                    continue
                if self.filter.apply((phrase, count)):
                    continue
                candidates.append((phrase, count))

        candidates = self.filter.batch_apply(candidates)
        candidates = sorted(candidates, key=lambda x: x[1], reverse=True)
        phrases = [x[0] for x in candidates[:topk]]
        return phrases
