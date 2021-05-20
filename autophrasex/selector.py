import abc
import logging

from . import utils
from .extractors import NgramsExtractor


class AbstractPhraseFilter(abc.ABC):

    def apply(self, pair, **kwargs):
        """Filter phrase

        Args:
            pair: Python tuple of (phrase, freq)

        Returns:
            True if need to drop this phrase, else False
        """
        return False

    def batch_apply(self, batch_pairs, **kwargs):
        """Filter a batch of phrases.

        Args:
            batch_phrase: List of tuple (phrase, freq)

        Returns:
            candidates: Filtered List of phrase tuple (phrase, freq)
        """
        return batch_pairs


class PhraseFilterWrapper(AbstractPhraseFilter):

    def __init__(self, phrase_filters=None):
        super().__init__()
        self.filters = phrase_filters or []

    def apply(self, pair, **kwargs):
        if any(f.apply(pair) for f in self.filters):
            return True
        return False

    def batch_apply(self, batch_pairs, **kwargs):
        candidates = batch_pairs
        for f in self.filters:
            candidates = f.batch_apply(candidates)
        return candidates


class DefaultPhraseFilter(AbstractPhraseFilter):

    def __init__(self, min_len=2, min_freq=3, drop_stopwords=True) -> None:
        super().__init__()
        self.min_len = min_len
        self.min_freq = min_freq
        self.drop_stopwords = drop_stopwords

    def apply(self, pair, **kwargs):
        phrase, freq = pair
        if freq < self.min_freq:
            return True
        if len(phrase) < self.min_len:
            return True
        if self.drop_stopwords and utils.STOPWORDS.contains(''.join(phrase.split(' '))):
            return True
        return False


class AbstractPhraseSelector(abc.ABC):

    @abc.abstractmethod
    def select(self, **kwargs):
        raise NotImplementedError()


class DefaultPhraseSelector(AbstractPhraseSelector):
    """Frequent phrases selector."""

    def __init__(self,
                 phrase_filters=None,
                 use_default_phrase_filters=True,
                 **kwargs):
        """Init.
        Args:
            drop_stopwords: Python boolean, filter stopwords or not.
            min_freq: Python int, min frequence of phrase occur in corpus.
            min_len: Python int, filter shot phrase whose length is less than this.
            filters: List of AbstractPhraseFilter, used to filter phrases

        """
        super().__init__()
        if not phrase_filters and use_default_phrase_filters:
            self.phrase_filter = PhraseFilterWrapper(phrase_filters=[
                DefaultPhraseFilter(
                    min_len=kwargs.get('min_len', 2),
                    min_freq=kwargs.get('min_freq', 3),
                    drop_stopwords=kwargs.get('drop_stopwords', True))
            ])
            logging.info('Using default phrase filters.')
        else:
            self.phrase_filter = PhraseFilterWrapper(phrase_filters)

    def select(self, extractors, topk=300, **kwargs):
        """Select topk frequent phrases.

        Args:
            extractors: List of AbstractFeatureExtractor, used to select frequent phrases
            topk: Python int, max number of phrases to select.

        Returns:
            phrases: Python list, selected frequent phrases from NgramsExtractor
        """
        ngrams_extractor = None
        for e in extractors:
            if isinstance(e, NgramsExtractor):
                ngrams_extractor = e
                break
        if ngrams_extractor is None:
            raise ValueError('Must provide an instance of NgramsExtractor!')

        candidates = []
        for n in range(1, ngrams_extractor.N + 1):
            counter = ngrams_extractor.ngrams_freq[n]
            for phrase, count in counter.items():
                if self.phrase_filter.apply((phrase, count)):
                    continue
                candidates.append((phrase, count))

        candidates = self.phrase_filter.batch_apply(candidates)
        candidates = sorted(candidates, key=lambda x: x[1], reverse=True)
        phrases = [x[0] for x in candidates[:topk]]
        return phrases
