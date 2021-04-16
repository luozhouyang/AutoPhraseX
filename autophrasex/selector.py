import abc

from . import utils
from .extractors import NgramsExtractor


class AbstractPhraseSelector(abc.ABC):

    @abc.abstractmethod
    def select(self, **kwargs):
        raise NotImplementedError()


class DefaultPhraseSelector(AbstractPhraseSelector):
    """Frequent phrases selector."""

    def __init__(self, ngrams_extractor: NgramsExtractor):
        super().__init__()
        self.ngrams_extractor = ngrams_extractor

    def select(self, topk=300, drop_stopwords=True, min_freq=3, min_len=2, drop_verb=False, filter_fn=None, **kwargs):
        """Select topk frequent phrases.

        Args:
            topk: Python int, max number of phrases to select.
            drop_stopwords: Python boolean, filter stopwords or not.
            min_freq: Python int, min frequence of phrase occur in corpus.
            min_len: Python int, filter shot phrase whose length is less than this.
            drop_verb: Python boolean, drop verb phrase or not.
            filter_fn: Python callable, use custom filters to select phrases, signature is filter_fn(phrase, freq) 

        Returns:
            phrases: Python list, selected frequent phrases from NgramsExtractor
        """
        candidates = []
        for n in range(1, self.ngrams_extractor.N + 1):
            counter = self.ngrams_extractor.ngrams_freq[n]
            for phrase, count in counter.items():
                # filter low freq phrase
                if count < min_freq:
                    continue
                # filter short phrase
                if len(phrase) < min_len:
                    continue
                # filter stopwords
                if drop_stopwords and utils.STOPWORDS.contains(''.join(phrase.split(' '))):
                    continue
                if filter_fn and filter_fn(phrase, count):
                    continue
                candidates.append((phrase, count))

        # drop verbs in batch for better performance
        if drop_verb:
            candidates = self._drop_verbs(candidates)
        candidates = sorted(candidates, key=lambda x: x[1], reverse=True)
        phrases = [x[0] for x in candidates[:self.phrase_max_count]]
        return phrases

    def _drop_verbs(self, candidates):
        from LAC import LAC
        lac = LAC()
        predictions = []
        for i in range(0, len(candidates), 100):
            # batch_count = [x[1] for x in candidates[i:i+100]]
            batch_texts = [x[0] for x in candidates[i:i+100]]
            batch_preds = lac.run(batch_texts)
            predictions.extend(batch_preds)
        filtered_candidates = []
        for i in range(len(predictions)):
            _, pos_tags = predictions[i]
            if any(pos in ['v', 'vn', 'vd'] for pos in pos_tags):
                continue
            filtered_candidates.append(candidates[i])
        return filtered_candidates
