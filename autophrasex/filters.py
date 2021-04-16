import abc

from LAC import LAC


class AbstractPhraseFilter(abc.ABC):

    def apply(self, phrase, **kwargs):
        """Filter phrase

        Args:
            phrase: Python tuple (phrase, freq)

        Returns:
            True if need to drop this phrase, else False
        """
        return False

    def batch_apply(self, batch_phrases, **kwargs):
        """Filter a batch of phrases.

        Args:
            batch_phrase: List of tuple (phrase, freq)

        Returns:
            candidates: Filtered List of phrase tuple (phrase, freq)
        """
        return batch_phrases


class PhraseFilterWrapper(AbstractPhraseFilter):

    def __init__(self, filters=None):
        super().__init__()
        self.filters = filters or []

    def apply(self, phrase, **kwargs):
        if any(f.apply(phrase) for f in self.filters):
            return True
        return False

    def batch_apply(self, batch_phrases, **kwargs):
        candidates = batch_phrases
        for f in self.filters:
            candidates = f.batch_apply(candidates)
        return candidates


class VerbPhraseFilter(AbstractPhraseFilter):
    """Use LAC to filter verb phrases."""

    def __init__(self, batch_size=100):
        """Init.

        Args:
            batch_size: Python integer, batch size to filter phrases, for better performance
        """
        super().__init__()
        self.lac = LAC()
        self.batch_size = batch_size

    def apply(self, batch_phrases, **kwargs):
        predictions = []
        for i in range(0, len(batch_phrases), self.batch_size):
            batch_texts = [x[0] for x in batch_phrases[i: i + self.batch_size]]
            batch_preds = self.lac.run(batch_texts)
            predictions.extend(batch_preds)
        candidates = []
        for i in range(len(predictions)):
            _, pos_tags = predictions[i]
            if any(pos in ['v', 'vn', 'vd'] for pos in pos_tags):
                continue
            candidates.append(batch_phrases[i])
        return candidates
