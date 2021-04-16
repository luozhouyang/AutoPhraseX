import abc

from .callbacks import EntropyCallback, IDFCallback, NgramsCallback


class AbstractFeatureComposer(abc.ABC):

    @abc.abstractmethod
    def compose(self, phrase, **kwargs):
        """Compose input features to classifier.

        Args: 
            phrase: Python str, ' ' joined ngrams. e.g '中国 北京'

        Returns:
            example: Python list, training example 
        """
        raise NotImplementedError()


class DefaultFeatureComposer(abc.ABC):

    def __init__(self,
                 idf_callback: IDFCallback,
                 ngrams_callbak: NgramsCallback,
                 entropy_callback: EntropyCallback):
        super().__init__()
        self.idf_callback = idf_callback
        self.ngrams_callback = ngrams_callbak
        self.entropy_callback = entropy_callback

    def compose(self, phrase, **kwargs):
        ngrams = phrase.split(' ')
        counter = self.ngrams_callback.ngrams_freq[len(ngrams)]
        freq = counter[' '.join(ngrams)] / sum(counter.values())

        features = {
            'unigram': 1 if len(ngrams) == 1 else 0,
            'term_freq': freq,
            'doc_freq': self.idf_callback.doc_freq_of(phrase) / self.idf_callback.n_docs,
            'idf': self.idf_callback.idf_of(phrase),
            'pmi': self.ngrams_callback.pmi_of(phrase),
            'le': self.entropy_callback.left_entropy_of(phrase),
            're': self.entropy_callback.right_entropy_of(phrase),
        }
        return self._convert_to_example(features)

    def _convert_to_example(self, features):
        example = []
        for k in ['unigram', 'term_freq', 'doc_freq', 'idf', 'pmi', 'le', 're']:
            example.append(features[k])
        return example
