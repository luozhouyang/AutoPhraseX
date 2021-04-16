import abc

from .extractors import EntropyExtractor, IDFExtractor, NgramsExtractor


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
                 idf_extractor: IDFExtractor,
                 ngrams_extractor: NgramsExtractor,
                 entropy_extractor: EntropyExtractor):
        super().__init__()
        self.idf_extractor = idf_extractor
        self.ngrams_extractor = ngrams_extractor
        self.entropy_extractor = entropy_extractor

    def compose(self, phrase, **kwargs):
        ngrams = phrase.split(' ')
        counter = self.ngrams_extractor.ngrams_freq[len(ngrams)]
        freq = counter[' '.join(ngrams)] / sum(counter.values())

        features = {
            'unigram': 1 if len(ngrams) == 1 else 0,
            'term_freq': freq,
            'doc_freq': self.idf_extractor.doc_freq_of(phrase) / self.idf_extractor.n_docs,
            'idf': self.idf_extractor.idf_of(phrase),
            'pmi': self.ngrams_extractor.pmi_of(phrase),
            'le': self.entropy_extractor.left_entropy_of(phrase),
            're': self.entropy_extractor.right_entropy_of(phrase),
        }
        return self._convert_to_example(features)

    def _convert_to_example(self, features):
        example = []
        for k in ['unigram', 'term_freq', 'doc_freq', 'idf', 'pmi', 'le', 're']:
            example.append(features[k])
        return example
