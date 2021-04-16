import abc
import logging
import os

from . import utils
from .extractors import (EntropyExtractor, ExtractorCallbackWrapper,
                         IDFExtractor, NgramsExtractor)
from .tokenizer import AbstractTokenizer


class AbstractCorpusReader(abc.ABC):

    @abc.abstractmethod
    def read(self, corpus_files, **kwargs):
        raise NotImplementedError()


def read_corpus_files(input_files, callback, verbose=True, logsteps=100, **kwargs):
    if isinstance(input_files, str):
        input_files = [input_files]
    count = 0
    for f in input_files:
        if not os.path.exists(f):
            logging.warning('File: %s does not exist. Skipped.', f)
            continue
        with open(f, mode='rt', encoding='utf-8') as fin:
            for line in fin:
                line = line.rstrip('\n')
                if not line:
                    continue
                if callback:
                    callback(line)

                count += 1
                if verbose and count % logsteps == 0:
                    logging.info('Processes %d lines.', count)
        logging.info('Finished to process file: %s', f)
    logging.info('Done! Processed %d lines in total.', count)


class DefaultCorpusReader(AbstractCorpusReader):

    def __init__(self, tokenizer: AbstractTokenizer, extractors=None):
        super().__init__()
        self.extractor = ExtractorCallbackWrapper(extractors=extractors)
        self.tokenizer = tokenizer

    def read(self, corpus_files, N=4, verbose=True, logsteps=100, **kwargs):

        def read_line(line):
            # callbacks process doc begin
            self.extractor.on_process_doc_begin()
            tokens = self.tokenizer.tokenize(line, **kwargs)
            # callbacks process tokens
            self.extractor.update_tokens(tokens, **kwargs)
            # callbacks process ngrams
            for n in range(1, N + 1):
                for (start, end), window in utils.ngrams(tokens, n=n):
                    self.extractor.update_ngrams(start, end, window, n, **kwargs)
            # callbacks process doc end
            self.extractor.on_process_doc_end()

        read_corpus_files(corpus_files, callback=read_line, verbose=verbose, logsteps=logsteps, **kwargs)
