import abc
import logging
import os

from . import utils
from .tokenizer import AbstractTokenizer


class AbstractCorpusReadCallback(abc.ABC):

    def on_process_doc_begin(self):
        """Starting to process a doc"""
        pass

    def update_tokens(self, tokens, **kwargs):
        """Process tokens, tokenized from current doc.

        Args:
            tokens: List of string, all tokens tokenized from doc
        """
        pass

    def update_ngrams(self, start, end, ngram, n, **kwargs):
        """Process ngrams.

        Args:
            start: Python integer, start index of this ngram in the whole token list
            end: Python integer, end index of this ngram in the whole token list
            ngram: Python tuple, ngram tokens
            n: Python integer, N of n-grams
        """
        pass

    def on_process_doc_end(self):
        """Finished to process a document"""
        pass


class AbstractCorpusReader(abc.ABC):

    @abc.abstractmethod
    def read(self, corpus_files, *args, **kwargs):
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

    def __init__(self, tokenizer: AbstractTokenizer):
        super().__init__()
        self.tokenizer = tokenizer

    def read(self, corpus_files, extractor, N=4, verbose=True, logsteps=100, **kwargs):

        def read_line(line):
            # callbacks process doc begin
            extractor.on_process_doc_begin()
            tokens = self.tokenizer.tokenize(line, **kwargs)
            # callbacks process tokens
            extractor.update_tokens(tokens, **kwargs)
            # callbacks process ngrams
            for n in range(1, N + 1):
                for (start, end), window in utils.ngrams(tokens, n=n):
                    extractor.update_ngrams(start, end, window, n, **kwargs)
            # callbacks process doc end
            extractor.on_process_doc_end()

        read_corpus_files(corpus_files, callback=read_line, verbose=verbose, logsteps=logsteps, **kwargs)
