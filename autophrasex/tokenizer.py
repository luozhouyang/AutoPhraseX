import os
import logging

from LAC import LAC


class BaiduLacTokenizer:

    def __init__(self, custom_vocab_path=None, model_path=None, mode='seg', use_cuda=False, **kwargs):
        """Initialize LAC.
        
        Args:
            custom_vocab_path: Path to customize vocabulary file for LAC
            model_path: Path of custom lac model. Optional.
            mode: Mode of LAC, one of ['seg', 'lac']
            use_cuda: Boolean, use GPU or not
        """
        self.lac = LAC(model_path=model_path, mode=mode, use_cuda=use_cuda)
        logging.info('LAC initialized successfully.')
        if custom_vocab_path:
            self.lac.load_customization(custom_vocab_path)
            logging.info('LAC load custom vocab successfully.')

    def tokenize(self, text, **kwargs):
        results = self.lac.run(text)
        return results


