import abc
import logging


class Callback(abc.ABC):

    def begin(self):
        pass

    def on_read_corpus_begin(self):
        pass

    def on_read_corpus_end(self):
        pass

    def on_build_quality_phrases_begin(self, quality_file):
        pass

    def on_build_quality_phrases_end(self, quality_phrases):
        pass

    def on_select_frequent_phrases_begin(self):
        pass

    def on_select_frequent_phrases_end(self, frequent_phrases):
        pass

    def on_organize_phrase_pools_begin(self, quality_phrases, frequent_phrases):
        pass

    def on_organize_phrase_pools_end(self, pos_pool, neg_pool):
        pass

    def on_epoch_begin(self, epoch):
        pass

    def on_epoch_prepare_training_data_begin(self, epoch):
        pass

    def on_epoch_prepare_training_data_end(self, epoch, x, y):
        pass

    def on_epoch_reorganize_phrase_pools_begin(self, epoch, pos_pool, neg_pool):
        pass

    def on_epoch_reorganize_phrase_pools_end(self, epoch, pos_pool, neg_pool):
        pass

    def on_epoch_end(self, epoch):
        pass

    def on_predict_neg_pool_begin(self, neg_pool):
        pass

    def on_predict_neg_pool_end(self, predictions):
        pass

    def end(self):
        pass


class CallbackWrapper(Callback):

    def __init__(self, callbacks=None):
        super().__init__()
        self.callbacks = callbacks or []

    def begin(self):
        for cb in self.callbacks:
            cb.begin()

    def on_read_corpus_begin(self):
        for cb in self.callbacks:
            cb.on_read_corpus_begin()

    def on_read_corpus_end(self):
        for cb in self.callbacks:
            cb.on_read_corpus_end()

    def on_build_quality_phrases_begin(self, quality_file):
        for cb in self.callbacks:
            cb.on_build_quality_phrases_begin(quality_file)

    def on_build_quality_phrases_end(self, quality_phrases):
        for cb in self.callbacks:
            cb.on_build_quality_phrases_end(quality_phrases)

    def on_select_frequent_phrases_begin(self):
        for cb in self.callbacks:
            cb.on_select_frequent_phrases_begin()

    def on_select_frequent_phrases_end(self, frequent_phrases):
        for cb in self.callbacks:
            cb.on_select_frequent_phrases_end(frequent_phrases)

    def on_organize_phrase_pools_begin(self, quality_phrases, frequent_phrases):
        for cb in self.callbacks:
            cb.on_organize_phrase_pools_begin(quality_phrases, frequent_phrases)

    def on_organize_phrase_pools_end(self, pos_pool, neg_pool):
        for cb in self.callbacks:
            cb.on_organize_phrase_pools_end(pos_pool, neg_pool)

    def on_epoch_begin(self, epoch):
        for cb in self.callbacks:
            cb.on_epoch_begin(epoch)

    def on_epoch_prepare_training_data_begin(self, epoch):
        for cb in self.callbacks:
            cb.on_epoch_prepare_training_data_begin(epoch)

    def on_epoch_prepare_training_data_end(self, epoch, x, y):
        for cb in self.callbacks:
            cb.on_epoch_prepare_training_data_end(epoch, x, y)

    def on_epoch_reorganize_phrase_pools_begin(self, epoch, pos_pool, neg_pool):
        for cb in self.callbacks:
            cb.on_epoch_reorganize_phrase_pools_begin(epoch, pos_pool, neg_pool)

    def on_epoch_reorganize_phrase_pools_end(self, epoch, pos_pool, neg_pool):
        for cb in self.callbacks:
            cb.on_epoch_reorganize_phrase_pools_end(epoch, pos_pool, neg_pool)

    def on_epoch_end(self, epoch):
        for cb in self.callbacks:
            cb.on_epoch_end(epoch)

    def on_predict_neg_pool_begin(self, neg_pool):
        for cb in self.callbacks:
            cb.on_predict_neg_pool_begin(neg_pool)

    def on_predict_neg_pool_end(self, predictions):
        for cb in self.callbacks:
            cb.on_predict_neg_pool_end(predictions)

    def end(self):
        for cb in self.callbacks:
            cb.end()


class LoggingCallback(Callback):

    def begin(self):
        logging.info('Starting to mining phrases...')

    def on_read_corpus_begin(self):
        logging.info('Starting to read corpus...')

    def on_read_corpus_end(self):
        logging.info('Finished to read corpus.')

    def on_build_quality_phrases_begin(self, quality_file):
        pass

    def on_build_quality_phrases_end(self, quality_phrases):
        logging.info('Build quality phrases done!')
        logging.info('    size of quality phrases: %d', len(quality_phrases))

    def on_select_frequent_phrases_begin(self):
        pass

    def on_select_frequent_phrases_end(self, frequent_phrases):
        logging.info('Select frequent phrases done!')
        logging.info('    size of frequent phrases: %d', len(frequent_phrases))

    def on_organize_phrase_pools_begin(self, quality_phrases, frequent_phrases):
        pass

    def on_organize_phrase_pools_end(self, pos_pool, neg_pool):
        logging.info('Organize inital phrase pools done!')
        logging.info('    size of initial positive pool: %d', len(pos_pool))
        logging.info('    size of initial negative pool: %d', len(neg_pool))

    def on_epoch_begin(self, epoch):
        logging.info('Starting epoch: %d', epoch)

    def on_epoch_prepare_training_data_begin(self, epoch):
        pass

    def on_epoch_prepare_training_data_end(self, epoch, x, y):
        logging.info('    prepared training data done, size of examples: %d', len(x))

    def on_epoch_reorganize_phrase_pools_begin(self, epoch, pos_pool, neg_pool):
        pass

    def on_epoch_reorganize_phrase_pools_end(self, epoch, pos_pool, neg_pool):
        logging.info('    reorganize phrase pools done!')
        logging.info('    size of positive pool: %d', len(pos_pool))
        logging.info('    size of negative pool: %d', len(neg_pool))

    def on_epoch_end(self, epoch):
        logging.info('Finished epoch: %d', epoch)

    def on_predict_neg_pool_begin(self, neg_pool):
        pass

    def on_predict_neg_pool_end(self, predictions):
        pass

    def end(self):
        logging.info('Finished to mine phrases. Done!')


class StateCallback(Callback):

    def __init__(self, autophrase, **kwargs):
        super().__init__()
        self.autophrase = autophrase


class ConstantThresholdScheduler(StateCallback):

    def __init__(self, autophrase, **kwargs):
        super().__init__(autophrase)

    def on_epoch_begin(self, epoch):
        pass

    def on_epoch_end(self, epoch):
        pass


class EarlyStopping(StateCallback):

    def __init__(self, autophrase, patience=1, min_delta=3, **kwargs):
        super().__init__(autophrase)
        self.patience = patience
        self.min_delta = min_delta
        self.prev_pos_pool_size = 0
        self.curr_pos_pool_size = 0

    def on_epoch_reorganize_phrase_pools_begin(self, epoch, pos_pool, neg_pool):
        self.prev_pos_pool_size = len(pos_pool)

    def on_epoch_reorganize_phrase_pools_end(self, epoch, pos_pool, neg_pool):
        self.curr_pos_pool_size = len(pos_pool)
        if self.curr_pos_pool_size - self.prev_pos_pool_size < self.min_delta:
            self.patience -= 1
        if self.patience == 0:
            self.autophrase.early_stop = True
