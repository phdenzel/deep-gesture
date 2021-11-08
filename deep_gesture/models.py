"""
deep_gesture.models

@author: phdenzel
"""
import os
import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import Input
import tensorflow.keras.optimizers
import tensorflow.keras.layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
import deep_gesture as dg


class ModelConfigurator(object):

    lstm3_conn = [('LSTM',  64, 'relu'),
                  ('LSTM', 128, 'relu'),
                  ('LSTM',  64, 'relu'),
                  ('Dropout', 0.2, None),
                  ('Dense', 64, 'relu'),
                  ('Dense', 32, 'relu'),
                  ('Dense', 0, 'softmax')]
    """
    """
    def __init__(self, features=None, labels=None,
                 categorical=False, random_state=42):
        """
        """
        self.categorical = categorical
        self.random_state = random_state
        self.load_data(features, labels)
        self.configure()

    def load_data(self, features, labels, categorical=None):
        """
        """
        if categorical is not None:
            self.categorical = categorical
        self.features = np.asarray(features)
        self.labels = np.asarray(labels)
        self.label_set = set(self.labels) if np.any(self.labels.shape) else []
        self.label_map = self.create_label_map(
            self.labels, label_set=self.label_set)
        self.test_train_split()

    def roll_filename(self, directory=None):
        """
        """
        directory = dg.DOT_DIR if directory is None else directory
        filename = os.path.join(directory, '_'.join(
            dg.utils.generate_filename(extension='').split('_')[:-1]))
        return filename
        

    def configure(self, **kwargs):
        """
        """
        for (conf, dval) in [('layer_specs', self.lstm3_conn), ('optimizer', 'Adam'),
                             ('learning_rate', 0.001), ('loss', 'categorical_crossentropy'),
                             ('metrics', ['categorical_accuracy']), ('epochs', 1000),
                             ('batch_size', None), ('validation_split', 0.2),
                             ('validation_data', None),
                             ('validation', {'accuracy': None, 'confm': None}),
                             ('filename', self.roll_filename())]:
            default = dval if not hasattr(self, conf) else self.__getattribute__(conf)
            setattr(self, conf, kwargs.get(conf, default))
        return self

    @property
    def configs(self):
        return {
            'filename': self.filename,
            'layer_specs': self.layer_specs,
            'optimizer': self.optimizer,
            'learning_rate': self.learning_rate,
            'loss': self.loss,
            'metrics': self.metrics,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'validation_split': self.validation_split,
            'validation_data': self.validation_data,
            'validation': self.validation,
        }

    @staticmethod
    def create_label_map(labels, label_set=None):
        """
        Map labels to numbers for modelling
        
        Args:
            labels <list> - a list of all labels

        Return:
            label_map <dict> - labels -> index (int)
            labels_mapped <list> - mapped index labels
        """
        unique_labels = set(labels) if label_set is None else label_set
        label_map = {}
        for idx, label in enumerate(unique_labels):
            label_map[label] = idx
        return label_map

    @property
    def labels_as_int(self):
        if np.any(self.labels.shape):
            return np.array([self.label_map[l] for l in self.labels])
        return np.array(self.labels)

    @property
    def X(self):
        return self.features

    @property
    def y(self):
        y = self.labels_as_int
        if self.categorical:
            y = to_categorical(y).astype(int)
        return y

    @property
    def X_train(self):
        if not hasattr(self, 'train_data'):
            self.test_train_split()
        return self.train_data[0]

    @property
    def y_train(self):
        if not hasattr(self, 'train_data'):
            self.test_train_split()
        return self.train_data[1]

    @property
    def X_test(self):
        if not hasattr(self, 'test_data'):
            self.test_train_split()
        return self.test_data[0]

    @property
    def y_test(self):
        if not hasattr(self, 'test_data'):
            self.test_train_split()
        return self.test_data[1]

    def test_train_split(self, test_size=0.33, random_state=None):
        """
        """
        if random_state is None:
            random_state = self.random_state
        if np.any(self.X.shape) and np.any(self.y.shape):
            X_train, X_test, y_train, y_test = train_test_split(
                self.X, self.y, test_size=test_size, random_state=random_state)
            self.train_data = X_train, y_train
            self.test_data = X_test, y_test

    def sequential(self, **kwargs):
        """
        Simple, generic generator for Sequential models

        Kwargs:
            layer_specs <list> - tuples of (layer_type, nodes, activation)
            optimizer <str> - appropriate optimizer function name
            loss <str> - appropriate loss function name
            metrics <list(str)> - appropriate metrics
            verbose <bool> - print information to stdout

        Return:
            model <keras.tensorflow.models.Sequential>
        """
        verbose = kwargs.pop('verbose', False)
        self.configure(**kwargs)
        model = Sequential()
        n_layers = len(self.layer_specs)
        layer_types = [l[0] for l in self.layer_specs]
        input_shape = self.X.shape[1:]
        output_shape = self.y.shape[-1]
        for i, (layer_type_str, nodes, activation) in enumerate(self.layer_specs):
            layer_type = getattr(tensorflow.keras.layers, layer_type_str)
            if i == 0:
                model.add(Input(shape=input_shape))
            args = (output_shape,) if i == n_layers-1 else (nodes,)
            kw = dict(activation=activation)
            if layer_type_str == 'LSTM':
                kw['return_sequences'] = True
            if layer_type_str == 'Dropout':
                kw.pop('activation')
            if i == n_layers-layer_types[::-1].index('LSTM')-1:
                kw['return_sequences'] = False
            model.add(layer_type(*args, **kw))
        opt = getattr(tensorflow.keras.optimizers, self.optimizer)(
            learning_rate=self.learning_rate)
        model.compile(optimizer=opt, loss=self.loss, metrics=self.metrics)
        if verbose:
            model.summary()
        return model

    def generate(self, **kwargs):
        """
        Generate a Tensorflow model with the current configurations

        Kwargs:
            layer_specs <list> - tuples of (layer_type, nodes, activation)
            optimizer <str> - appropriate optimizer function name
            loss <str> - appropriate loss function name
            metrics <list(str)> - appropriate metrics
            verbose <bool> - print information to stdout

        Return:
            model <keras.tensorflow.models.Sequential>
        """
        self.model = self.sequential(**kwargs)
        return self.model

    def train_model(self, epochs=None, batch_size=None, validation_split=None,
                    callbacks=[],
                    checkpoint_callback=True,
                    tensor_board_callback=False):
        """
        """
        epochs = self.epochs if epochs is None else epochs
        self.epochs = epochs
        batch_size = self.batch_size if batch_size is None else batch_size
        self.batch_size = batch_size
        validation_split = self.validation_split if validation_split is None else validation_split
        self.validation_split = validation_split
        
        if checkpoint_callback:
            cp_callback = ModelCheckpoint(self.filename+'_model_cp.h5',
                                          monitor='val_'+self.metrics[0],
                                          save_best_only=True, mode='max')
            
            callbacks.append(cp_callback)
        if tensor_board_callback:
            log_dir = dg.LOG_DIR
            dg.utils.mkdir_p(log_dir)
            tb_callback = TensorBoard(log_dir=log_dir)
            callbacks.append(tb_callback)
        if not hasattr(self, 'model'):
            self.generate()
        self.model.fit(self.X_train, self.y_train,
                       epochs=epochs, batch_size=batch_size,
                       validation_split=validation_split,
                       callbacks=callbacks)
        if os.path.isfile(self.filename+'_model_cp.h5'):
            self.model.load_weights(self.filename+'_model_cp.h5')
        

    def save(self, directory=None):
        """
        Save configs and model

        Kwargs:
            directory <str> - directory where the files are saved
        """
        with open(self.filename+'_configs.json', 'w') as f:
            json.dump(self.configs, f)
        if not os.path.isfile(self.filename+'_model.h5'):
            self.model.save(self.filename+'_model.h5')

    def validate(self, verbose=False):
        """
        """
        y_hat = self.model.predict(self.X_test)
        y_hat = np.argmax(y_hat, axis=1).tolist()
        y_true = np.argmax(self.y_test, axis=1).tolist()
        self.validation['accuracy'] = accuracy_score(y_true, y_hat)
        self.validation['confm'] = \
            multilabel_confusion_matrix(y_true, y_hat).tolist()
        if verbose:
            print("Accuracy:        \t{}".format(self.validation['accuracy']))
            print("Confusion matrix:\t{}".format(self.validation['confm']))


def lstm3_conn():
    features, labels = dg.utils.load_data(dg.DATA_DIR, extract_from_tar=True)
    dg_mc = ModelConfigurator(features, labels, categorical=True, random_state=16)
    dg_mc.test_train_split(test_size=0.1)
    dg_mc.configure(optimizer='Adam', learning_rate=0.0001).generate(verbose=True)
    # dg_mc.configure(optimizer='SGD', learning_rate=0.001).generate(verbose=True)
    dg_mc.train_model(epochs=1500, batch_size=32, validation_split=0.1,
                      checkpoint_callback=True, tensor_board_callback=True)
    dg_mc.validate(verbose=True)
    dg_mc.save()


if __name__ == "__main__":
    lstm3_conn()
