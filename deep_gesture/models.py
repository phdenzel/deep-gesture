"""
deep_gesture.models

@author: phdenzel
"""
import os
import shutil
import json
import pprint
import numpy as np
from itertools import product as iter_prod
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import multilabel_confusion_matrix
import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import TensorBoard
tf.get_logger().setLevel('ERROR')
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import deep_gesture as dg


class ModelConfigurator(object):
    """
    """
    lstm3_conn = [('LSTM',  64, 'relu'),
                  ('LSTM', 128, 'relu'),
                  ('LSTM',  64, 'relu'),
                  # ('Dropout', 0.2, None),
                  ('Dense', 64, 'relu'),
                  ('Dense', 32, 'relu'),
                  ('Dense', 0, 'softmax')]
    def __init__(self, features=None, labels=None,
                 label_map=None,
                 categorical=False, random_state=42):
        """
        Kwargs:
            features <list/np.ndarray> -
            labels <list/np.ndarray> -
            categorical <bool> -
            random_state <int> -
        """
        self.categorical = categorical
        self.random_state = random_state
        self.label_map = label_map
        self.load_data(features, labels)
        self.configure()

    @classmethod
    def from_json(cls, json_file,
                  features=None, labels=None,
                  generate=True,
                  categorical=False, random_state=42,
                  verbose=False):
        """
        """
        with open(json_file) as f:
            kwargs = json.load(f)
            if verbose:
                print("Loading ", json_file)
        self = cls(features=features, labels=labels,
                   categorical=categorical, random_state=random_state)
        if generate:
            self.configure(**kwargs).generate()
        else:
            self.configure(**kwargs)
        return self

    @classmethod
    def from_archive(cls, model_name=None, from_checkpoints=False,
                     verbose=False, **kwargs):
        mdl_dir = dg.MDL_DIR
        if model_name is None:
            models = os.listdir(mdl_dir)
            models = models if len(models) > 0 else [None]
            model_name = 'dg' if 'dg' in models else models[-1]
        if model_name is None:
            print("Please first build and train a model!")
        model_path = os.path.join(mdl_dir, model_name)
        jsn = [f for f in os.listdir(model_path) if f.endswith('json')][0]
        jsn_path = os.path.join(model_path, jsn)
        self = cls.from_json(jsn_path, generate=False, **kwargs)
        self.load_model(from_checkpoints=from_checkpoints,
                        verbose=verbose)
        return self

    def load_data(self, features, labels, categorical=None):
        """
        Args:
            features <list/np.ndarray> -
            labels <list/np.ndarray> -

        Kwargs:
            categorical <bool> -
        """
        if categorical is not None:
            self.categorical = categorical
        self.features = np.asarray(features)
        self.labels = np.asarray(labels)
        self.label_set = set(self.labels) if np.any(self.labels.shape) else []
        if self.label_map is None:
            self.label_map = self.create_label_map(
                self.labels, label_set=self.label_set)
        self.test_train_split()

    @staticmethod
    def create_label_map(labels, label_set=None):
        """
        Map labels to numbers for modelling
        
        Args:
            labels <list> - a list of all labels

        Return:
            label_map <dict> - labels (str) -> index (int)
        """
        labels = [None] if not labels else labels
        unique_labels = set(labels) if label_set is None else label_set
        label_map = {}
        for idx, label in enumerate(unique_labels):
            label_map[label] = idx
        return label_map

    @property
    def inv_label_map(self):
        """
        Reversed label map to map numbers to labels (for inference)

        Return:
            inv_label_map <dict> - index (int) -> labels (str)
        """
        if self.label_map is not None:
            return {self.label_map[k]: k for k in self.label_map}

    def roll_basename(self, directory=None, randomize=True):
        """
        Kwargs:
            directory <str> -
        """
        directory = dg.MDL_DIR if directory is None else directory
        name_id = os.urandom(self.random_state) if randomize else None
        basename = os.path.join(directory, '_'.join(
            dg.utils.generate_filename(name_id=name_id,
                                       extension='').split('_')[:-1]))
        return basename

    def configure(self, **kwargs):
        """
        Kwargs:
            basename <str> -
            layer_specs <list(tuple)> -
            optimizer <str> -
            learning_rate <float> -
            loss <str> -
            metrics <list(str)> -
            epochs <int> -
            batch_size <int> -
            validation_split <float> -
            validation_data <tuple(np.ndarray)> -
            validation <dict> -
        """
        for (conf, dval) in [('layer_specs', self.lstm3_conn), ('optimizer', 'Adam'),
                             ('learning_rate', 1e-3), ('loss', 'categorical_crossentropy'),
                             ('metrics', ['categorical_accuracy']), ('epochs', 1000),
                             ('batch_size', None), ('validation_split', 0.2),
                             ('validation_data', None),
                             ('validation', {'accuracy': None, 'confm': None}),
                             ('basename', self.roll_basename()),
                             ('label_map', self.create_label_map(self.labels))]:
            default = dval if not hasattr(self, conf) else self.__getattribute__(conf)
            setattr(self, conf, kwargs.get(conf, default))
        return self

    @property
    def configs(self):
        return {
            'basename': self.basename,
            'label_map': self.label_map,
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
        Kwargs:
            test_size <float> -
            random_state <int> -
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
            model <tf.keras.models.Sequential>
        """
        verbose = kwargs.pop('verbose', False)
        self.configure(**kwargs)
        model = Sequential()
        n_layers = len(self.layer_specs)
        layer_types = [l[0] for l in self.layer_specs]
        input_shape = self.X.shape[1:]
        output_shape = self.y.shape[-1]
        for i, (layer_type_str, nodes, activation) in enumerate(self.layer_specs):
            layer_type = getattr(tf.keras.layers, layer_type_str)
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
        opt = getattr(tf.keras.optimizers, self.optimizer)(
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
            model <tf.keras.models.Sequential>
        """
        self.model = self.sequential(**kwargs)
        return self.model

    def train_model(self, epochs=None, batch_size=None, validation_split=None,
                    callbacks=[],
                    checkpoint_callback=False,
                    earlystop_callback=False,
                    tensor_board_callback=False,
                    archive=True,
                    save=False,
                    verbose=False):
        """
        Kwargs:
            epochs <int> -
            batch_size <int> -
            validation_split <float> -
            callbacks <list> -
            checkpoint_callback <bool> -
            earlystop_callback <bool> -
            tensor_board_callback <bool> -
            save <bool> -
            verbose <bool> -
        """
        epochs = self.epochs if epochs is None else epochs
        self.epochs = epochs
        batch_size = self.batch_size if batch_size is None else batch_size
        self.batch_size = batch_size
        validation_split = self.validation_split if validation_split is None else validation_split
        self.validation_split = validation_split
        log_dir = dg.LOG_DIR
        dg.utils.mkdir_p(log_dir)
        mdl_dir = os.path.dirname(self.basename)
        dg.utils.mkdir_p(mdl_dir)

        # callbacks
        if checkpoint_callback:
            cp_callback = ModelCheckpoint(self.basename+'_model{epoch:04d}_cp.h5',
                                          monitor='val_'+self.metrics[0],
                                          save_best_only=True, mode='max',
                                          verbose=verbose)
            
            callbacks.append(cp_callback)
        if earlystop_callback:
            es_callback = EarlyStopping(monitor='val_'+self.metrics[0],
                                        min_delta=0.1, patience=25,
                                        restore_best_weights=False)
            callbacks.append(es_callback)
        if tensor_board_callback:
            tb_callback = TensorBoard(log_dir=log_dir)
            callbacks.append(tb_callback)
        if not hasattr(self, 'model'):
            self.generate()

        # fit model
        self.model.fit(self.X_train, self.y_train,
                       epochs=epochs, batch_size=batch_size,
                       validation_split=validation_split,
                       callbacks=callbacks)
        # prevent memory leak? -TODO
        for c in callbacks:
            del c
        # archive model+log files and remove non-relevant files
        self.basename = self.archive_model(mdl_dir=mdl_dir, log_dir=log_dir)
        if save:
            self.save()
        self.load_model()  # redundant

    def archive_model(self, mdl_dir=None, log_dir=None, intermediate_checkpoints=True):
        """
        Kwargs:
            mdl_dir <str> -
            log_dir <str> -
            intermediate_checkpoints <bool> -
        """
        mdl_dir = os.path.dirname(self.basename) if mdl_dir is None else mdl_dir
        log_dir = dg.LOG_DIR if log_dir is None else log_dir
        # handle checkpoint files
        cp_files = sorted([fmdl for fmdl in os.listdir(mdl_dir)
                           if fmdl.endswith('cp.h5')])
        if not intermediate_checkpoints:
            for f in cp_files[:-1]:
                os.remove(os.path.join(mdl_dir, f))
        for f in cp_files[-1:]:
            shutil.copy2(os.path.join(mdl_dir, f), self.basename+'_model_cp.h5')
        # move models into archive directory
        mdl_name = self.basename.split('_')[4] + f'_{self.optimizer}'
        mdl_files = [f for f in os.listdir(mdl_dir)
                     if os.path.isfile(os.path.join(mdl_dir, f))]
        dg.utils.mkdir_p(os.path.join(mdl_dir, mdl_name))
        for f in mdl_files:
            fsrc = os.path.join(mdl_dir, f)
            fdst = os.path.join(os.path.join(mdl_dir, mdl_name), f)
            os.rename(fsrc, fdst)
        # archive log directory
        mdl_log_dir = os.path.join(os.path.join(mdl_dir, mdl_name),
                                   os.path.basename(log_dir))
        shutil.copytree(log_dir, mdl_log_dir)
        shutil.rmtree(log_dir)
        # for p in [os.path.join(log_dir, d) for d in os.listdir(log_dir)]:
        #     shutil.rmtree(p)
        return os.path.join(os.path.join(mdl_dir, mdl_name),
                            os.path.basename(self.basename))

    def save(self, directory=None):
        """
        Save configs and model

        Kwargs:
            directory <str> - directory where the files are saved
        """
        with open(self.basename+'_configs.json', 'w') as f:
            json.dump(self.configs, f)
        if not os.path.isfile(self.basename+'_model.h5'):
            self.model.save(self.basename+'_model.h5')

    def load_model(self, from_checkpoints=False, verbose=False):
        """
        Kwargs:
            from_checkpoints <bool> - load the checkpoint saves instead
        """
        model_filename = self.basename+'_model.h5'
        cp_filename = self.basename+'_model_cp.h5'
        if from_checkpoints and os.path.isfile(self.basename+'_model_cp.h5'):
            self.model = tf.keras.models.load_model(self.basename+'_model_cp.h5')
            if verbose:
                print("Loading", self.basename+'_model_cp.h5')
        elif os.path.isfile(self.basename+'_model.h5'):
            self.model = tf.keras.models.load_model(self.basename+'_model.h5')
            if verbose:
                print("Loading", self.basename+'_model.h5')

    def validate(self, verbose=False):
        """
        Kwargs:
            verbose <bool> -
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


def lstm3_conn(return_obj=True, **kwargs):
    """
    Kwargs:
        return_obj <bool> - return ModelConfigurator instance
    """
    # settigs and defaults
    categorical = kwargs.pop('categorical', True)
    test_size = kwargs.pop('test_size', 0.1)
    random_state = kwargs.pop('random_state', 8)
    verbose = kwargs.pop('verbose', True)

    kwargs.setdefault('optimizer', dg.optimizer)
    kwargs.setdefault('learning_rate', dg.learning_rate)
    kwargs.setdefault('epochs', dg.epochs)
    kwargs.setdefault('batch_size', dg.batch_size)

    # load training data
    features, labels = dg.utils.load_data(dg.DATA_DIR, extract_from_tar=True)

    configurator = ModelConfigurator(features, labels, categorical=categorical,
                                     random_state=random_state)
    configurator.test_train_split(test_size=test_size)
    configurator.configure(**kwargs).generate(verbose=verbose)
    configurator.train_model(checkpoint_callback=True,
                             tensor_board_callback=True,
                             save=True)
    if verbose:
        print(f"# Validation (Epoch {configurator.epochs})")
    configurator.validate(verbose=verbose)
    validation_end = configurator.validation.copy()
    if verbose:
        print("# Validation (Checkpoint)")
    configurator.load_model(from_checkpoints=True)
    configurator.validate(verbose=verbose)
    validation_cp = configurator.validation.copy()
    configurator.save()
    if return_obj:
        return configurator
    else:
        del configurator
        tf.keras.backend.clear_session()
        return validation_cp, validation_end


def hyperparameter_tuning():
    """
    Run a series of trainings to optimize the hyperparameters
    """
    import gc
    optimizers = ['Adam', 'SGD', 'RMSprop', 'Adadelta',
                  'Adagrad', 'Adamax', 'Nadam', 'Ftrl']
    learning_rates = [1e-4, 1e-3, 1e-2, 1e-1]
    epochs = [100, 500, 1000]
    batch_sizes = [8, 16, 32, 64]
    
    hyperpars = [
        {k: v for k, v in zip(
            ['optimizer', 'learning_rate', 'epochs', 'batch_size', 'validation_split'],
            pars)}
        for pars in iter_prod(optimizers, learning_rates, epochs, batch_sizes)]
    validation_metrics = []
    for pars in hyperpars:
        print(pars)
        res = lstm3_conn(return_obj=False, **pars)
        validation_metrics.append(res)
        gc.collect()
    val_accuracy_cp = np.array([vals[0]['accuracy'] for vals in val_metrics])
    best_acc_cp_idx = np.argmax(val_accuracy_cp)
    val_accuracy_end = np.array([vals[1]['accuracy'] for vals in val_metrics])
    best_acc_end_idx = np.argmax(val_accuracy_end)
    print(f"\nBest hyperpars from checkpoints: acc={val_accuracy_cp[best_acc_cp_idx]}")
    print(hyperpars[best_acc_cp_idx])
    print(f"Best hyperpars from model end: acc={val_accuracy_end[best_acc_end_idx]}")
    print(hyperpars[best_acc_end_idx])


def eval_model(mdl_dir_name, **kwargs):
    """
    Load model from model directory and validate
    """
    model_dir = os.path.join(dg.MDL_DIR, mdl_dir_name)
    features = kwargs.pop('features', None)
    labels = kwargs.pop('labels', None)
    categorical = kwargs.pop('categorical', True)
    test_size = kwargs.pop('test_size', 0.1)
    random_state = kwargs.pop('random_state', 8)
    verbose = kwargs.pop('verbose', True)

    if features is None or labels is None:
        features, labels = dg.utils.load_data(dg.DATA_DIR, extract_from_tar=True)
    jsnf = [os.path.join(model_dir, f) for f in os.listdir(model_dir)
            if f.endswith('.json')][0]
    configurator = ModelConfigurator.from_json(jsnf,
                                               features=features, labels=labels,
                                               categorical=categorical,
                                               random_state=random_state)
    configurator.test_train_split(test_size=test_size)
    if verbose:
        print(f"\n### {model_dir}")
        pprint.pprint(configurator.configs)
    configurator.load_model(from_checkpoints=False, verbose=verbose)
    configurator.validate(verbose=verbose)
    model_end = configurator.validation.copy()
    configurator.load_model(from_checkpoints=True, verbose=verbose)
    configurator.validate(verbose=verbose)
    model_cp = configurator.validation.copy()
    del configurator
    return model_cp, model_end


def eval_all_models(**kwargs):
    """
    Load all models and validate
    """
    mdl_dir = dg.MDL_DIR
    models = kwargs.pop('models', os.listdir(mdl_dir))
    categorical = kwargs.pop('categorical', True)
    test_size = kwargs.pop('test_size', 0.1)
    random_state = kwargs.pop('random_state', 8)
    verbose = kwargs.pop('verbose', False)

    features, labels = dg.utils.load_data(dg.DATA_DIR, extract_from_tar=True)

    accuracies_end = []
    accuracies_cp = []
    configs = []
    for model in models:
        # print(model)
        res = eval_model(model, features=features, labels=labels,
                         verbose=verbose)
        accuracies_cp.append(res[0])
        accuracies_end.append(res[1])
    best_idx = np.argmax([d['accuracy'] for d in accuracies_cp])
    print("Best model", models[best_idx])
    print("Checkpoint acc:", accuracies_cp[best_idx])
    print("Model acc:     ", accuracies_end[best_idx])
    return accuracies_cp, accuracies_end


if __name__ == "__main__":
    """
    Hyperparameter training results:
        1) (optimizer='Adam', learning_rate=1e-4, epochs=1000, batch_size=?)
        2) (optimizer='SGD', learning_rate=1e-3, batch_size=32)
    """
    # # Train a lstm3_conn model
    # lstm3_conn(optimizer='Adam', learning_rate=1e-4,
    #            epochs=1000, batch_size=8,
    #            validation_split=0.1)
    # hyperparameter_tuning()

    # # Model evaluations
    # eval_model('284SUBVCVVVRI_Adam')
    # eval_all_models()
    eval_all_models(
        models=['284SUBVCVVVRI_Adam', '4IV6655OA1DHC_Adam', '6BVEMP2H6BVOL_Adam',
                '45LSU9BHJ0CU8_Adam', '2QU9GPN8RFLGL_Adam', '4V5IKLFIE8PE8_SGD',
                '4O2D3C4KLE0HT_SGD'])
    
    

