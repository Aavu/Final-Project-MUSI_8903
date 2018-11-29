import os
import glob
from keras.utils import Sequence, to_categorical
import numpy as np
from keras.callbacks import Callback
from sklearn.metrics import f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import itertools

eps = np.finfo(np.float).resolution


def get_path(folders):
    """
    Converts lists of file/folders into URL format
    :param folders: List of files/folders in order
    :return: URL path
    """
    path = ""
    for f in folders:
        path = path + str(f) + "/"
    return path[:-1]


def get_data_path(data_folder, raga):
    """ Similar to get_path(). This function additionally takes raga as argument. """
    data_paths = []
    path = get_path([data_folder, raga])
    files = list_ragas(path)
    for file in files:
        data_paths.append(get_path([path, file]))
    return data_paths


def load_data(ragas, data_folder):
    """
    Load data from the data folder
    :param ragas: list of selected ragas
    :param data_folder: data folder from which data has to be loased
    :return: Tuple containing numpy arrays of X_data and y_data
    """
    X_data = []
    y_data = []
    for raga in ragas:
        print("Getting", raga, "data from", data_folder)
        items = glob.glob(get_path([data_folder, raga, '*.phasespace']))
        for item in items:
            item = os.path.relpath(os.path.realpath(item), get_path([data_folder, raga]))
            data = np.loadtxt(item)
            if not np.isnan(data).any():
                X_data.append(data)
                label = int(np.argwhere(ragas == raga).squeeze())
                y_data.append(label)
    return np.array(X_data).reshape(-1, 121, 121, 1), to_categorical(np.array(y_data).reshape(-1, 1), num_classes=len(ragas))


def getDelayCoordinates(inp, m, t):
    """
    Extracts phase space embedding.
    :param inp: input signal
    :param m: number of delay corrdinates
    :param t: delay per coordinate
    :return: numpy array of delay coordinates
    """
    tDelay = (m - 1) * t
    nSamples = len(inp)
    outArray = np.zeros((nSamples - tDelay, m))
    for ii in range(m):
        outArray[:, ii] = inp[tDelay - (ii * t):nSamples - (ii * t)]
    return outArray


def getCentBinMapp(range_cents=(-1200, 3600), resolution=10, oct_fold=0):
    mapp = {}
    bins = []
    if oct_fold == 1:
        range_cents = (0, 1200)
    for r in range(range_cents[0], range_cents[1]):
        bin = int(-1 * range_cents[0] / float(resolution)) + int(np.round(r / float(resolution)))
        mapp[r] = bin
        bins.append(bin)
    return mapp, np.unique(np.array(bins))


def get2DRagaRepresentation(delay_coord, tonic, resolution=10, range_cents=(-1200, 3600), oct_fold=0):
    """
    Extracts a 2 dimensional mapping.
    :param delay_coord: output of getDelayCoordinates function
    :param tonic: Tonic of the piece in Hz
    :param resolution: Cents compression amount
    :param range_cents: Range of cents
    :param oct_fold: 0 means no fold, 1 means fold octaves
    :return: Poincare map
    """
    mapp, bins = getCentBinMapp(range_cents, resolution, oct_fold)
    mtx = np.zeros((len(bins), len(bins)))
    if oct_fold == 1:
        for d in delay_coord:
            x_coord = int(1200 * np.log2(d[0] / tonic)) % 1200
            y_coord = int(1200 * np.log2(d[1] / tonic)) % 1200
            mtx[mapp[x_coord], mapp[y_coord]] += 1
    else:
        for d in delay_coord:
            x_coord = int(1200 * np.log2(d[0] / tonic))
            y_coord = int(1200 * np.log2(d[1] / tonic))
            mtx[mapp[x_coord], mapp[y_coord]] += 1
    return mtx


def getPhaseSpaceEmbPitchSignal(time_pitch, delay_tap, tonic, min_pitch=60, resolution=5, range_cents=(-1200, 3600), oct_fold=0):
    """ Computes phase space embedding (delay coordinates) for pitch pitch_file """
    hop_size = time_pitch[1, 0] - time_pitch[0, 0]
    delay_tap = int(np.round(delay_tap / hop_size))
    delay_coord = getDelayCoordinates(time_pitch[:, 1], 2, delay_tap)
    ind_sil1 = np.where(delay_coord[:, 0] < min_pitch)[0]
    ind_sil2 = np.where(delay_coord[:, 1] < min_pitch)[0]
    ind_sil = np.unique(np.append(ind_sil1, ind_sil2))
    delay_coord = np.delete(delay_coord, ind_sil, axis=0)
    mtx = get2DRagaRepresentation(delay_coord, tonic, resolution=resolution, range_cents=range_cents, oct_fold=oct_fold)
    return mtx


def raga_from_id(mapfile="ragaId_to_ragaName_mapping.txt"):
    """ Get the raga name from raga_id """
    lines = open(mapfile, 'r').readlines()
    id2raga = {}
    for line in lines:
        line = line.strip()
        line = line.split("\t")
        id2raga[line[0]] = line[1]
    return id2raga


def id_from_raga(mapfile="ragaId_to_ragaName_mapping.txt"):
    """ Get the raga_id from raga name """
    lines = open(mapfile, 'r').readlines()
    raga2id = {}
    for line in lines:
        line = line.strip()
        line = line.split("\t")
        raga2id[line[1]] = line[0]
    return raga2id


def make_dir(directory_name):
    if not os.path.isdir(directory_name):
        os.makedirs(directory_name)


def make_raga_dir(validation_data_folder, raga):
    if raga not in os.listdir(validation_data_folder):
        os.mkdir(validation_data_folder + "/" + raga)


def list_ragas(data_folder="phasespace"):
    """ List ragas that are inside the data_folder """
    ragas = os.listdir(data_folder)
    try:
        ragas.remove(".DS_Store")
    except ValueError:
        # print(".DS_Store not found")
        pass
    return ragas


def select_ragas(ragas_list=""):
    """ Select ragaes that are in the ragas_list file """
    raga_list = open(ragas_list).readlines()
    for i, raga in enumerate(raga_list):
        raga_list[i] = raga.strip()
    raga_dict = raga_from_id()
    selected = []
    for key, value in raga_dict.items():
        if key in raga_list:
            selected.append(value)
    return np.array(selected)


# Used only when the data is huge. This loads data into memory batch wise.
class DataGenerator(Sequence):
    def __init__(self, filenames, labels, batch_size=32, shuffle=True):
        self.filenames, self.labels = filenames, labels
        self.batch_size = batch_size
        self.indices = np.arange(len(self.filenames))
        self.shuffle = shuffle

    def __len__(self):
        return int(np.ceil(len(self.filenames) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.filenames[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]
        X = []
        for file_name in batch_x:
            try:
                data = np.loadtxt(file_name)
                X.append(data.reshape(121, 121, 1))
            except UnicodeDecodeError:
                print(file_name)
                break
        return np.array(X), np.array(batch_y)


# For computing f1, precision and recall scores
class MetricsPRF(Callback):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self._data = {}

    def on_train_begin(self, logs={}):
        self._data['val_recall'] = []
        self._data['val_precision'] = []
        self._data['val_f1'] = []

    def on_epoch_end(self, batch, logs={}):
        X_validation, y_validation = self.validation_data[0], self.validation_data[1]
        y_predict = np.asarray(self.model.predict(X_validation))

        y_validation = np.argmax(y_validation, axis=1)
        y_predict = np.argmax(y_predict, axis=1)

        val_recall = recall_score(y_validation, y_predict, average='weighted', labels=range(7))
        val_precision = precision_score(y_validation, y_predict, average='weighted', labels=range(7))
        val_f1 = f1_score(y_validation, y_predict, average='weighted', labels=range(7))

        self._data['val_recall'].append(val_recall)
        self._data['val_precision'].append(val_precision)
        self._data['val_f1'].append(val_f1)
        print(" — f1: % f — precision: % f — recall: % f" % (val_f1, val_precision, val_recall))
        return

    def get_data(self):
        return self._data


def evaluate_model(model, X_test, y_test, batch_size=32):
    return model.evaluate(X_test, y_test, batch_size=batch_size)


def plot(values, title, file_name, ylabel, xlabel='Epochs', legends=['Train', 'Val'], legend_loc=1, fig=1, figsize=(7, 5), show=False):
    plt.figure(fig, figsize=figsize)
    for v in values:
        plt.plot(v)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(legends, loc=legend_loc)
    if len(file_name) > 0:
        try:
            if len(file_name.split('/')) > 1:
                make_dir(file_name.split('/')[0])
            plt.savefig(file_name)
        except FileNotFoundError:
            print("Please provide a valid path. Ex: plots/myplot.png \nNot saving plot...")
    if show:
        plt.show()


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues, fig=1):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.figure(fig, figsize=(7, 5))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
