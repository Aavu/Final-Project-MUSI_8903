import glob
import numpy as np
import matplotlib.pyplot as plt

eps = np.finfo(np.float).resolution


def trim_zeros(time_pitch):
    """
    Remove zeros from the given array
    :param time_pitch: data from .pitch file that consist of time values and its corrsponding pitch values
    :return: array of values without pitch zeros
    :rtype: numpy array that is same as the input dimension
    """
    return time_pitch[time_pitch[:, 1] > 0]


def normalize(pitches, tonic_freq, remove_zeros=True):
    """
    Normalize the frequency values with respect to it's tonic and calculate log2 of the normalized value.
    :param pitches: 1D array of pitch values in Hz
    :type pitches: An array of floats
    :param tonic_freq: the root frequency or the tonic frequency
    :type tonic_freq: float
    :param remove_zeros: Choose whether to truncate zeros in the pitches.
    :return: Normalized pitch values
    :rtype: numpy.array
    """
    trimmed_pitches = pitches
    if remove_zeros:
        trimmed_pitches = trim_zeros(pitches)
    c = np.log2(((trimmed_pitches[:, 1] + eps) / tonic_freq))
    return c


def load_from_file(array):
    """
    Read .pitch or .tonicFine or .tonic files and return all contents as a numpy array
    :param array: list of file paths
    :type array: List of String
    :return: Array of the read pitch or tonic files from the given list
    :rtype: numpy array
    """
    data_list = []
    for file_name in array:
        data = np.loadtxt(file_name)
        if data.shape == ():
            data = float(data)
        data_list.append(data)
    return np.array(data_list)


def get_hopsize(ds_folder, ragas, set_type='train'):
    """
    calculate the hop size by reading two consecutive samples from the .pitch file
    :param set_type: choose 'train' or 'validation' to get the pitch file from
    :param ds_folder: name of the dataset folder
    :type ds_folder: String
    :return: hopsize
    """
    file = glob.glob(ds_folder + '/' + set_type + "/" + ragas[0] + '/*.pitch')[0]
    time_pitch = np.loadtxt(file)
    return time_pitch[1, 0] - time_pitch[0, 0]


def save_dataset(file_name, data_x, data_y):
    print("saving in", file_name)
    np.savez(file_name, data_x, data_y)
    print("successfully saved!!!")


def plot(values, title, file_name, ylabel, xlabel='Epochs', legends=['Train', 'Val'], legend_loc=1, fig=1, figsize=(7, 5), show=False):
    plt.figure(fig, figsize=figsize)
    for v in values:
        plt.plot(v)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(legends, loc=legend_loc)
    plt.savefig(file_name)
    if show:
        plt.show()
