import os
import numpy as np
from scipy.ndimage import gaussian_filter
from multiprocessing import Pool
import util_functions as UF

# delay_tap = 0.1
# alpha = 0.25
# sigma = 1.1

# This is the name of the train data folder
data_folder = "phasespace_data"
UF.make_dir(data_folder)

delay_tap = 0.3
alpha = 0.6
sigma = 1.1

# Map raga to their respective ragaID as provided by Dunya Dataset
mapfile = "ragaId_to_ragaName_mapping.txt"
id2raga = UF.raga_from_id(mapfile=mapfile)
raga2id = UF.id_from_raga(mapfile=mapfile)

data = {}
ragas = []

# This is useful if you have a large set and you want to choose selected ragas from it.
# Add the names of the ragas you want to work with in this file.
lines = open("selected_raga_list.txt").readlines()
selected_ragas = []
for line in lines:
    selected_ragas.append(line.strip())

print("selected ragas are", selected_ragas)

# walk through the dataset and get all the feature files for the selected ragas
for root, dirnames, filenames in os.walk('Dunya_Dataset/features/'):
    for filename in filenames:
        extn = filename.rpartition(".")[-1]
        if extn == "pitch" or extn == "tonic":
            raga_id = root.split("/")[2]
            if raga_id not in ragas:
                if id2raga[raga_id] in selected_ragas:
                    ragas.append(raga_id)
            try:
                data[raga_id].append(os.path.join(root, filename))
            except KeyError:
                data[raga_id] = [os.path.join(root, filename)]

print(len(ragas), "ragas")

# length of zero computes phase space for the entire recording
lengths = [0, 0.5, 1, 2, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]     # values are in "minutes"


# This function runs in parallel using multithreading for each length
def augment_data(l=0):
    print("length:", l)
    for raga in ragas:
        pitch_files = []
        tonic_files = []
        if id2raga[raga] not in os.listdir(data_folder):
            os.mkdir(UF.get_path([data_folder, id2raga[raga]]))
        files = data[raga]
        for file in files:
            if file.split(".")[-1] == "pitch":
                pitch_files.append(file)
            else:
                tonic_files.append(file)
        pitch_files.sort()
        tonic_files.sort()
        for i in range(len(pitch_files)):
            tonic = np.loadtxt(tonic_files[i])
            time_pitch = np.loadtxt(pitch_files[i])
            hop_size = time_pitch[1, 0] - time_pitch[0, 0]
            length = (1 / hop_size) * 60 * l
            if length != 0:
                split = int(len(time_pitch) / length)
                if split >= 1:
                    time_pitch_set = np.array_split(time_pitch, split)
                    for j in range(len(time_pitch_set)):
                        mat = UF.getPhaseSpaceEmbPitchSignal(time_pitch_set[j], delay_tap, tonic, resolution=10, oct_fold=1)
                        mat = np.power(mat, alpha)
                        mat = gaussian_filter(mat, sigma)
                        mat -= np.min(mat)
                        max_val = float(np.max(mat))
                        if max_val != 0:
                            mat /= max_val
                        fname, ext = pitch_files[i].split(".")
                        fname = fname.split("/")
                        fname = str(i) + "_" + str(int(l*60)) + '_sec-' + fname[-1] + "-" + fname[3] + "-" + fname[4]
                        fname = UF.get_path([data_folder, id2raga[raga], fname + '-' + str(j) + '.phasespace'])
                        np.savetxt(fname, mat)
            else:
                mat = UF.getPhaseSpaceEmbPitchSignal(time_pitch, delay_tap, tonic, resolution=10, oct_fold=1)
                mat = np.power(mat, alpha)
                mat = gaussian_filter(mat, sigma)
                mat -= np.min(mat)
                max_val = float(np.max(mat))
                if max_val != 0:
                    mat /= max_val
                fname, ext = os.path.splitext(pitch_files[i])
                fname = fname.split('/')[-1]
                fname = str(i) + "_" + str(fname)
                fname = UF.get_path([data_folder, id2raga[raga], fname + '.phasespace'])
                np.savetxt(fname, mat)


with Pool(8) as p:
    p.map(augment_data, lengths)
