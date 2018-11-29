import glob
import os
import numpy as np
import util_functions as UF

dataset_folder = 'pc_dataset'
ragas = os.listdir(dataset_folder+'/'+'train')
try:
    ragas.remove('.DS_Store')
except ValueError:
    pass
ragas = np.array(ragas)

print(ragas)


def prepare_data(set_type='train', window=10):
    print("preparing data for:", set_type)
    hop_size = UF.get_hopsize(dataset_folder, ragas)
    pitch_files = {}
    tonic_files = {}
    tonic = {}
    pitch = {}
    normalized_pitch = {}
    pitch_blocks = np.array([])
    labels = []
    for raga in ragas:
        print("Loading raga", raga)
        pitch_files[raga] = np.sort(glob.glob(dataset_folder + '/' + set_type + '/' + raga + '/*.pitch'))
        tonic_files[raga] = np.sort(glob.glob(dataset_folder + '/' + set_type + '/' + raga + '/*.tonic'))
        tonic[raga] = UF.load_from_file(tonic_files[raga])
        pitch[raga] = UF.load_from_file(pitch_files[raga])
        normalized_pitch[raga] = []
        for i in range(len(pitch[raga])):
            normalized_pitch[raga].extend(UF.normalize(pitch[raga][i], tonic[raga][i]))
        normalized_pitch[raga] = np.array(normalized_pitch[raga])
        length = int(np.ceil(window * (1 / hop_size)))
        pitches = np.array(normalized_pitch[raga])
        hop_samples = int(length/2)     # 50% overlap
        for i in range(0, len(pitches) - length, hop_samples):
            value = np.array(pitches[i:(i+length)])
            pitch_blocks = np.vstack((pitch_blocks, value)) if len(pitch_blocks) else value
            label = int(np.argwhere(ragas == raga).squeeze())
            labels.append(label)
    labels = np.array(labels).reshape(-1, 1)
    pitch_blocks = np.array(pitch_blocks)
    pitch_blocks = pitch_blocks.reshape(-1, pitch_blocks.shape[1], 1)
    file_name = dataset_folder+"_"+set_type
    UF.save_dataset(file_name, pitch_blocks, labels)


prepare_data("train")
prepare_data("validation")
