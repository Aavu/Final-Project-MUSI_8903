import os
from os import symlink
import glob
import util_functions as UF

# Specify respective folders
data_folder = "phasespace_data"
train_data_folder = "train_data"
validation_data_folder = "validation_data"

UF.make_dir(validation_data_folder)
UF.make_dir(train_data_folder)

ragas = sorted(os.listdir(data_folder))

# Specify which piece(s) to include in validation set
validation_indices = {'Harikambhoji': [2, 5], 'Kalyani': [6], 'Karaharapriya': [9, 11], 'Mayamalavagowla': [11],
                      'Sankarabharanam': [11], 'Sanmukhapriya': [8], 'Todi': [8, 10]}

for raga in ragas:
    items = sorted(glob.glob(data_folder + '/' + raga + '/*.phasespace'))
    print("Getting train set for", raga)
    print(items[0])
    for item in items:
        song_id = int(item.split("/")[-1].split("_")[0])
        if song_id in validation_indices[raga]:
            UF.make_raga_dir(validation_data_folder, raga)
            dst = UF.get_path([validation_data_folder, raga, item.split("/")[-1]])
        else:
            UF.make_raga_dir(train_data_folder, raga)
            dst = UF.get_path([train_data_folder, raga, item.split("/")[-1]])
        try:
            symlink(item, dst)
        except FileExistsError:
            os.remove(dst)
            symlink(item, dst)
