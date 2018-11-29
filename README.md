# Final Project - MUSI_8903
## Carnatic Melakarta Raga Recognition using Phase Space Embedded features and Convolutional Neural Network

Please download the dataset from the following link
https://www.dropbox.com/sh/onow81ffnqm07l9/AADpQ6fS5zBC6UJD8WTJYWqia?dl=0

The folder contains 3 zip files. Un-zip all the zip files.
1) Place "Dunya_Dataset" and "phasespace_data" inside "phase_space_embedding" folder.
2) Place "pc_dataset", "pc_dataset_train.npz" and "pc_dataset_validation.npz" inside "pitch_contour" folder.

I have included screen-shots of folder structure for your reference.

Use python version 3.6
Please install the dependencies from "requirements.txt". 
Due to a bug with pycharm, it might sometimes fail to install scikit-learn if under conda environment.
In such case, please install using ```conda install scikit-learn==0.20.1``` command.

### Phase Space Embedding
1) createPhaseSpace.py - Creates the phase space data from the pitch contour features
2) split_validation.py - Splits the data into train and validation sets by creating a symlink for each file
3) train.py - Trains and validates the CNN model defined inside this file
4) evalModel.py - Evaluates prediction time
5) util_functions.py - contains helper functions

### Pitch Contour
1) prepare_data.py - Prepares the train and validation data and saves it as .npz files
2) train.py - Trains and validates the CNN model defined inside this file
3) util_functions.py - contains helper functions
