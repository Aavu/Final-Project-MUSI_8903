import matplotlib.pyplot as plt
import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.layers import MaxPooling2D, Dense, Flatten, Conv2D, Dropout, BatchNormalization
from keras.models import Sequential
from keras.optimizers import Adam
from keras.regularizers import l2
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import util_functions as UF

# Choose data folders and specify file names
train_data_folder = "train_data"
validation_data_folder = "validation_data"
best_model_folder = 'best'
plot_folder = 'plots'
mapfile = "ragaId_to_ragaName_mapping.txt"
selected_ragas_list = "selected_raga_list.txt"

id2raga = UF.raga_from_id(mapfile=mapfile)
raga2id = UF.id_from_raga(mapfile=mapfile)

# Get list of selected ragas from selected_raga_file.txt
ragas = UF.list_ragas(train_data_folder)
lines = open(selected_ragas_list).readlines()
selected_ragas = []
for line in lines:
    selected_ragas.append(line.strip())
selected_ragas = np.sort(np.array(selected_ragas))
ragas = np.sort(np.array(ragas))

print("selected ragas: ", selected_ragas)

# Get paths of data for selected ragas
train_data_path = []
validation_data_path = []
for raga in selected_ragas:
    train_data_path.append(UF.get_data_path(train_data_folder, raga))
    validation_data_path.append(UF.get_data_path(train_data_folder, raga))

# Load data on to the memory
X_train, y_train = UF.load_data(ragas, train_data_folder)
print(X_train.shape, y_train.shape)
X_val, y_val = UF.load_data(ragas, validation_data_folder)
print(X_val.shape, y_val.shape)

# split test data from validation
X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=0.5)

# Define the model
model = Sequential([Conv2D(8, 3, strides=(2, 2), activation='relu', input_shape=(121, 121, 1), kernel_regularizer=l2(0.0001)), MaxPooling2D(),
                    Conv2D(64, 3, strides=(2, 2), activation='relu', kernel_regularizer=l2(0.0001)), MaxPooling2D(),
                    Conv2D(128, 2, activation='relu', kernel_regularizer=l2(0.0001)), MaxPooling2D(),
                    Conv2D(512, 2, activation='relu', kernel_regularizer=l2(0.0001)), MaxPooling2D(),
                    Dropout(0.25),
                    Flatten(),
                    Dense(256, activation='relu', kernel_regularizer=l2(0.0001)), Dropout(0.5),
                    Dense(64, activation='relu', kernel_regularizer=l2(0.0001)),
                    BatchNormalization(axis=1), Dropout(0.5),
                    Dense(7, activation='softmax')])

print(model.summary())


# Uncomment and change this part if you want to configure multiple GPUs
# config = tf.ConfigProto()
# # config.gpu_options.per_process_gpu_memory_fraction = 0.5
# config.gpu_options.visible_device_list = "1"
# set_session(tf.Session(config=config))

# Precision, recall and f1 score
metrics = UF.MetricsPRF(model)

# build the computational graph and setup optimizers, loss and metrics
model.compile(Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

UF.make_dir(best_model_folder)

# Checkpoint to save best weights
checkpointer = ModelCheckpoint(filepath=UF.get_path([best_model_folder, 'weights_7c.h5']), verbose=0, save_best_only=True)

# Train and validate the model
history = model.fit(X_train, y_train, batch_size=32, epochs=100, verbose=2, validation_data=(X_val, y_val),
                    callbacks=[checkpointer, metrics])

# plot parameters
prf_scores = metrics.get_data()
train_loss = history.history['loss']
val_loss = history.history['val_loss']
train_acc = history.history['acc']
val_acc = history.history['val_acc']

# save model with weights
model.save(UF.get_path([best_model_folder, 'best_model_7c.h5']))


# Evaluate model
score = UF.evaluate_model(model, X_test, y_test, batch_size=32)
print('test score:', score[0])
print('test accuracy:', score[1])

# plot confusion matrix for test set
rounded_predictions = model.predict_classes(X_test, batch_size=32, verbose=0)
cm = confusion_matrix(np.argmax(y_test, axis=1), rounded_predictions)
print(cm)


# plots
values = [prf_scores['val_recall'], prf_scores['val_precision'], prf_scores['val_f1']]
UF.plot(values, title='precision, recall and f1', file_name=plot_folder+'/prf.png', ylabel='Values', legends=['precision', 'recall', 'f1'], legend_loc=4, fig=3)
values = [train_acc, val_acc]
UF.plot(values, title='Train & Validation Accuracies', file_name=plot_folder+'/accuracy.png', ylabel='Accuracy', legend_loc=4, fig=2)
values = [train_loss, val_loss]
UF.plot(values, title='Train & Validation Losses', file_name=plot_folder+'/loss.png', ylabel='Loss', legend_loc=4, fig=1)

UF.plot_confusion_matrix(cm, ragas, title="Confusion Matrix", fig=4)
UF.plot_confusion_matrix(cm, ragas, title="Confusion Matrix Normalized", normalize=True, fig=5)
plt.show()
