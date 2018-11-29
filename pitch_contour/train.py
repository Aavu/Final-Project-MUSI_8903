import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.layers import MaxPooling1D, Dense, Flatten, Conv1D, Dropout
from keras.models import Sequential
from keras.optimizers import Adam
from keras.regularizers import l2
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import util_functions as UF

# Load train data
train = np.load("dataset_train.npz")
X_train = train['arr_0']
y_train = to_categorical(train['arr_1'])
print(X_train.shape, y_train.shape)

# Load validation data
val = np.load("dataset_validation.npz")
X_val = val['arr_0']
y_val = to_categorical(val['arr_1'])
print(X_val.shape, y_val.shape)

X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=0.5, shuffle=True)

# Defining the model
model = Sequential([Conv1D(8, 25, strides=2, input_shape=(X_train.shape[1], 1), activation='relu', kernel_regularizer=l2(0.0001)),
                    MaxPooling1D(),
                    Conv1D(32, 19, strides=2, activation='relu', kernel_regularizer=l2(0.0001)),
                    MaxPooling1D(),
                    Conv1D(128, 13, activation='relu', kernel_regularizer=l2(0.0001)),
                    MaxPooling1D(),
                    Conv1D(256, 7, activation='relu', kernel_regularizer=l2(0.0001)),
                    MaxPooling1D(),
                    Conv1D(512, 3, activation='relu', kernel_regularizer=l2(0.0001)),
                    MaxPooling1D(),
                    Dropout(0.25),
                    Flatten(),
                    Dense(512, activation='relu', kernel_regularizer=l2(0.0001)),
                    Dropout(0.6),
                    Dense(64, activation='relu', kernel_regularizer=l2(0.0001)),
                    Dropout(0.4),
                    Dense(7, activation='softmax')])

print(model.summary())
model.compile(Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

checkpointer = ModelCheckpoint(filepath='weights.h5', verbose=0, save_best_only=True)
history = model.fit(X_train, y_train, batch_size=64, epochs=200, verbose=2, validation_data=(X_val, y_val), callbacks=[checkpointer], shuffle=True)
train_loss = history.history['loss']
val_loss = history.history['val_loss']
train_acc = history.history['acc']
val_acc = history.history['val_acc']
model.save('best_model_v2.h5')

score = model.evaluate(X_test, y_test, batch_size=32)
print('test score:', score[0])
print('test accuracy:', score[1])

UF.plot([train_acc, val_acc], title='Train & Validation Accuracies', ylabel='Accuracy', legend_loc=4, fig=2, file_name="accuracy.png")
UF.plot([train_loss, val_loss], title='Train & Validation Losses', ylabel='Loss', legend_loc=1, fig=1, file_name="loss.png", show=True)
# plt.show()
