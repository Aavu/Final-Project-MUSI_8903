import glob
import os
import numpy as np
from keras.models import load_model
import timeit
import matplotlib.pyplot as plt
import util_functions as UF
from keras.utils import to_categorical
import collections

validation_data_folder = "validation_data"
ragas = UF.list_ragas(validation_data_folder)
ragas = np.sort(np.array(ragas))
print(ragas)
X_val, y_val = UF.load_data(ragas, validation_data_folder)

model = load_model('best/best_model_7c.h5')


def time_this():
    model.predict(np.array(X_val), batch_size=32, verbose=0)


# prediction time calculation
number = 100
print(timeit.timeit(time_this, number=100) * 1000 / (len(X_val) * number), "ms")  # avg 0.21351296351528154 ms

# Get data for each length
lengths = {}
for raga in ragas:
    print("Getting", raga, "data from", validation_data_folder)
    items = glob.glob(UF.get_path([validation_data_folder, raga, '*.phasespace']))
    for item in items:
        item = os.path.relpath(os.path.realpath(item), UF.get_path([validation_data_folder, raga]))
        try:
            length = int(item.split("/")[-1].split("_")[1])
        except ValueError:
            length = 0
        data = np.loadtxt(item).reshape(121, 121, 1)
        label = int(np.argwhere(ragas == raga).squeeze())
        if length in lengths:
            lengths[length].append((data, label))
        else:
            lengths[length] = [(data, label)]

print(lengths.keys())

# Evaluate the model for each length and get corresponding validation losses and accuracies
loss = {}
accuracy = {}
for length in lengths.keys():
    X_val = []
    y_val = []
    for i in range(len(lengths[length])):
        X_val.append(np.array(lengths[length][i][0]))
        y_val.append(to_categorical(np.array(lengths[length][i][1]), num_classes=len(ragas)))
    scores = model.evaluate(x=np.array(X_val), y=np.array(y_val), batch_size=32)
    loss[length] = scores[0]
    accuracy[length] = scores[1]

# Plot accuracy
loss = collections.OrderedDict(sorted(loss.items()))
accuracy = collections.OrderedDict(sorted(accuracy.items()))
names = list(accuracy.keys())
values = list(accuracy.values())
plt.bar(range(len(accuracy)), values, tick_label=names)
plt.ylim([0.5, 1.0])
print(loss, accuracy)
plt.show()
