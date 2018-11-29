import glob
import numpy as np
from keras.models import load_model
import timeit

val_set = []
item_names = []
items = glob.glob('validation_set_0/**/*.phasespace')
for item in items:
    item_names.append(items)
    data = np.loadtxt(item)
    if not np.isnan(data).any():
        val_set.append(data)
    else:
        print("found NaN in", item)

model = load_model('bestv2/best_model_40c.h5')


def time_this():
    model.predict(np.array(val_set).reshape(-1, 121, 121, 1), batch_size=32, verbose=0)


number = 100
print(timeit.timeit(time_this, number=100)*1000/(len(val_set)*number))      # avg 0.21351296351528154 ms
