import glob
import numpy as np
from keras.models import load_model
import timeit
import util_functions as UF

validation_data_folder = "validation_data"
ragas = UF.list_ragas(validation_data_folder)
ragas = np.sort(np.array(ragas))
print(ragas)
X_val, y_val = UF.load_data(ragas, validation_data_folder)

model = load_model('best/best_model_7c.h5')


def time_this():
    model.predict(np.array(X_val), batch_size=32, verbose=0)


number = 100
print(timeit.timeit(time_this, number=100) * 1000 / (len(X_val) * number), "ms")  # avg 0.21351296351528154 ms
