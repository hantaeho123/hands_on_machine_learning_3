import datetime
import numpy as np
import os
import tensorflow as tf

x_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
y_data = np.array([[0], [1], [1], [0]], dtype=np.float32)

tf.model = tf.keras.Sequential([
    tf.keras.Input(shape=(2,)),
    tf.keras.layers.Dense(units=2, activation='sigmoid'),
    tf.keras.layers.Dense(units=1, activation='sigmoid')
])

tf.model.compile(loss='binary_crossentropy', optimizer=tf.optimizers.SGD(learning_rate=0.1),  metrics=['accuracy'])

tf.model.summary()

# prepare callback
log_dir = os.path.join(".", "logs", "fit", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# add callback param to fit()
history = tf.model.fit(x_data, y_data, epochs=10000, callbacks=[tensorboard_callback])

predictions = tf.model.predict(x_data)
print('Prediction: \n', predictions)

score = tf.model.evaluate(x_data, y_data)
print('Accuracy: ', score[1])

'''
at the end of the run, open terminal / command window
cd to the source directory
tensorboard --logdir logs/fit

read more on tensorboard: https://www.tensorflow.org/tensorboard/get_started
'''