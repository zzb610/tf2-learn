# %%
from tensorflow.keras import datasets, layers, Sequential, optimizers, losses
import tensorflow as tf
# %%


def preprocess(x, y):
    x = 2 * tf.cast(x, dtype=tf.float32) / 255. - 1
    y = tf.cast(y, dtype=tf.int32)
    y = tf.one_hot(y, depth=10)
    return x, y


def load_data():
    (x, y), (x_test, y_test) = datasets.cifar10.load_data()
    y = tf.squeeze(y, axis=1)
    y_test = tf.squeeze(y_test, axis=1)
    print(x.shape, y.shape, x_test.shape, y_test.shape)
    train_db = tf.data.Dataset.from_tensor_slices((x, y))
    train_db = train_db.shuffle(1000).map(preprocess).batch(128)
    test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_db = test_db.map(preprocess).batch(128)
    return train_db, test_db


train_db, test_db = load_data()
sample = next(iter(train_db))
sample[0].shape, sample[1].shape
# %%

model = Sequential([
    layers.Conv2D(64, kernel_size=[3, 3], padding='same'),
    layers.Conv2D(64, kernel_size=[3, 3], padding='same'),
    layers.MaxPool2D([2, 2], strides=2, padding='same'),

    layers.Conv2D(128, kernel_size=[3, 3],
                  padding='same', activation=tf.nn.relu),
    layers.Conv2D(128, kernel_size=[3, 3],
                  padding='same', activation=tf.nn.relu),
    layers.MaxPool2D([2, 2], strides=2, padding='same'),


    layers.Conv2D(256, kernel_size=[3, 3],
                  padding='same', activation=tf.nn.relu),
    layers.Conv2D(256, kernel_size=[3, 3],
                  padding='same', activation=tf.nn.relu),
    layers.MaxPool2D([2, 2], strides=2, padding='same'),

    layers.Conv2D(512, kernel_size=[3, 3],
                  padding='same', activation=tf.nn.relu),
    layers.Conv2D(512, kernel_size=[3, 3],
                  padding='same', activation=tf.nn.relu),
    layers.MaxPool2D([2, 2], strides=2, padding='same'),

    layers.Conv2D(512, kernel_size=[3, 3],
                  padding='same', activation=tf.nn.relu),
    layers.Conv2D(512, kernel_size=[3, 3],
                  padding='same', activation=tf.nn.relu),
    layers.MaxPool2D([2, 2], strides=2, padding='same'),

    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation=None)]
)


model.build(input_shape=[None, 32, 32, 3])
model.summary()
# %%
model.compile(optimizer=optimizers.Adam(learning_rate=0.01), loss=losses.CategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
model.fit(train_db, epochs=50)
# %%
 
# %%
 
# %%
