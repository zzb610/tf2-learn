# %%
from tensorflow.keras import Sequential, layers, losses, optimizers, datasets
import tensorflow as tf
# %%
network = Sequential([
    layers.Conv2D(6, kernel_size=3),
    layers.MaxPooling2D(2, strides=2),
    layers.ReLU(),
    layers.Conv2D(16, kernel_size=3),
    layers.MaxPooling2D(2, strides=2),
    layers.ReLU(),
    layers.Flatten(),
    layers.Dense(120, activation='relu'),
    layers.Dense(84, activation='relu'),
    layers.Dense(10)
])

network.build(input_shape=[None, 28, 28, 1])
network.summary()

# %%
network.compile(optimizer=optimizers.Adam(learning_rate=0.01), loss=losses.CategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
# %%

(x, y), (x_val, y_val) = datasets.mnist.load_data()
x = 2 * tf.convert_to_tensor(x, dtype=tf.float32) / 255.- 1
x = tf.expand_dims(x, axis=3)
y = tf.convert_to_tensor(y, dtype=tf.int32)
y = tf.one_hot(y, depth=10)

x_val = 2 * tf.convert_to_tensor(x_val, dtype=tf.float32) / 255.- 1
x_val = tf.expand_dims(x_val, axis=3)
y_val = tf.convert_to_tensor(y_val, dtype=tf.int32)
y_val = tf.one_hot(y_val, depth=10)

print(x.shape, y.shape)
print(x_val.shape, y_val.shape)
train_dataset = tf.data.Dataset.from_tensor_slices((x, y))
train_dataset = train_dataset.batch(512)

test_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
test_dataset = test_dataset.batch(512)
# %%
history = network.fit(train_dataset,epochs=30)
# %%
network.evaluate(test_dataset)
# %%
 
# %%
