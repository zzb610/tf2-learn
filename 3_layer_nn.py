# %%
from tensorflow.keras import datasets
import tensorflow as tf
# %%
(x, y), (x_val, y_val) = datasets.mnist.load_data()
# %%
x = 2 * tf.convert_to_tensor(x, dtype=tf.float32) / 255 - 1
y = tf.convert_to_tensor(y)
train_dataset = tf.data.Dataset.from_tensor_slices((x, y))
train_dataset = train_dataset.batch(512)
# %%

# model
w1 = tf.Variable(tf.random.truncated_normal([28*28, 256], stddev=0.1))
b1 = tf.Variable(tf.zeros([256]))
w2 = tf.Variable(tf.random.truncated_normal([256, 128], stddev=0.1))
b2 = tf.Variable(tf.zeros([128]))
w3 = tf.Variable(tf.random.truncated_normal([128, 10], stddev=0.1))
b3 = tf.Variable(tf.zeros([10]))
# %%
lr = 0.1
def train_epoch(epoch):
    for step, (x, y) in enumerate(train_dataset):
        with tf.GradientTape() as tape:
            x = tf.reshape(x, [-1, 28*28])
            h1 = x @ w1 + b1
            h1 = tf.nn.relu(h1)
            h2 = h1 @ w2 + b2
            h2 = tf.nn.relu(h2)
            out = h2 @ w3 + b3

            y_onehot = tf.one_hot(y, depth=10)
            loss = tf.square(y_onehot - out)
            loss = tf.reduce_mean(loss)
        
        grads = tape.gradient(loss, [w1, b1, w2, b2, w3, b3])

        w1.assign_sub(lr * grads[0])
        b1.assign_sub(lr * grads[1])
        w2.assign_sub(lr * grads[2])
        b2.assign_sub(lr * grads[3])
        w3.assign_sub(lr * grads[4])
        b3.assign_sub(lr * grads[5])
    print(epoch, loss.numpy())

for epoch in range(30):
    train_epoch(epoch)
# %%
