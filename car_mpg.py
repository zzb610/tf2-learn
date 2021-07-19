# %%
from tensorflow import keras
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from matplotlib import pyplot as plt
# %%
dataset_path = keras.utils.get_file("auto - mpg.data", "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")
column_names = ['MPG', 'Cylinders','Displacement','Horsepower','Weight','Acceleration','Model Year','Origin']
raw_dataset = pd.read_csv(dataset_path, names=column_names, na_values='?', comment='\t', sep=' ', skipinitialspace=True)
dataset = raw_dataset.copy()
dataset.head()
# %%
print(dataset.isna().sum())
dataset = dataset.dropna()
print(dataset.isna().sum())
# %%
origin = dataset.pop('Origin')
dataset['USA'] = (origin == 1) * 1.0
dataset['Europe'] = (origin == 2) * 1.0
dataset['Japan'] = (origin == 3) * 1.0
dataset.tail()
# %%
train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

# %%
train_stats = train_dataset.describe()
train_stats.pop('MPG')
train_stats = train_stats.transpose()
# %%
train_labels = train_dataset.pop('MPG')
test_labels = test_dataset.pop('MPG')
# %%
def norm(x):
    return (x - train_stats['mean']) / train_stats['std']
normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)

print(normed_train_data.shape, train_labels.shape)
print(normed_test_data.shape, test_labels.shape)


# %%
train_db = tf.data.Dataset.from_tensor_slices((normed_train_data.values, train_labels.values))
train_db = train_db.shuffle(100).batch(32)
# %%

class NetWork(keras.Model):

    def __init__(self):
        super(NetWork, self).__init__()
        self.fc1 = layers.Dense(64, activation='relu')
        self.fc2 = layers.Dense(64, activation='relu')
        self.fc3 = layers.Dense(1)
    
    def call(self, inputs, training = None, mask = None):
        x = self.fc1(inputs)
        x = self.fc2(x)
        x = self.fc3(x)

        return x

model = NetWork()
model.build(input_shape=(4, 9))
print(model.summary())
optimizer = tf.keras.optimizers.RMSprop(0.001)
# %%
train_maes = []
test_maes = []
for epoch in range(200):

    for step, (x, y) in enumerate(train_db):
        with tf.GradientTape() as tape:
            out = model(x)
            loss = tf.reduce_mean(keras.losses.MSE(y, out))
            mae_loss = tf.reduce_mean(keras.losses.MAE(y, out))
        if step % 10 == 0:
            print(epoch, step, float(loss))

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

    train_maes.append(float(mae_loss))

    test_out = model(tf.constant(normed_test_data.values))
    test_mae = tf.reduce_mean(keras.losses.MAE(test_labels.values, test_out))
    test_maes.append(test_mae)

plt.figure()
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.plot(train_maes, label='Train')
plt.plot(test_maes, label='Test')
plt.legend()
plt.show()
    
 
 
# %%
