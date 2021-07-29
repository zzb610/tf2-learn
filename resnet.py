# %%
from tensorflow.keras import layers, Sequential, optimizers, losses, datasets
from tensorflow import keras
import tensorflow as tf 

class BasicBlock(layers.Layer):

    def __init__(self, filter_num, strides=1):
        super(BasicBlock, self).__init__()

        self.conv1 = layers.Conv2D(
            filter_num, [3, 3], strides=strides, padding='same')
        self.bn1 = layers.BatchNormalization()
        self.relu = layers.Activation('relu')

        self.conv2 = layers.Conv2D(
            filter_num, [3, 3], strides=1, padding='same')
        self.bn2 = layers.BatchNormalization()

        if strides != 1:
            self.downsample = Sequential()
            self.downsample.add(layers.Conv2D(
                filter_num, [1, 1], strides=strides))
        else:
            self.downsample = lambda x: x

    def call(self, inputs, training=None):

        out = self.conv1(inputs)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        identity = self.downsample(inputs)
        ouputs = layers.add([out, identity])

        outputs = tf.nn.relu(ouputs)
        return outputs


class ResNet(keras.Model):
    def __init__(self, layer_dims, num_classes=10):
        super(ResNet, self).__init__()

        self.stem = Sequential([layers.Conv2D(64, [3, 3], strides=1), layers.BatchNormalization(
        ), layers.Activation('relu'), layers.MaxPool2D([2, 2], strides=(1, 1), padding='same')])

        self.layer1 = self.build_resblock(64, layer_dims[0])
        self.layer2 = self.build_resblock(128, layer_dims[1], stride=2)
        self.layer3 = self.build_resblock(256, layer_dims[2], stride=2)
        self.layer4 = self.build_resblock(512, layer_dims[3], stride=2)

        self.avgpool = layers.GlobalAvgPool2D()
        self.fc = layers.Dense(num_classes)

    def call(self, inputs, training=None):
        x = self.stem(inputs)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = self.fc(x)

        return x

    def build_resblock(self, filter_num, blocks, stride=1):
        res_blocks = Sequential()
        # only first BasicBlock May be not equal to 1
        res_blocks.add(BasicBlock(filter_num, stride))

        for _ in range(1, blocks):
            res_blocks.add(BasicBlock(filter_num, strides=1))

        return res_blocks


def resnet18():
    return ResNet([2, 2, 2, 2], num_classes=10)


model = resnet18()
model.build(input_shape=[None, 32, 32, 3])
model.compile(optimizer=optimizers.Adam(learning_rate=0.01),
              loss=losses.CategoricalCrossentropy(from_logits=True), metrics='accuracy')
model.summary()
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
model.fit(train_db, epochs=50)
# %%
