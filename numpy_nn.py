# %%
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
# %%

def load_dataset():

    N_SAMPLES = 2000
    TEST_SIZE = 0.3

    X, y = make_moons(n_samples=N_SAMPLES, noise=0.2, random_state=100)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=42)
    return X, y, X_train, X_test, y_train, y_test
# %%
def make_plot(X, y, plot_name, file_name=None, XX=None, YY=None, preds=None, dark=False):
    if (dark):
        plt.style.use('dark_background')
    else:
        sns.set_style('whitegrid')
    plt.figure(figsize=(16, 12))
    axes = plt.gca()
    axes.set(xlabel="$x_1$", ylabel="$x_2$")

    plt.title(plot_name, fontsize=30)
    plt.subplots_adjust(left=0.20)
    plt.subplots_adjust(right=0.80)

    if(XX is not None and YY is not None and preds is not None):
        plt.contourf(XX, YY, preds.reshape(XX.shape), 25, alpha = 1, cmap=plt.cm.get_cmap("Spectral"))
        plt.contour(XX, YY, preds.reshape(XX.shape), levels=[.5], cmap="Greys", vmin=0,vmax=.6)
    plt.scatter(X[:, 0], X[:, 1], c=y.ravel(), s=40, cmap=plt.cm.get_cmap("Spectral"), edgecolors='none')
    plt.close
# %%
class Layer:
    def __init__(self, n_input, n_neurous, activation = None, weights = None, bias = None):
        """
        Args:
            n_input ([type]): [description]
            n_neurous ([type]): [description]
            activation ([type], optional): [description]. Defaults to None.
            weights ([type], optional): [description]. Defaults to None.
            bias ([type], optional): [description]. Defaults to None.
        """

        self.weights = weights if weights is not None else np.random.randn(n_input, n_neurous) * np.sqrt(1 / n_neurous)
        self.bias = bias if bias is not None else np.random.rand(n_neurous) * 0.1
        self.activation = activation
        self.last_activation = None
        self.error = None
        self.delta = None
    
    def activate(self, x):
        r = np.dot(x, self.weights) + self.bias
        self.last_activation = self._apply_activation(r)
        return self.last_activation

    def _apply_activation(self, r):
        
        if self.activation is None:
            return r
        
        elif self.activation == 'relu':
            return np.maximum(r, 0)
        elif self.activation == 'tanh':
            return np.tanh(r)
        elif self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-r))
        return r 
    
    def apply_activation_derivative(self, r):
        if self.activation is None:
            return np.ones_like(r)
        elif self.activation == 'relu':
            grad = np.array(r, copy=True)
            grad[r > 0] = 1
            grad[r <= 0] = 0
            return grad
        elif self.activation == 'tanh':
            return 1 - r**2
        elif self.activation == 'sigmoid':
            return r * (1 - r)
        return r


class NeuralNetwork:
    def __init__(self):
        self._layers = []
    def add_layer(self, layer):
        self._layers.append(layer)
    
    def feed_forward(self, X):
        for layer in self._layers:
            X = layer.activate(X)
        return X
    
    def backpropagation(self, X, y, learning_rate):
        output = self.feed_forward(X)
        for i in reversed(range(len(self._layers))):
            layer = self._layers[i]
            if layer == self._layers[-1]: # last layer
                layer.error = y - output
                layer.delta = layer.error * layer.apply_activation_derivative(output)
            else:
                next_layer = self._layers[i+1]
                layer.error = np.dot(next_layer.weights, next_layer.delta)
                layer.delta = layer.error * layer.apply_activation_derivative(layer.last_activation)
        # update
        for i in range(len(self._layers)):
            layer = self._layers[i]
            o_i = np.atleast_2d(X if i ==0 else self._layers[i-1].last_activation)
            layer.weights += layer.delta * o_i.T * learning_rate

    def train(self, X_train, X_test, y_train, y_test, learning_rate, max_epochs):

        y_onehot = np.zeros((y_train.shape[0], 2))
        y_onehot[np.arange(y_train.shape[0]), y_train] = 1

        mses = []
        accuracys = []
        for i in range(max_epochs + 1):
            for j in range(len(X_train)):
                self.backpropagation(X_train[j], y_onehot[j], learning_rate)
            if i % 10 == 0:
                mse = np.mean(np.square(y_onehot - self.feed_forward(X_train)))
                mses.append(mse)
                accuracy = self.accuracy(self.predict(X_test), y_test.flatten())
                accuracys.append(accuracy)
                print('Epoch: # %s, MSE: %f' % (i, float(mse)))
                print('Accuracy: {}% '.format(accuracy* 100))
        return mses, accuracys
    
    def predict(self, x):

        return self.feed_forward(x)
    
    def accuracy(self, X, y):
        return np.sum(np.equal(np.argmax(X, axis=1), y)) / y.shape[0]


nn = NeuralNetwork()
nn.add_layer(Layer(2, 25, 'sigmoid'))
nn.add_layer(Layer(25, 50, 'sigmoid'))
nn.add_layer(Layer(50, 25, 'sigmoid'))
nn.add_layer(Layer(25, 2, 'sigmoid'))

X, y, X_train, X_test, y_train, y_test = load_dataset()
# 调用 make_plot 函数绘制数据的分布，其中 X 为 2D 坐标， y 为标签
make_plot(X, y, "Classification Dataset Visualization ")
plt.show()
mses, accuracys = nn.train(X_train, X_test, y_train, y_test, 0.01, 200)

x = [i for i in range(0, 101, 10)]

# 绘制MES曲线
plt.title("MES Loss")
plt.plot(x, mses[:11], color='blue')
plt.xlabel('Epoch')
plt.ylabel('MSE')

 

# 绘制Accuracy曲线
plt.title("Accuracy")
plt.plot(x, accuracys[:11], color='blue')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')

plt.show()
plt.close()

# %%

 
