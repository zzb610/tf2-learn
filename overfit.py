# %%
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from tensorflow.keras import Sequential
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense
import numpy as np
from tensorflow.python.keras import regularizers
# %%
N_SAMPLES = 1000
TEST_SIZE = 0.3
def load_data():
    X, y = make_moons(n_samples=N_SAMPLES, noise=0.25, random_state=100)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=42)
    return X_train, X_test, y_train, y_test
# %%

xmin, xmax = -2, 3
ymin, ymax = -1.5, 2

def mscatter(x, y, ax=None, m=None, **kw):
    import matplotlib.markers as mmarkers
    if not ax: ax = plt.gca()
    sc = ax.scatter(x, y, **kw)
    if (m is not None) and (len(m) == len(x)):
        paths = []
        for marker in m:
            if isinstance(marker, mmarkers.MarkerStyle):
                marker_obj = marker
            else:
                marker_obj = mmarkers.MarkerStyle(marker)
            path = marker_obj.get_path().transformed(
                marker_obj.get_transform())
            paths.append(path)
        sc.set_paths(paths)
    return sc

def make_plot(X, y, plot_name, file_name, XX = None, YY = None, preds = None):
    plt.figure()

    axes = plt.gca()
    axes.set_xlim([xmin, xmax])
    axes.set_ylim([ymin, ymax])
    axes.set(xlabel="$x_1$", ylabel="$x_2$")

    if(XX is not None and YY is not None and preds is not None):
            plt.contourf(XX, YY, preds.reshape(XX.shape), 25, alpha = 0.08, cmap=plt.cm.get_cmap("Spectral"))
            plt.contour(XX, YY, preds.reshape(XX.shape), levels=[.5], cmap="Greys", vmin=0,vmax=.6)
    markers = ['o' if i == 1 else 's' for i in y.ravel()]
    mscatter(X[:, 0], X[:,1], c = y.ravel(), s=20, cmap=plt.cm.get_cmap('Spectral'), edgecolors='none',m=markers)
    plt.show()
    plt.close()
X, y = make_moons(n_samples=N_SAMPLES, noise=0.25, random_state=100)
make_plot(X, y, None, '')

# %%
X_train, X_test, y_train, y_test = load_data()
for n in range(5):
    model = Sequential()
    model.add(Dense(8, input_dim=2, activation='relu'))

    for _ in range(n):
        model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=500, verbose=0)
    xx = np.arange(-2, 3, 0.01)
    # 可视化的 y 坐标范围为[-1.5, 2]
    yy = np.arange(-1.5, 2, 0.01)
    # 生成 x-y 平面采样网格点，方便可视化
    XX, YY = np.meshgrid(xx, yy)
    preds = model.predict_classes(np.c_[XX.ravel(), YY.ravel()])
    title = "网络层数：{0}".format(2 + n)
    file = "网络容量_%i.png" % (2 + n)
    make_plot(X_train, y_train, title, file, XX, YY, preds)
# %%
for n in range(5):
    model = Sequential()
    model.add(Dense(8, input_dim=2, activation='relu'))
    counter = 0
    for _ in range(5):
        model.add(Dense(64, activation='relu'))
        if counter < n:
            counter += 1
            model.add(layers.Dropout(rate=0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=500, verbose=0)

    xx = np.arange(-2, 3, 0.01)
    # 可视化的 y 坐标范围为[-1.5, 2]
    yy = np.arange(-1.5, 2, 0.01)
    # 生成 x-y 平面采样网格点，方便可视化
    XX, YY = np.meshgrid(xx, yy)
    preds = model.predict_classes(np.c_[XX.ravel(), YY.ravel()])
    title = "Dropout: {0}".format(n)
    file = "Dropout_%i.png" % (n)
    make_plot(X_train, y_train, title, file, XX, YY, preds)
# %%
def plot_weights_matrix(model, layer_index, plot_name, file_name):
    # 绘制权值范围函数
    # 提取指定层的权值矩阵
    weights = model.layers[layer_index].get_weights()[0]
    shape = weights.shape
    # 生成和权值矩阵等大小的网格坐标
    X = np.array(range(shape[1]))
    Y = np.array(range(shape[0]))
    X, Y = np.meshgrid(X, Y)
    # 绘制3D图
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    plt.title(plot_name, fontsize=20, fontproperties='SimHei')
    # 绘制权值矩阵范围
    ax.plot_surface(X, Y, weights, cmap=plt.get_cmap('rainbow'), linewidth=0)
    # 设置坐标轴名
    ax.set_xlabel('x', fontsize=16, rotation=0, fontproperties='SimHei')
    ax.set_ylabel('y', fontsize=16, rotation=0, fontproperties='SimHei')
    ax.set_zlabel('weight', fontsize=16, rotation=90, fontproperties='SimHei')
    # 保存矩阵范围图
    plt.show()
    plt.close(fig)

def build_model_with_regularization(_lamba):
    model = Sequential()
    model.add(Dense(8, input_dim=2, activation='relu'))
    model.add(Dense(256, activation='relu', kernel_regularizer=regularizers.l2(_lamba)))
    model.add(Dense(256, activation='relu', kernel_regularizer=regularizers.l2(_lamba)))
    model.add(Dense(256, activation='relu', kernel_regularizer=regularizers.l2(_lamba)))

    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

X_train, X_test, y_train, y_test = load_data()
for _lambda in [1e-5, 1e-3, 1e-1, 0.12, 0.13]:

    model = build_model_with_regularization(_lambda)

    history = model.fit(X_train, y_train, epochs=500, verbose=0)

    layer_index = 2
    
    plot_weights_matrix(model, layer_index,'', '')

    xx = np.arange(-2, 3, 0.01)
    # 可视化的 y 坐标范围为[-1.5, 2]
    yy = np.arange(-1.5, 2, 0.01)
    # 生成 x-y 平面采样网格点，方便可视化
    XX, YY = np.meshgrid(xx, yy)
    preds = model.predict_classes(np.c_[XX.ravel(), YY.ravel()])
    make_plot(X_train, y_train, title, file, XX, YY, preds)

# %%
