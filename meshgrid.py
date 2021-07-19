# %%
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import tensorflow as tf
# %%
x = tf.linspace(-8., 8, 100)
y = tf.linspace(-8., 8, 100)

x, y = tf.meshgrid(x, y)
x.shape, y.shape
# %%
z = tf.sqrt(x**2 + y**2)
z = tf.sin(z) / z

fig = plt.figure()
ax = Axes3D(fig)
ax.contour3D(x.numpy(), y.numpy(), z.numpy(), 50)
plt.show()
# %%
