# %%
import numpy as np
from numpy.lib.polynomial import polymul
data = []
num = 100
for i in range(num):
    x = np.random.uniform(-10, 10)
    eps = np.random.normal(0, 0.01)
    y = 1.477 * x + eps
    data.append([x, y])
data = np.array(data)
data
# %%
def mse(b, w, points):
    totalError = 0
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        totalError += (w*x + b - y)**2
    return totalError
# %%
def step_gradient(b_current, w_current, points, lr):
    b_gradient = 0
    w_gradient = 0
    M = float(len(points))
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]

        w_gradient += (w_current * x + b_current - y) * x
        b_gradient += (w_current * x + b_current - y)
    w_gradient *= 2 / M
    b_gradient *= 2 / M

    new_b = b_current - lr * b_gradient
    new_w = w_current - lr * w_gradient
    return [new_b, new_w]
# %%
def gradient_descent(points, starting_b, starting_w, lr, num_iteration):

    b = starting_b
    w = starting_w
    for step in range(num_iteration):
        b, w = step_gradient(b, w, points, lr)
        loss = mse(b, w, points)
        if step % 50 == 0:
            print(f"iteration: {step} loss: {loss}, w: {w}, b: {b}")
    return [b, w]
# %%
lr = 0.01
[b, w] = gradient_descent(data, 0, 0, lr, 1000)
loss = mse(b,w,data)
print(f'Final loss: {loss}, w: {w}, b: {b}')     
# %%
