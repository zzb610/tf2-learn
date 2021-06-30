# %%
import tensorflow as tf

a = tf.constant(1.)
b = tf.constant(2.)
c = tf.constant(3.)
w = tf.constant(4.)

with tf.GradientTape() as tape: # 构建梯度环境
    tape.watch(w) # w加入梯度跟踪列表
    y = a * w**2 + b * w + c

[dw_dy] = tape.gradient(y, [w])
print(dw_dy)

# %%
