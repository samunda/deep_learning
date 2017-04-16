from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# load data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# input x
x = tf.placeholder(tf.float32, [None, 784])

# model parameters - weights
W = tf.Variable(tf.zeros([784, 10]))

# model parameters - bias
b = tf.Variable(tf.zeros([10]))

# output y
y = tf.nn.softmax(tf.matmul(x, W) + b)

# known labels
y_ = tf.placeholder(tf.float32, [None, 10])

# cost function
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

# minimization
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# evaluation functions
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# validation set
validation_data = {x: mnist.validation.images, y_: mnist.validation.labels}

# test set
test_data = {x: mnist.test.images, y_: mnist.test.labels}

# setup training
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

N = 1000

costs = np.zeros((3, N), np.float32)
accuracies = np.zeros((3, N), np.float32)

for i in range(N):
    # training set
    batch_xs, batch_ys = mnist.train.next_batch(100)
    training_data = {x: batch_xs, y_: batch_ys}

    # train
    sess.run(train_step, feed_dict=training_data)

    # evaluations
    accuracies[0][i], costs[0][i] = sess.run([accuracy, cross_entropy], feed_dict=training_data)
    accuracies[1][i], costs[1][i] = sess.run([accuracy, cross_entropy], feed_dict=validation_data)
    accuracies[2][i], costs[2][i] = sess.run([accuracy, cross_entropy], feed_dict=test_data)

# results on test set
print(sess.run(accuracy, feed_dict=test_data))

# plot
plt.figure()

plt.subplot(1, 2, 1)
plt.plot(accuracies[0], label='training')
plt.hold(True)
plt.plot(accuracies[1], label='validation')
plt.plot(accuracies[2], label='test')
plt.hold(False)
plt.title('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(costs[0], label='training')
plt.hold(True)
plt.plot(costs[1], label='validation')
plt.plot(costs[2], label='test')
plt.hold(False)
plt.title('Cost')
plt.legend()
