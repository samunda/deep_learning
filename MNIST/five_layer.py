from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# load data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# input X (batch_size x 28 x 28 x 1)
X = tf.placeholder(tf.float32, [None, 28, 28, 1])
X = tf.reshape(X, [-1, 28 * 28])

# middle layers
L1 = 200
L2 = 100
L3 = 60
L4 = 30

# model parameters - weights
# initialize to small random numbers
W1 = tf.Variable(tf.truncated_normal([28 * 28, L1], stddev=0.1))
W2 = tf.Variable(tf.truncated_normal([L1, L2], stddev=0.1))
W3 = tf.Variable(tf.truncated_normal([L2, L3], stddev=0.1))
W4 = tf.Variable(tf.truncated_normal([L3, L4], stddev=0.1))
W5 = tf.Variable(tf.truncated_normal([L4, 10], stddev=0.1))

# model parameters - bias
b1 = tf.Variable(tf.zeros([L1]))
b2 = tf.Variable(tf.zeros([L2]))
b3 = tf.Variable(tf.zeros([L3]))
b4 = tf.Variable(tf.zeros([L4]))
b5 = tf.Variable(tf.zeros([10]))

# output y
Y1 = tf.nn.relu(tf.matmul(X, W1) + b1)
Y2 = tf.nn.relu(tf.matmul(Y1, W2) + b2)
Y3 = tf.nn.relu(tf.matmul(Y2, W3) + b3)
Y4 = tf.nn.relu(tf.matmul(Y3, W4) + b4)
Y = tf.nn.softmax(tf.matmul(Y4, W5) + b5)

# known labels
Y_ = tf.placeholder(tf.float32, [None, 10])

# cost function
cross_entropy = tf.reduce_mean(-tf.reduce_sum(Y_ * tf.log(Y), reduction_indices=[1]))

# minimization
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# evaluation functions
correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# validation set
validation_data = {X: mnist.validation.images, Y_: mnist.validation.labels}

# test set
test_data = {X: mnist.test.images, Y_: mnist.test.labels}

# setup training
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

N = 1000

costs = np.zeros((3, N), np.float32)
accuracies = np.zeros((3, N), np.float32)

for i in range(N):
    # training set
    batch_xs, batch_ys = mnist.train.next_batch(100)
    training_data = {X: batch_xs, Y_: batch_ys}

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
