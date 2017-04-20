from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


def display_data(data):
    n, height, width = data.shape
    # size of display grid (g x g)
    g = int(np.sqrt(n))
    # data[0:g, :], data[g:2g, :], ... , data[g*(g-1):g*g, :]
    seq = [data[x * g: x * g + g, :].reshape(height * g, width) for x in range(g)]
    grid = np.column_stack(tuple(seq))
    plt.imshow(grid, 'Greys')
    plt.axis('off')
    plt.pause(0.01)


def weights_to_image(W):
    # scale weights to [0 255] and convert to uint8 (maybe change scaling?)
    x_min = tf.reduce_min(W)
    x_max = tf.reduce_max(W)
    weights_0_to_1 = (W - x_min) / (x_max - x_min)
    weights_0_to_255_uint8 = tf.image.convert_image_dtype(weights_0_to_1, dtype=tf.uint8)

    # convert to image summary format
    # [channels_out, height, width, channels_in]
    # number of images = channels_out
    return tf.transpose(weights_0_to_255_uint8, [3, 0, 1, 2])


def conv_layer(input, filter_size, channels_in, channels_out, stride, name='conv'):
    with tf.name_scope(name):
        W = tf.Variable(tf.truncated_normal([filter_size, filter_size, channels_in, channels_out], stddev=0.1),
                        name="W")
        b = tf.Variable(tf.ones([channels_out]) / 10.0, name="B")
        conv = tf.nn.conv2d(input, W, strides=[1, stride, stride, 1], padding='SAME') + b
        act = tf.nn.relu(conv)
        tf.summary.histogram("weights", W)
        tf.summary.histogram("biases", b)
        tf.summary.histogram("activations", act)

        if channels_in == 1:
            tf.summary.image("W", weights_to_image(W), channels_out)

        return act


def fc_layer(input, channels_in, channels_out, activation='relu', name='fc'):
    with tf.name_scope(name):
        W = tf.Variable(tf.truncated_normal([channels_in, channels_out], stddev=0.1))
        b = tf.Variable(tf.ones([channels_out]) / 10.0)
        a = tf.matmul(input, W) + b

        if activation == 'relu':
            act = tf.nn.relu(a)
        else:
            act = tf.nn.softmax(a)

        return act


# load data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# input X (batch_size x 28 x 28 x 1)
X = tf.placeholder(tf.float32, [None, 28, 28, 1])
tf.summary.image("X", X)

# middle layer parameters
L1 = 12
L2 = 20
L3 = 20
L4 = 100

# network
Y1 = conv_layer(X, 8, 1, L1, 1)  # stride 1 means output is 28 x 28 x L1
Y2 = conv_layer(Y1, 4, L1, L2, 2)  # stride 2 means output is 14 x 14 x L2
Y3 = conv_layer(Y2, 4, L2, L3, 2)  # stride 2 means output is 7 x 7 x L3
Y3_r = tf.reshape(Y3, shape=[-1, 7 * 7 * L3])
Y4 = fc_layer(Y3_r, 7 * 7 * L3, L4, 'relu')
Y = fc_layer(Y4, L4, 10, 'softmax')

# known labels
Y_ = tf.placeholder(tf.float32, [None, 10])

# cost function
with tf.name_scope('xent'):
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(Y_ * tf.log(Y), reduction_indices=[1]))
    tf.summary.scalar("cross_entropy", cross_entropy)

# minimization
with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.25).minimize(cross_entropy)

# evaluation functions
with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar("accuracy", accuracy)

# validation set
validation_data = {X: mnist.validation.images.reshape(-1, 28, 28, 1), Y_: mnist.validation.labels}

# test set
test_data = {X: mnist.test.images.reshape(-1, 28, 28, 1), Y_: mnist.test.labels}

# setup training
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

# TensorBoard
merged_summary = tf.summary.merge_all()
writer = tf.summary.FileWriter('tmp/demo15')
writer.add_graph(sess.graph)

N = 1000

factor = 10

costs = np.zeros((3, int(N / factor)), np.float32)
accuracies = np.zeros((3, int(N / factor)), np.float32)

for i in range(N):
    # training set
    batch_xs, batch_ys = mnist.train.next_batch(100)
    training_data = {X: batch_xs.reshape(-1, 28, 28, 1), Y_: batch_ys}

    # train
    sess.run(train_step, feed_dict=training_data)

    # evaluations
    if i % factor == 0:
        # display training data
        display_data(batch_xs.reshape(-1, 28, 28))

        # run and add TensorBoard summaries
        s = sess.run(merged_summary, feed_dict=training_data)
        writer.add_summary(s, i)

        j = int(i / factor)
        accuracies[0][j], costs[0][j] = sess.run([accuracy, cross_entropy], feed_dict=training_data)
        accuracies[1][j], costs[1][j] = sess.run([accuracy, cross_entropy], feed_dict=validation_data)
        accuracies[2][j], costs[2][j] = sess.run([accuracy, cross_entropy], feed_dict=test_data)

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
