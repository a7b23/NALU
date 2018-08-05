import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data", one_hot=False)


sequence_length = 1000
hidden_size = 256

def CNNNetwork(input_image,scope='Mnist',reuse = False):
	input_image = tf.reshape(input_image,[-1,28,28,1])
	with tf.variable_scope(scope,reuse = reuse) as sc :
		with slim.arg_scope([slim.conv2d,slim.fully_connected],
										  biases_initializer=tf.constant_initializer(0.0),
										  activation_fn=tf.nn.relu):
			net = slim.conv2d(input_image, 10, [5, 5], scope='conv1',padding = 'VALID')
			net = slim.max_pool2d(net,[2, 2], 2, scope='pool1')
			net = slim.conv2d(net, 20, [5, 5], scope='conv2', padding = 'VALID')
			net = slim.max_pool2d(net,[2, 2], 2, scope='pool2')
			net = slim.flatten(net)
			net = slim.fully_connected(net, 50, activation_fn = tf.nn.relu, scope = 'fc1')
			net = slim.fully_connected(net,10,activation_fn=None,scope='fc2')
			net = tf.nn.softmax(net)
			return net


x = tf.placeholder(shape = [None, 784], dtype = tf.float32)
y_ = tf.placeholder(shape = [None], dtype = tf.float32)

cnn_out = CNNNetwork(x)
gru_cell = tf.contrib.rnn.GRUCell(256)
cnn_out = tf.reshape(cnn_out, [-1, sequence_length, 10])
cnn_out_unrolled = tf.unstack(cnn_out,cnn_out.get_shape()[1],1)

outputs,_ = tf.contrib.rnn.static_rnn(gru_cell,cnn_out_unrolled,dtype=tf.float32)
outputs = outputs[-1]
net = outputs
net = slim.fully_connected(net, 10, activation_fn = tf.nn.relu)
net = slim.fully_connected(net, 1, activation_fn = tf.nn.relu)
final_output = tf.squeeze(net)
print(net.get_shape().as_list())

loss = tf.reduce_mean(tf.abs(final_output - y_))
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.cast(tf.round(final_output), tf.float32),y_), tf.float32))
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()



saver.restore(sess, './GRU_models')
test_images = mnist.test.images
test_labels= mnist.test.labels
test_batch_size = len(test_images)/sequence_length
labels = np.split(test_labels, test_batch_size)
labels = np.sum(labels, axis = -1)

l = sess.run(loss, feed_dict = {x:test_images, y_ : labels})
print('the mean test loss is ', l)
if sequence_length == 1 :
	acc = sess.run(accuracy, feed_dict = {x:test_images, y_ : labels})
	print('the mean acc is ', acc)





