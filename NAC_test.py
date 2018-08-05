import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data", one_hot=False)

test_images = mnist.test.images
test_labels = mnist.test.labels

sequence_length = 1
batch_size = len(test_images)/sequence_length
labels = np.split(test_labels, batch_size)
labels = np.sum(labels, axis = -1)
hidden_units = 256

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
			net = tf.nn.dropout(net, 1.0)
			net = slim.fully_connected(net,10,activation_fn=None,scope='fc2')
			net = tf.nn.softmax(net)
			return net

def NACUnit(scope, inp, out, reuse) :
	with tf.variable_scope(scope, reuse = reuse) :
		w_tanh = tf.get_variable('w_tanh', shape = [inp, out], dtype = tf.float32)
		w_sig = tf.get_variable('w_sigmoid', shape = [inp, out], dtype = tf.float32)
		return tf.multiply(tf.nn.tanh(w_tanh), tf.sigmoid(w_sig))

def NAC_cell(inp, state, inp_units, hidden_units, reuse = False) :
	with tf.variable_scope('nac_cell', reuse = reuse) :
		w_input = NACUnit('w_input', inp_units, hidden_units, reuse)
		return tf.matmul(inp, w_input) + state

x = tf.placeholder(shape = [None, 784], dtype = tf.float32)
y_ = tf.placeholder(shape = [None], dtype = tf.float32)

cnn_out = CNNNetwork(x)


cnn_out = tf.reshape(cnn_out, [-1, sequence_length, 10])
cnn_out_unrolled = tf.unstack(cnn_out,cnn_out.get_shape()[1],1)
outputs = []

state = tf.zeros([batch_size, hidden_units], dtype = tf.float32)
for i in range(len(cnn_out_unrolled)) :
	if i == 0 :
		state = NAC_cell(cnn_out_unrolled[i], state, 10, hidden_units, reuse = False)
	else :
		state = NAC_cell(cnn_out_unrolled[i], state, 10, hidden_units, reuse = True)
	outputs.append(state)

outputs = outputs[-1]			

net = outputs
net = slim.fully_connected(net, 10, activation_fn = tf.nn.relu)
net = slim.fully_connected(net, 1, activation_fn = tf.nn.relu)
#net = tf.matmul(net, NACUnit('final', hidden_units, 1, reuse= False))
final_output = tf.squeeze(net)
print(net.get_shape().as_list())

loss = tf.reduce_mean(tf.abs(final_output - y_))
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.cast(tf.round(final_output), tf.float32),y_), tf.float32))

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()



saver.restore(sess, './models')


l = sess.run(loss, feed_dict = {x:test_images, y_ : labels})
print('the mean test loss is ', l)

if sequence_length == 1 :
	acc = sess.run(accuracy, feed_dict = {x:test_images, y_ : labels})
	print('the mean acc is ', acc)



