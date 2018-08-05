import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data", one_hot=False)


batch_size = 25
sequence_length = 10
lr = 1e-4
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
			net = tf.nn.dropout(net, 0.5)
			net = slim.fully_connected(net,10,activation_fn=None,scope='fc2')
			net = tf.nn.softmax(net)
			return net


x = tf.placeholder(shape = [batch_size*sequence_length, 784], dtype = tf.float32)
y_ = tf.placeholder(shape = [batch_size], dtype = tf.float32)

cnn_out = CNNNetwork(x)
gru_cell = tf.contrib.rnn.GRUCell(hidden_size)
cnn_out = tf.reshape(cnn_out, [batch_size, sequence_length, -1])
cnn_out_unrolled = tf.unstack(cnn_out,cnn_out.get_shape()[1],1)

outputs,_ = tf.contrib.rnn.static_rnn(gru_cell,cnn_out_unrolled,dtype=tf.float32)
outputs = outputs[-1]
net = outputs
net = slim.fully_connected(net, 10, activation_fn = tf.nn.relu)
net = slim.fully_connected(net, 1, activation_fn = tf.nn.relu)
final_output = tf.squeeze(net)
print(net.get_shape().as_list())

loss = tf.reduce_mean(tf.pow(final_output - y_, 2))
train_step = tf.train.AdamOptimizer(lr).minimize(loss)

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()


total_steps = 20000
display_steps = 100

for i in range(total_steps):
	batch = mnist.train.next_batch(batch_size *  sequence_length)
	labels = np.split(batch[1], batch_size)
	labels = np.sum(labels, axis = -1)
	
	_, l = sess.run([train_step, loss], feed_dict = {x : batch[0], y_ : labels})
	if i%display_steps == 0:
		print("step %d, training loss %g"%(i, l))

saver.save(sess, './GRU_models')







