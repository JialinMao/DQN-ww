import tensorflow as tf

from dqn.ops import conv2d, linear

sess = tf.Session()

w = {}

x = tf.placeholder(tf.float32, [None, 8, 2, 1])
l1, w['l1_w'], w['l1_b'] = conv2d(x, 4, [2, 2], [2, 1], tf.constant_initializer(0.0))
l2, w['l2_w'], w['l2_b'] = conv2d(l1, 2, [1, 1], [1, 1], tf.constant_initializer(0.0))
shape = l2.get_shape().as_list()
l2_flat = tf.reshape(l2, [-1, reduce(lambda x, y: x * y, shape[1:])])
l3, w['l3_w'], w['l3_b'] = linear(l2_flat, 16, activation_fn=tf.nn.relu)
q, w['q_w'], w['q_b'] = linear(l3, 5, activation_fn=tf.nn.relu)
y, w['y_w'], w['y_w'] = tf.nn.softmax(q)

y_ = tf.placeholder(tf.float32, [None, 5])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)


tf.initialize_all_variables().run()
for i in range(1000):
    train_step.run({x: batch_xs, y_: batch_ys})

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print accuracy.eval({x: , y_: })
