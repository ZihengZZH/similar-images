import tensorflow as tf
import matplotlib.pyplot as plt
from src.data_mnist import DataMNIST
from src.siamese_network import *


flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('batch_size', 512, 'Batch Size')
flags.DEFINE_integer('train_iter', 2000, 'Iteration')
flags.DEFINE_integer('step', 50, 'Save every iteration')


def train_model():
    dataset = DataMNIST()
    placeholder_shape = [None] + list(dataset.image_train.shape[1:])
    print("placeholder shape", placeholder_shape)
    colours = ['#ff0000', '#ffff00', '#00ff00', '#00ffff', '#0000ff', '#ff00ff', '#990000', '#999900', '#009900', '#009999']

    next_batch = dataset.get_siamese_batch
    left = tf.placeholder(tf.float32, placeholder_shape, name='left')
    right = tf.placeholder(tf.float32, placeholder_shape, name='right')

    with tf.name_scope("similarity"):
        label = tf.placeholder(tf.int32, [None, 1], name='label')
        # 1 if same, 0 if different
        label_float = tf.to_float(label)
    margin = 0.5

    left_output = mnist_model(left, reuse=False)
    right_output = mnist_model(right, reuse=True)
    loss = contrastive_loss(left_output, right_output, label_float, margin)

    global_step = tf.Variable(0, trainable=False)
    train_step = tf.train.MomentumOptimizer(0.01, 0.99, use_nesterov=True).minimize(loss, global_step=global_step)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        tf.summary.scalar('step', global_step)
        tf.summary.scalar('loss', loss)
        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var)
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter('train_log', sess.graph)

        for i in range(FLAGS.train_iter):
            batch_left, batch_right, batch_similarity = next_batch(FLAGS.batch_size)
            _, l, summary_str = sess.run([train_step, loss, merged],
                                        feed_dict={left:batch_left,
                                                    right:batch_right,
                                                    label: batch_similarity})
            writer.add_summary(summary_str, i)
            print("\r#%d - loss %.4f" % (i, l))
            
            if (i+1) % FLAGS.step == 0:
                feat = sess.run(left_output,
                                feed_dict={left:dataset.image_test})
                labels = dataset.label_test

                f = plt.figure(figsize=(16,9))
                f.set_tight_layout(True)
                for j in range(10):
                    plt.plot(feat[labels==j, 0].flatten(), feat[labels==j, 1].flatten(), '.', c=colours[j], alpha=0.8)
                plt.legend(['0','1','2','3','4','5','6','7','8','9'])
                plt.savefig('images/train_%d.png' % (i+1))

    saver.save(sess, "models/model.ckpt")
