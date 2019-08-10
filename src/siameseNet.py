import os, json
import tensorflow as tf
import matplotlib.pyplot as plt
from src.data_mnist import DataMNIST
from src.convNet import *
from src.utils import show_similar_image
from scipy.spatial.distance import cdist


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 


class SiameseNet(object):
    def __init__(self):
        self.flags = tf.app.flags
        self.FLAGS = self.flags.FLAGS
        self.config = json.load(open('config.json', 'r'))
        self.flags.DEFINE_integer('batch_size', self.config['network']['batch_size'], 'Batch Size')
        self.flags.DEFINE_integer('epoch', self.config['network']['epoch'], 'Epochs')
        self.flags.DEFINE_integer('step2save', self.config['network']['step2save'], 'Steps to save')
        self.dataset = DataMNIST()

    def train_model(self):
        # remove pre-trained log / images
        if os.path.isfile(os.path.join('images', 'train_%d.png' % self.FLAGS.step2save)):
            import shutil
            shutil.rmtree('train_log')

        placeholder_shape = [None] + list(self.dataset.image_train.shape[1:])
        print("placeholder shape", placeholder_shape)
        colours = ['#ff0000', '#ffff00', '#00ff00', '#00ffff', '#0000ff', '#ff00ff', '#990000', '#999900', '#009900', '#009999']

        next_batch = self.dataset.get_siamese_batch
        left = tf.placeholder(tf.float32, placeholder_shape, name='left')
        right = tf.placeholder(tf.float32, placeholder_shape, name='right')

        with tf.name_scope("similarity"):
            label = tf.placeholder(tf.int32, [None, 1], name='label')
            # 1 if same, 0 if different
            label_float = tf.to_float(label)
        margin = 0.5

        left_output = ConvNet(left, reuse=False)
        right_output = ConvNet(right, reuse=True)
        loss = contrastive_loss(left_output, right_output, label_float, margin)

        global_step = tf.Variable(0, trainable=False)
        train_step = tf.train.MomentumOptimizer(0.01, 0.99, use_nesterov=True).minimize(loss, global_step=global_step)

        saver = tf.train.Saver()
        with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
            sess.run(tf.global_variables_initializer())

            tf.summary.scalar('step', global_step)
            tf.summary.scalar('loss', loss)
            for var in tf.trainable_variables():
                tf.summary.histogram(var.op.name, var)
            merged = tf.summary.merge_all()
            writer = tf.summary.FileWriter('train_log', sess.graph)

            for i in range(self.FLAGS.epoch):
                batch_left, batch_right, batch_similarity = next_batch(self.FLAGS.batch_size)
                _, l, summary_str = sess.run([train_step, loss, merged],
                                            feed_dict={left:batch_left,
                                                        right:batch_right,
                                                        label: batch_similarity})
                writer.add_summary(summary_str, i)
                print("-"*5, "\repoch \t%d \t loss %.4f" % (i, l), "-"*5)

                if (i+1) % self.FLAGS.step2save == 0:
                    feat = sess.run(left_output,
                                    feed_dict={left:self.dataset.image_test})
                    labels = self.dataset.label_test

                    f = plt.figure(figsize=(16,9))
                    f.set_tight_layout(True)
                    for j in range(10):
                        plt.plot(feat[labels==j, 0].flatten(), feat[labels==j, 1].flatten(), '.', c=colours[j], alpha=0.8)
                    plt.legend(['0','1','2','3','4','5','6','7','8','9'])
                    plt.savefig('images/train_%d.png' % (i+1))
                    print("-"*5, "saving training images", "-"*5)

            saver.save(sess, "models/model.ckpt")

    def random_test(self):
        size = self.config['mnist']['size']
        img_plchd = tf.placeholder(tf.float32, [None, size, size, 1], name='img')
        net = ConvNet(img_plchd, reuse=False)
        
        idx = np.random.randint(0, len(self.dataset.label_test))
        im = self.dataset.image_test[idx]

        # load pre-trained model
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            ckpt = tf.train.get_checkpoint_state("models")
            saver.restore(sess, "models/model.ckpt")
            train_feat = sess.run(net, 
                            feed_dict={img_plchd:self.dataset.image_train[:2000]})
            search_feat = sess.run(net,
                            feed_dict={img_plchd:[im]})
        
        dist = cdist(train_feat, search_feat, 'cosine')
        rank = np.argsort(dist.ravel())

        n = 7
        show_similar_image([idx], self.dataset.image_test, rank[:n], self.dataset.image_train)
        print("ID of randomly tested image", idx)
        print("ID of retrieved images:", rank[:n])