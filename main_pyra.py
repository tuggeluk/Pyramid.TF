import os.path
import time

import numpy as np
import tensorflow as tf
import prep_mnist as prp
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import PyramidCell2D

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', './checkpoints/train_store_pyramid_lstm',
                           """dir to store trained net""")
tf.app.flags.DEFINE_integer('max_step', 200000,
                            """max num of steps""")
tf.app.flags.DEFINE_float('keep_prob', .8,
                          """for dropout""")
tf.app.flags.DEFINE_float('lr', .001,
                          """for dropout""")
tf.app.flags.DEFINE_integer('batch_size', 16,
                            """batch size for training""")
tf.app.flags.DEFINE_float('weight_init', .1,
                          """weight init for fully connected layers""")
tf.app.flags.DEFINE_bool('train', False,
                          """Set to train or test""")
tf.app.flags.DEFINE_string('data_folder', 'mnist_data',
                          """weight init for fully connected layers""")
tf.app.flags.DEFINE_float('learning_rate', .001,
                          """learning rate""")
tf.app.flags.DEFINE_bool('debug', False , """dir to store trained net""")
tf.app.flags.DEFINE_string('logs_dir', 'fully_conv_logs' , """dir to store trained net""")
tf.app.flags.DEFINE_integer('nr_batches', 20000, """dir to store trained net""")
tf.app.flags.DEFINE_string('plots_dir', 'plots',
                           """dir to store trained net""")


def process_dimension(input, cell, dim, scope):
    act_img = input
    # flip dimension
    if np.sign(dim) < 0:
        act_img = tf.reverse(act_img, [np.abs(dim)])

    # transpose to make the relevant dim, dim1
    if np.abs(dim) > 1:
        perm = range(len(act_img.shape))
        perm[1] = np.abs(dim)
        perm[np.abs(dim)] = 1
        act_img = tf.transpose(act_img, perm)


    hidden  = cell.zero_state(FLAGS.batch_size, tf.float32)
    outputs = []

    for i in range(input.shape.as_list()[1]):
        out, hidden = cell(act_img[:, i], hidden, dim, scope)
        outputs.append(out)


    print("process dimension")
    outputs = tf.stack(outputs)
    outputs = tf.transpose(outputs, [1, 0, 2, 3])
    return outputs


def inference(image, keep_prob):
    print("do inference")
    # determine dimensions of image
    if len(image.get_shape().as_list())!=4:
        print("dimension not supported yet")
        assert False

    dims = np.array(range(1, len(image.get_shape().as_list()) - 1))


    dims = np.concatenate((dims, dims*-1))
    print("todo")


    # allocate pyramid-cell
    hidden = None
    with tf.variable_scope('pyramid_lstm', initializer=tf.random_uniform_initializer(-.01, 0.1)) as scope:
        cell = PyramidCell2D.BasicPyramidLSTMCell2D([28], [3], 16)
        cell.init_variables(dims, image[:, 1].get_shape().as_list()[2],scope)
    # for all dimensions
    processed_dims = []

    for dim in dims:
        output = process_dimension(image, cell, dim, scope)
        processed_dims.append(output)

    processed_dims = tf.add_n(processed_dims)

    dense1 = tf.layers.dense(inputs=processed_dims, units=45, activation=tf.nn.tanh)

    dense2 = tf.layers.dense(inputs=dense1, units=11)

    logits = tf.nn.softmax(dense2, -1)

    classification = tf.argmax(logits, 3)
    classification = tf.reshape(classification, [FLAGS.batch_size, 28, 28, 1])

    # hidden = None
    # with tf.variable_scope('pyramid_lstm', initializer=tf.random_uniform_initializer(-.01, 0.1)):
    #     cell = PyramidCell2D.BasicPyramidLSTMCell2D([28], [3], 16)
    #     if hidden is None:
    #         hidden = cell.zero_state(FLAGS.batch_size, tf.float32)
    #     y_1, hidden = cell(image[:,1], hidden, dims[0])


    # return logits and predictions
    return classification, logits

def train(loss_val, var_list):
    optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
    grads = optimizer.compute_gradients(loss_val, var_list=var_list)
    return optimizer.apply_gradients(grads)


def run():
    # placeholders
    x = tf.placeholder(tf.float32, shape=[None, 28,28,1])
    y_ = tf.placeholder(tf.float32, shape=[None,28,28,11])
    keep_probability = tf.placeholder(tf.float32, name="keep_probabilty")

    # inference
    pred_annotation, logits = inference(x, keep_probability)

    # compute loss
    loss = tf.reduce_mean((tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_, name="entropy")))

    # make summary
    tf.summary.image("input_image", x, max_outputs=2)
    tf.summary.image("ground_truth", tf.cast(tf.expand_dims(tf.argmax(y_, dimension=3, name="gt"), dim=3), tf.uint8),
                     max_outputs=2)
    tf.summary.image("pred_annotation", tf.cast(pred_annotation, tf.uint8), max_outputs=2)
    tf.summary.scalar("entropy", loss)

    # set up training op
    trainable_var = tf.trainable_variables()
    train_op = train(loss, trainable_var)

    # start up tf
    sess = tf.Session()

    # set up saver
    saver = tf.train.Saver()
    summary_writer = tf.summary.FileWriter(FLAGS.logs_dir, sess.graph)

    # set up summary op
    summary_op = tf.summary.merge_all()

    # initialize
    sess.run(tf.global_variables_initializer())
    ckpt = tf.train.get_checkpoint_state(FLAGS.logs_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("Model restored...")


    # read data
    dataset = prp.DataSet(FLAGS.data_folder, 1, 1, False)

    if FLAGS.train:
        print("training net")
        for ba in xrange(FLAGS.nr_batches):
            batch = dataset.next_batch(FLAGS.batch_size, False)

            feed_dict = {x: batch[0], y_: batch[1], keep_probability: 0.85}
            sess.run(train_op, feed_dict=feed_dict)

            if ba % 100 == 0 and ba !=0:
                train_loss, summary_str = sess.run([loss, summary_op], feed_dict=feed_dict)
                print("Step: %d, Train_loss:%g" % (ba, train_loss))
                summary_writer.add_summary(summary_str, ba)

                saver.save(sess, FLAGS.logs_dir + "/model.ckpt", ba)

    else:
        # pick random image
        batch = dataset.next_batch(FLAGS.batch_size, True)

        feed_dict = {x: batch[0], y_: batch[1], keep_probability: 1}
        valid_loss, ims = sess.run([loss, pred_annotation], feed_dict=feed_dict)

        print("Validation Done")
        print("Validation Loss:%g" % (valid_loss))

        # print images from validation batch
        plt.ioff()

        for i in xrange(ims.shape[0]):
            single_img = ims[i].reshape(28, 28)
            single_img[0, 0:11] = range(11)
            mpimg.imsave(FLAGS.plots_dir + "/out"+str(i)+".png", single_img, cmap="nipy_spectral")

def main(argv=None):  # pylint: disable=unused-argument
    if tf.gfile.Exists(FLAGS.train_dir):
        tf.gfile.DeleteRecursively(FLAGS.train_dir)
    tf.gfile.MakeDirs(FLAGS.train_dir)
    run()


if __name__ == '__main__':
    tf.app.run()

