from __future__ import print_function
import tensorflow as tf
import numpy as np

import pdb
import PyramidCell2D
import TensorflowUtils as utils
import read_MITSceneParsingData as scene_parsing
import datetime
import BatchDatsetReader as dataset
import prep_mnist as prp
from six.moves import xrange


FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("batch_size", "2", "batch size for training")
tf.flags.DEFINE_string("logs_dir", "logs_pyramid/", "path to logs directory")
tf.flags.DEFINE_string("dataset", "MNIST", "MIT or MNIST")
tf.flags.DEFINE_string("data_dir", "Data_zoo/MIT_SceneParsing/", "path to dataset")
tf.flags.DEFINE_float("learning_rate", "1e-4", "Learning rate for Adam Optimizer")
tf.flags.DEFINE_bool('debug', "False", "Debug mode: True/ False")
tf.flags.DEFINE_string('mode', "visualize", "Mode train/ test/ visualize")


MAX_ITERATION = int(1e5 + 1)
NUM_OF_CLASSES = 11
IMAGE_SIZE = 28


def process_dimension(input, cell, dim, scope):
    """
    Processes the image in a given dimension
    :param input: 
    :param cell: 
    :param dim: 
    :param scope: 
    :return: 
    """
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
    # use tf.loop here
    for i in range(input.shape.as_list()[1]):
        out, hidden = cell(act_img[:, i], hidden, dim, scope)
        outputs.append(out)


    print("process dimension")
    outputs = tf.stack(outputs)
    outputs = tf.transpose(outputs, [1, 0, 2, 3])
    return outputs


def allocate_pyramid_cell(dims, kernel_size, state_size,dense_hidden, input, scope_name):
    """
    Allocates on pyramid cell and processes the inputs in all dimensions according to dims
    :param dims: 
    :param kernel_size: 
    :param state_size: 
    :param dense_hidden: 
    :param input: 
    :param scope_name: 
    :return: 
    """

    # allocate pyramid cell
    with tf.variable_scope(scope_name, initializer=tf.random_uniform_initializer(-.01, 0.1)) as scope:
        cell = PyramidCell2D.BasicPyramidLSTMCell2D(input.get_shape().as_list()[1:3], kernel_size, state_size)
        cell.init_variables(dims, input[:, 1].get_shape().as_list()[2], scope)

    # for all dimensions
    processed_dims = []
    # process dims
    for dim in dims:
        output = process_dimension(input, cell, dim, scope)
        processed_dims.append(output)

    processed_dims = tf.add_n(processed_dims)

    # fully-connected
    out_dense = tf.layers.dense(inputs=processed_dims, units=dense_hidden, activation=tf.nn.tanh)

    return out_dense



def inference(image, keep_prob):
    """
    Allocates three pyramid layers, one dense layer and builds logits
    :param image: Tensor containing an Image
    :param keep_prob: 
    :return: 
    """
    print("do inference")
    # determine dimensions of image
    if len(image.get_shape().as_list()) != 4:
        print("dimension not supported yet")
        assert False

    dims = np.array(range(1, len(image.get_shape().as_list()) - 1))

    dims = np.concatenate((dims, dims * -1))

    print("Allocate cell 0")
    out0 = allocate_pyramid_cell(dims, [5], 4, 4, image, "pyramid_0")
    out1 = allocate_pyramid_cell(dims, [5], 8, 8, out0,  "pyramid_1")
    out2 = allocate_pyramid_cell(dims, [5], 16, 16, out1,  "pyramid_2")

    dense2 = tf.layers.dense(inputs=out2, units=NUM_OF_CLASSES)

    logits = tf.nn.softmax(dense2, -1)

    classification = tf.argmax(logits, 3)
    classification = tf.reshape(classification, [FLAGS.batch_size, IMAGE_SIZE, IMAGE_SIZE, 1])

    return classification, logits


def train(loss_val, var_list):
    optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
    grads = optimizer.compute_gradients(loss_val, var_list=var_list)
    if FLAGS.debug:
        # print(len(var_list))
        for grad, var in grads:
            utils.add_gradient_summary(grad, var)
    return optimizer.apply_gradients(grads)


def main(argv=None):

    # Placeholders
    keep_probability = tf.placeholder(tf.float32, name="keep_probabilty")
    image = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 1], name="input_image")
    annotation = tf.placeholder(tf.int32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 1], name="annotation")

    # Set up computation Graph
    pred_annotation, logits = inference(image, keep_probability)

    # Summaries
    tf.summary.image("input_image", image, max_outputs=2)
    tf.summary.image("ground_truth", tf.cast(annotation, tf.uint8), max_outputs=2)
    tf.summary.image("pred_annotation", tf.cast(pred_annotation, tf.uint8), max_outputs=2)

    # Cross entropy loss
    loss = tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                          labels=tf.squeeze(annotation, squeeze_dims=[3]),
                                                                          name="entropy")))
    tf.summary.scalar("entropy", loss)

    trainable_var = tf.trainable_variables()
    if FLAGS.debug:
        for var in trainable_var:
            utils.add_to_regularization_and_summary(var)
    train_op = train(loss, trainable_var)

    print("Setting up summary op...")
    summary_op = tf.summary.merge_all()

    if FLAGS.dataset == 'MIT':
        print("Setting up image reader MIT")
        train_records, valid_records = scene_parsing.read_dataset(FLAGS.data_dir)
        print(len(train_records))
        print(len(valid_records))
        print("Setting up dataset reader")
        image_options = {'resize': True, 'resize_size': IMAGE_SIZE}
        if FLAGS.mode == 'train':
            train_dataset_reader = dataset.BatchDatset(train_records, image_options)
        validation_dataset_reader = dataset.BatchDatset(valid_records, image_options)

    else:
        print("Setting up image reader MNIST")
        if FLAGS.mode == 'train':
            train_dataset_reader = prp.DataSet(FLAGS.data_dir, 1, 1, test=False, emode=False)
        validation_dataset_reader = prp.DataSet(FLAGS.data_dir, 1, 1, test=True, emode=False)


    sess = tf.Session()

    print("Setting up Saver...")
    saver = tf.train.Saver()
    summary_writer = tf.summary.FileWriter(FLAGS.logs_dir, sess.graph)

    sess.run(tf.global_variables_initializer())
    ckpt = tf.train.get_checkpoint_state(FLAGS.logs_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("Model restored...")

    if FLAGS.mode == "train":
        for itr in xrange(MAX_ITERATION):
            train_images, train_annotations = train_dataset_reader.next_batch(FLAGS.batch_size)
            feed_dict = {image: train_images, annotation: train_annotations, keep_probability: 0.85}

            sess.run(train_op, feed_dict=feed_dict)

            if itr % 10 == 0:
                train_loss, summary_str = sess.run([loss, summary_op], feed_dict=feed_dict)
                print("Step: %d, Train_loss:%g" % (itr, train_loss))
                summary_writer.add_summary(summary_str, itr)

            if itr % 500 == 0:
                valid_images, valid_annotations = validation_dataset_reader.next_batch(FLAGS.batch_size)
                valid_loss = sess.run(loss, feed_dict={image: valid_images, annotation: valid_annotations,
                                                       keep_probability: 1.0})
                print("%s ---> Validation_loss: %g" % (datetime.datetime.now(), valid_loss))
                saver.save(sess, FLAGS.logs_dir + "model.ckpt", itr)

    elif FLAGS.mode == "visualize":
        valid_images, valid_annotations = validation_dataset_reader.get_random_batch(FLAGS.batch_size)
        pred = sess.run(pred_annotation, feed_dict={image: valid_images, annotation: valid_annotations,
                                                     keep_probability: 1.0})
        valid_annotations = np.squeeze(valid_annotations, axis=3)
        pred = np.squeeze(pred, axis=3)


        for itr in range(FLAGS.batch_size):
            if valid_images[itr].astype(np.uint8).shape[2] == 1:
                print("bad shape")
                utils.save_image(valid_images[itr].astype(np.uint8).reshape(IMAGE_SIZE,IMAGE_SIZE), FLAGS.logs_dir, name="inp_" + str(5+itr))
                utils.save_image(valid_annotations[itr].astype(np.uint8).reshape(IMAGE_SIZE,IMAGE_SIZE), FLAGS.logs_dir, name="gt_" + str(5+itr))
                utils.save_image(pred[itr].astype(np.uint8).reshape(IMAGE_SIZE,IMAGE_SIZE), FLAGS.logs_dir, name="pred_" + str(5+itr))
            else:
                utils.save_image(valid_images[itr].astype(np.uint8), FLAGS.logs_dir, name="inp_" + str(5+itr))
                utils.save_image(valid_annotations[itr].astype(np.uint8), FLAGS.logs_dir, name="gt_" + str(5+itr))
                utils.save_image(pred[itr].astype(np.uint8), FLAGS.logs_dir, name="pred_" + str(5+itr))
                print("Saved image: %d" % itr)


if __name__ == "__main__":
    tf.app.run()
