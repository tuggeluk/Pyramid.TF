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
import cv2
from tensorflow.python import debug as tf_debug
from six.moves import xrange

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("batch_size", "2", "batch size for training")
tf.flags.DEFINE_string("logs_dir", "logs_pyramid/", "path to logs directory")
tf.flags.DEFINE_string("dataset", "MIT", "MIT or MNIST")
tf.flags.DEFINE_string("data_dir", "Data_zoo/MIT_SceneParsing/", "path to dataset")
tf.flags.DEFINE_float("learning_rate", "1e-3", "Learning rate for Adam Optimizer")
tf.flags.DEFINE_bool('debug', "False", "Debug mode: True/ False")
tf.flags.DEFINE_string('mode', "train", "Mode train/ test/ visualize")

# print(tf.flags.DEFINE_string)

MAX_ITERATION = int(1e5 + 1)
NUM_OF_CLASSES = 151
IMAGE_SIZE = 128


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
        perm = list(range(len(act_img.shape)))
        perm[1] = np.abs(dim)
        perm[np.abs(dim)] = 1
        act_img = tf.transpose(act_img, perm)

    # use tf.loop here
    # print(input.shape)
    # hidden = cell.zero_state(FLAGS.batch_size, tf.float32)
    # outputs = cell(act_img, hidden, dim, scope)
    '''  
    outputs = []
    hidden = cell.zero_state(FLAGS.batch_size, tf.float32)
    for i in range(input.shape.as_list()[1]):
        #print(outputs)
        out, hidden = cell(act_img[:, i],hidden , dim, scope)
        outputs.append(out)
    '''
    # outputs = []
    hidden = cell.zero_state(FLAGS.batch_size, tf.float32)
    column_image = tf.reshape(tf.transpose(act_img, perm=[0, 2, 1, 3]),
                              [FLAGS.batch_size, 1, -1, act_img.get_shape().as_list()[-1]])
    out, _ = cell(column_image[:, 0], hidden, dim, scope)
    h = act_img.get_shape().as_list()[0:3]
    h.append(out.get_shape().as_list()[-1])
    outputs = tf.reshape(tf.expand_dims(out, axis=1), h)

    # hidden = cell.zero_state(FLAGS.batch_size, tf.float32)
    # outputs, hidden = cell(act_img, hidden, dim, scope)

    # print(act_img[:, 1])
    # print(out.shape)
    # print(outputs)


    print("process dimension")
    # print(outputs)
    # outputs = tf.stack(outputs)
    # outputs = tf.transpose(outputs, [1, 0, 2, 3])
    return outputs


def allocate_pyramid_cell(dims, kernel_size, state_size, dense_hidden, input, scope_name):
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
    with tf.variable_scope(scope_name, initializer=tf.random_uniform_initializer(-.01, 1)) as scope:
        # cell = PyramidCell2D.BasicPyramidLSTMCell2D(input.get_shape().as_list()[1:3], kernel_size, state_size)
        # cell.init_variables(dims, input[:, 1].get_shape().as_list()[2], scope)
        cell = PyramidCell2D.BasicPyramidLSTMCell2D([1, np.prod(input.get_shape().as_list()[1:3])], kernel_size,
                                                    state_size)
        cell.init_variables(dims, input[:, 1].get_shape().as_list()[2], scope)

    # for all dimensions
    processed_dims = []
    # process dims
    for dim in dims:
        # print(input)
        output = process_dimension(input, cell, dim, scope)
        # print(output)
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
    # print(image.shape)
    # allocate_pyramid_cell(dims, kernel_size, state_size, dense_hidden, input, scope_name)
    out0 = allocate_pyramid_cell(dims, [3, 3], 10, 10, image, "pyramid_0")
    out0_r = tf.nn.relu(out0)
    out0_b = tf.contrib.layers.batch_norm(out0_r, center=True, scale=True, is_training=FLAGS.mode == 'train',
                                          scope='pyramid_0')

    # print(out0.shape)
    out1 = allocate_pyramid_cell(dims, [5, 5], 12, 12, out0_b, "pyramid_1")
    out1_r = tf.nn.relu(out1)
    out1_b = tf.contrib.layers.batch_norm(out1_r, center=True, scale=True, is_training=FLAGS.mode == 'train',
                                          scope='pyramid_1')

    # print(out1.shape)
    out2 = allocate_pyramid_cell(dims, [5, 5], 16, 16, out1_b, "pyramid_2")
    out2_r = tf.nn.relu(out2)
    out2_b = tf.contrib.layers.batch_norm(out2_r, center=True, scale=True, is_training=FLAGS.mode == 'train',
                                          scope='pyramid_2')

    # print(out2.shape)
    # dense2 = tf.layers.conv2d(inputs=out2_b, filters=NUM_OF_CLASSES, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)
    # dense2 = tf.contrib.layers.batch_norm(dense1, center=True, scale=True, is_training=FLAGS.mode == 'train')
    #dense1 = np.mean(out2_b)
    dense2 = tf.layers.dense(inputs=out2_b, units=NUM_OF_CLASSES)
    # print(dense2.shape)
    logits = tf.nn.softmax(dense2, -1)
    # print(logits.shape)
    # print(image.shape)
    classification = tf.argmax(logits, 3)
    # print(classification.shape)
    classification = tf.reshape(classification, [FLAGS.batch_size, IMAGE_SIZE, IMAGE_SIZE, 1])
    # print(classification.shape)

    return classification, logits


def train(loss_val, var_list):
    # optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
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
    if FLAGS.dataset == 'MIT':
        image_options = {'resize': True, 'resize_size': IMAGE_SIZE}
        train_records, valid_records = scene_parsing.read_dataset(FLAGS.data_dir)
        train_dataset_reader = dataset.BatchDatset(train_records, image_options)
        image = tf.placeholder(tf.float32, shape=[FLAGS.batch_size, IMAGE_SIZE, IMAGE_SIZE, 3], name="input_image")
        train_images, train_annotations = train_dataset_reader.next_batch(FLAGS.batch_size)
    else:
        image = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 1], name="input_image")
    annotation = tf.placeholder(tf.int32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 1], name="annotation")

    # Set up computation Graph
    # inputs2 = tf.reshape(inputs, [2, inputs.shape.as_list()[1] * inputs.shape.as_list()[1], 1, inputs.shape.as_list()[3]])
    # inputs2 = tf.squeeze(inputs2, axis=2)

    # image = tf.reshape(image, [FLAGS.batch_size, 1, -1, image.shape.as_list()[3]])

    pred_annotation, logits = inference(image, keep_probability)
    # logits = tf.placeholder(tf.int32, shape=[None, logits.shape[1], logits.shape[2], logits.shape[3]], name="logits")
    # print(pred_annotation)
    # print(logits.shape)
    # print(annotation.shape)

    # Summaries
    tf.summary.image("input_image", image, max_outputs=2)
    tf.summary.image("ground_truth", tf.cast(annotation, tf.uint8), max_outputs=2)
    tf.summary.image("pred_annotation", tf.cast(pred_annotation, tf.uint8), max_outputs=2)

    # Cross entropy loss
    # print(annotation.dtype)
    # print(logits.dtype)

    loss = tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(logits=tf.to_float(logits),
                                                                          labels=tf.squeeze(tf.to_int32(annotation),
                                                                                            squeeze_dims=[3]),
                                                                          name="entropy")))
    # loss = tf.reduce_mean((tf.losses.hinge_loss(logits=tf.to_float(logits), labels=tf.squeeze(tf.to_int32(annotation),
    # squeeze_dims=[3]))))
    # loss = tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(logits=tf.to_float(pred), labels=tf.squeeze(tf.to_int32(train_annotations), squeeze_dims=[3]), name="entropy")))

    # loss = tf.reduce_mean(tf.nn.sparse[tf.to_float(logits)-tf.to_float(train_annotations)])

    tf.summary.scalar("entropy", loss)
    # print(FLAGS.learning_rate)

    trainable_var = tf.trainable_variables()

    if FLAGS.debug:
        for var in trainable_var:
            utils.add_to_regularization_and_summary(var)
    train_op = train(loss, trainable_var)

    print("Setting up summary op...")
    summary_op = tf.summary.merge_all()

    if FLAGS.dataset == 'MIT':
        print("Setting up image reader MIT")
        # train_records, valid_records = scene_parsing.read_dataset(FLAGS.data_dir)
        # print(len(train_records))
        # print(len(valid_records))
        print("Setting up dataset reader")
        # image_options = {'resize': True, 'resize_size': IMAGE_SIZE}
        # if FLAGS.mode == 'train':
        # train_dataset_reader = dataset.BatchDatset(train_records, image_options)
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

    sess = tf_debug.LocalCLIDebugWrapperSession(sess, ui_type="curses")
    sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)

    if FLAGS.mode == "train":
        for itr in range(MAX_ITERATION):
            print(itr)
            train_images, train_annotations = train_dataset_reader.next_batch(FLAGS.batch_size)


            # Histogram Equalization:
            for i in range(FLAGS.batch_size):
                img_yuv = cv2.cvtColor(train_images[i, :], cv2.COLOR_BGR2YUV)
                img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
                train_images[i, :] = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

            # print(train_annotations.shape)
            # print(train_images.shape)
            # train_annotations2 = np.zeros(np.append(train_annotations.shape[0:3], NUM_OF_CLASSES))
            # for i0 in range(FLAGS.batch_size):
            #    for i1 in range(IMAGE_SIZE):
            #        for i2 in range(IMAGE_SIZE):
            #            train_annotations2[i0, i1, i2, train_annotations[i0, i1, i2, 0]] = 1

            feed_dict = {image: train_images, annotation: train_annotations,
                         keep_probability: 0.85}
            # feed_dict = {image: train_images, annotation: train_annotations2, keep_probability: 0.85}

            sess.run(train_op, feed_dict=feed_dict)

            if itr % 25 == 0:
                train_loss, summary_str = sess.run([loss, summary_op], feed_dict=feed_dict)
                print("Step: %d, Train_loss:%g" % (itr, train_loss))
                summary_writer.add_summary(summary_str, itr)

            if itr % 50 == 0:
                valid_images, valid_annotations = validation_dataset_reader.next_batch(FLAGS.batch_size)

                for i in range(FLAGS.batch_size):
                    img_yuv = cv2.cvtColor(valid_images[i, :], cv2.COLOR_BGR2YUV)
                    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
                    valid_images[i, :] = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

                valid_loss = sess.run(loss, feed_dict={image: valid_images, annotation: valid_annotations,
                                                       keep_probability: 1.0})
                print("%s ---> Validation_loss: %g" % (datetime.datetime.now(), valid_loss))
                saver.save(sess, FLAGS.logs_dir + "model.ckpt", itr)


    elif FLAGS.mode == "visualize":
        valid_images, valid_annotations = validation_dataset_reader.get_random_batch(FLAGS.batch_size)
        feed_dict = {image: valid_images, annotation: valid_annotations, keep_probability: 1}
        pred = sess.run(pred_annotation, feed_dict)
        # valid_annotations = np.squeeze(valid_annotations, axis=3)
        # pred = np.squeeze(pred, axis=3)


        for itr in range(FLAGS.batch_size):
            if valid_images[itr].astype(np.uint8).shape[2] == 1:
                print("bad shape")
                # print(valid_annotations[itr].astype(np.uint8).reshape(IMAGE_SIZE,IMAGE_SIZE).dtype)
                # print(pred[itr].astype(np.uint8).reshape(IMAGE_SIZE,IMAGE_SIZE).dtype)
                utils.save_image(valid_images[itr].astype(np.uint8).reshape(IMAGE_SIZE, IMAGE_SIZE), FLAGS.logs_dir,
                                 name="inp_" + str(5 + itr))
                utils.save_image(tf.Session().run(
                    tf.constant(valid_annotations[itr].astype(np.uint8).reshape(IMAGE_SIZE, IMAGE_SIZE))),
                    FLAGS.logs_dir, name="gt_" + str(5 + itr))
                utils.save_image(pred[itr].astype(np.uint8).reshape(IMAGE_SIZE, IMAGE_SIZE), FLAGS.logs_dir,
                                 name="pred_" + str(5 + itr))
            else:
                # print(valid_images)
                utils.save_image(valid_images[itr].astype(np.uint8), FLAGS.logs_dir, name="inp_" + str(5 + itr))
                # print(valid_annotations)
                print(pred)
                utils.save_image(valid_annotations[itr].astype(np.uint8).squeeze(), FLAGS.logs_dir,
                                 name="gt_" + str(5 + itr))
                utils.save_image(pred[itr].astype(np.uint8).squeeze(), FLAGS.logs_dir, name="pred_" + str(5 + itr))
                print("Saved image: %d" % itr)


if __name__ == "__main__":
    tf.app.run()
