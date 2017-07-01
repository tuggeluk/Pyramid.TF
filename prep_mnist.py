import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data


dat_train = dat_train_gt = dat_test = dat_test_gt = None
batch_pointer = None


def get_plot_version(dat_train_gt_img):

    dat_train_plot = np.zeros(dat_train_gt_img.shape[0:2])

    if dat_train_gt_img.shape[2] == 1:
        for i in xrange(dat_train_gt_img.shape[0]):
            for ii in xrange(dat_train_gt_img.shape[1]):
                dat_train_plot[i, ii] = dat_train_gt_img[i, ii, 0]
    else:
        for i in xrange(dat_train_gt_img.shape[0]):
            for ii in xrange(dat_train_gt_img.shape[1]):
                if np.count_nonzero(dat_train_gt_img[i, ii, :]) == 1:
                    dat_train_plot[i, ii] = np.nonzero(dat_train_gt_img[i, ii, :])[0][0]
                elif np.count_nonzero(dat_train_gt_img[i, ii, :]) > 1:
                    # hack softmax classification
                    # print dat_train_gt_img[i, ii, :]
                    # print np.argmax(dat_train_gt_img[i, ii, :])
                    dat_train_plot[i, ii] = np.argmax(dat_train_gt_img[i, ii, :])
                else:
                    assert False

    dat_train_plot[0, 0:11] = range(11)
    return dat_train_plot


def plot_gt_img(dat_train_gt_img):
    dat_train_plot = get_plot_version(dat_train_gt_img)
    plt.interactive(False)
    plt.imshow(dat_train_plot, cmap="nipy_spectral")
    plt.show(block=True)

class DataSet(object):

  def __init__(self,
               data_folder,
               gridx,
               gridy,
               ezmode=False):
    self.dat_train, self.dat_train_gt, self.dat_test, self.dat_test_gt = init_data(data_folder, gridy, gridx, ezmode=False)
    self._index = 0


  def next_batch(self, batch_size, test):
    if test:
        batch_x = self.dat_test[0:batch_size]
        batch_y = self.dat_test_gt[0:batch_size]
    else:
        if self._index + batch_size > self.dat_train.shape[0]:
            self._index = 0

        batch_x = self.dat_train[self._index:(self._index+batch_size)]
        batch_y = self.dat_train_gt[self._index:(self._index+batch_size)]
    return [batch_x, batch_y]




def init_data(data_folder, gridy, gridx, ezmode = False):
    if ezmode:
        dat_train, dat_train_gt, dat_test, dat_test_gt = get_mnist_data_ez_mode(data_folder, gridy, gridx)
    else:
        dat_train, dat_train_gt, dat_test, dat_test_gt = get_mnist_data(data_folder,gridy, gridx)

    return dat_train, dat_train_gt, dat_test, dat_test_gt


def get_mnist_data(data_folder,gridy, gridx):
    zeros_array = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.])
    one_array = np.array([1.])

    mnist = input_data.read_data_sets(data_folder, one_hot=True)

    nrs_per_img = gridx * gridy
    nr_img_train = mnist.train.images.shape[0] / nrs_per_img

    dat_train = np.zeros((nr_img_train, gridy * 28, gridx * 28, 1))
    dat_train_gt = np.zeros((nr_img_train, gridy * 28, gridx * 28, 11))

    digit_count = 0
    for i in xrange(nr_img_train):
        for ii in xrange(gridy):
            for iii in xrange(gridx):
                dat_train[i, 0 + (28 * ii):28 + (28 * ii), (28 * iii):28 + (28 * iii)] = mnist.train.images[
                    digit_count].reshape(28, 28, 1)

                gt_patch = (np.outer(np.ceil(mnist.train.images[digit_count]),
                                     np.concatenate((mnist.train.labels[digit_count], np.zeros(1)))) +
                            np.outer((np.ceil(mnist.train.images[digit_count]) - 1) * -1, zeros_array))

                dat_train_gt[i, 0 + (28 * ii):28 + (28 * ii), (28 * iii):28 + (28 * iii), :] = gt_patch.reshape(28, 28,
                                                                                                                11)

                digit_count += 1

    nr_img_test = mnist.test.images.shape[0] / nrs_per_img
    dat_test = np.zeros((nr_img_test, gridy * 28, gridx * 28, 1))
    dat_test_gt = np.zeros((nr_img_test, gridy * 28, gridx * 28, 11))

    digit_count = 0
    for i in xrange(nr_img_test):
        for ii in xrange(gridy):
            for iii in xrange(gridx):
                dat_test[i, 0 + (28 * ii):28 + (28 * ii), (28 * iii):28 + (28 * iii)] = mnist.test.images[
                    digit_count].reshape(28, 28, 1)

                gt_patch = (np.outer(np.ceil(mnist.test.images[digit_count]),
                                     np.concatenate((mnist.test.labels[digit_count], np.zeros(1)))) +
                            np.outer((np.ceil(mnist.test.images[digit_count]) - 1) * -1, zeros_array))

                dat_test_gt[i, 0 + (28 * ii):28 + (28 * ii), (28 * iii):28 + (28 * iii), :] = gt_patch.reshape(28, 28,
                                                                                                               11)

                digit_count += 1


    return dat_train, dat_train_gt, dat_test, dat_test_gt


def get_mnist_data_ez_mode(data_folder,gridy, gridx):
    zeros_array = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.])
    one_array = np.array([1.])

    mnist = input_data.read_data_sets(data_folder, one_hot=True)

    nrs_per_img = gridx * gridy
    nr_img_train = mnist.train.images.shape[0] / nrs_per_img

    dat_train = np.zeros((nr_img_train, gridy * 28, gridx * 28, 1))
    dat_train_gt = np.zeros((nr_img_train, gridy * 28, gridx * 28, 10))

    digit_count = 0
    for i in xrange(nr_img_train):
        for ii in xrange(gridy):
            for iii in xrange(gridx):
                dat_train[i, 0 + (28 * ii):28 + (28 * ii), (28 * iii):28 + (28 * iii)] = mnist.train.images[
                    digit_count].reshape(28, 28, 1)

                gt_patch = (np.outer(np.ceil(mnist.train.images[digit_count]),
                                     (mnist.train.labels[digit_count])) +
                            np.outer((np.ceil(mnist.train.images[digit_count]) - 1) * -1, (mnist.train.labels[digit_count])))

                dat_train_gt[i, 0 + (28 * ii):28 + (28 * ii), (28 * iii):28 + (28 * iii), :] = gt_patch.reshape(28, 28,
                                                                                                                10)

                digit_count += 1

    nr_img_test = mnist.test.images.shape[0] / nrs_per_img
    dat_test = np.zeros((nr_img_test, gridy * 28, gridx * 28,1))
    dat_test_gt = np.zeros((nr_img_test, gridy * 28, gridx * 28, 10))

    digit_count = 0
    for i in xrange(nr_img_test):
        for ii in xrange(gridy):
            for iii in xrange(gridx):
                dat_test[i, 0 + (28 * ii):28 + (28 * ii), (28 * iii):28 + (28 * iii)] = mnist.test.images[
                    digit_count].reshape(28, 28,1)

                gt_patch = (np.outer(np.ceil(mnist.test.images[digit_count]),
                                     (mnist.test.labels[digit_count])) +
                            np.outer((np.ceil(mnist.test.images[digit_count]) - 1) * -1,(mnist.test.labels[digit_count])))

                dat_test_gt[i, 0 + (28 * ii):28 + (28 * ii), (28 * iii):28 + (28 * iii), :] = gt_patch.reshape(28, 28,
                                                                                                               10)

                digit_count += 1


    return dat_train, dat_train_gt, dat_test, dat_test_gt
