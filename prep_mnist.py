import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
import pdb


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
               test= False,
               emode=False):
    self.dat_train, self.dat_train_gt = get_mnist_data(data_folder, gridy, gridx,test, emode)
    self._index = 0


  def next_batch(self, batch_size):
    if self._index + batch_size > self.dat_train.shape[0]:
        self._index = 0

    batch_x = self.dat_train[self._index:(self._index+batch_size)]
    batch_y = self.dat_train_gt[self._index:(self._index+batch_size)]
    return [batch_x, batch_y]

  def get_random_batch(self, batch_size):
    rand_ind = np.random.randint(0,self.dat_train.shape[0]-batch_size)

    batch_x = self.dat_train[rand_ind:(rand_ind+batch_size)]
    batch_y = self.dat_train_gt[rand_ind:(rand_ind+batch_size)]
    return [batch_x, batch_y]



def get_mnist_data(data_folder,gridy, gridx, test, emode):

    mnist = input_data.read_data_sets(data_folder, one_hot=True)
    nrs_per_img = gridx * gridy

    if test:
        images = mnist.test.images
        labels = mnist.test.labels
    else:
        images = mnist.train.images
        labels = mnist.train.labels

    nr_img_train = (images.shape[0] / nrs_per_img)

    dat_train = np.zeros((nr_img_train, gridy * 28, gridx * 28, 1))
    dat_train_gt = np.zeros((nr_img_train, gridy * 28, gridx * 28, 1))

    digit_count = 0
    for i in xrange(nr_img_train):
        for ii in xrange(gridy):
            for iii in xrange(gridx):
                dat_train[i, 0 + (28 * ii):28 + (28 * ii), (28 * iii):28 + (28 * iii)] = images[digit_count].reshape(28, 28, 1)

                if emode:
                    dat_train_gt[i, 0 + (28 * ii):28 + (28 * ii), (28 * iii):28 + (28 * iii), :] = \
                        np.ones(28*28).reshape(28, 28, 1) * (np.argmax(labels[digit_count]) + 1)
                else:
                    dat_train_gt[i, 0 + (28 * ii):28 + (28 * ii), (28 * iii):28 + (28 * iii), :] = \
                        np.ceil(images[digit_count]).reshape(28,28,1)*(np.argmax(labels[digit_count])+1)

                digit_count += 1

    return dat_train, dat_train_gt



