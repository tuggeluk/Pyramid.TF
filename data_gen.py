import numpy as np
#import matplotlib.pyplot as plt
import os
import shutil
import scipy.misc
#import matplotlib.cm as cm
from PIL import Image

image_size = 128
num_train_samples = 100000
num_valid_samples = 200

#os.mkdir('toy_data')
#os.mkdir('toy_dat/','images')
#os.mkdir('toy_data/annotations')
#os.mkdir('toy_data/images/training')
#os.mkdir('toy_data/images/validation')
#os.mkdir('toy_data/annotations/training')
#os.mkdir('toy_data/annotations/validation')


for i in range(num_train_samples):
    pos = np.random.randint(0, image_size, 2)
    im = np.zeros([image_size, image_size, 3], dtype=np.uint8)
    label = np.random.randint(0, 2, 1)
    for j in range(image_size):
        for k in range(image_size):
            if j <= pos[0] and k <= pos[1]:
                im[j, k, 2*label] = 255
            elif j > pos[0] and k <= pos[1]:
                im[j, k, 2*(1-label)] = 255
            elif j <= pos[0] and k > pos[1]:
                im[j, k, 2*(1-label)] = 255
            elif j > pos[0] and k > pos[1]:
                im[j, k, 2*label] = 255
    #plt.imshow(im)
    im2 = Image.fromarray(im)
    im2.save('toy_data/images/training/image_train_%07d.png' % i)
    labels = im[:, :, 0]
    labels[labels==255] = 1
    im2 = Image.fromarray(labels)
    im2.save('toy_data/annotations/training/image_train_%07d.png' % i)
    #plt.imshow(labels)


for i in range(num_valid_samples):
    pos = np.random.randint(0, image_size, 2)
    im = np.zeros([image_size, image_size, 3], dtype=np.uint8)
    label = np.random.randint(0, 2, 1)
    for j in range(image_size):
        for k in range(image_size):
            if j <= pos[0] and k <= pos[1]:
                im[j, k, 2*label] = 255
            elif j > pos[0] and k <= pos[1]:
                im[j, k, 2*(1-label)] = 255
            elif j <= pos[0] and k > pos[1]:
                im[j, k, 2*(1-label)] = 255
            elif j > pos[0] and k > pos[1]:
                im[j, k, 2*label] = 255
    #plt.imshow(im)
    im2 = Image.fromarray(im)
    im2.save('toy_data/images/validation/image_val_%07d.png' % i)
    labels = im[:, :, 0]
    labels[labels==255] = 1
    im2 = Image.fromarray(labels)
    im2.save('toy_data/annotations/validation/image_val_%07d.png' % i)
    #plt.imshow(labels)