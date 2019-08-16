
import os, sys
import pickle
from glob import glob
from tensorflow.python.platform import gfile

import numpy as np
import imageio
import scipy.misc as misc
from skimage import color
# import matplotlib.pyplot as plt

from six.moves import urllib
import tarfile, zipfile


def create_image_lists(data_dir,
                       image_dir,
                       annotation_dir,
                       ftype):
    """
    Read 'data_dir/image_dir/*.ftype' and 'data_dir/annotation_dir/*.ftype'
    into list of dict with keys: 'image', 'annotation', 'filename'
    """
    if not gfile.Exists(data_dir):
        print("Image directory '" + data_dir + "' not found.")
        return None

    # Find image list if its common in annotation
    image_pattern = os.path.join(data_dir, image_dir, '*.' + ftype)
    image_lst = glob(image_pattern)
    image_lst = [item.replace('\\','/') for item in image_lst]
    data = []
    if not image_lst:
        print('No files found')
    else:
        for image_file in image_lst:
            filename = image_file.split("/")[-1].split('.')[0]
            annotation_file = os.path.join(data_dir, annotation_dir, filename + '.' + ftype)
            if os.path.exists(annotation_file):
                record = {'image': image_file, 'annotation': annotation_file, 'filename': filename}
                data.append(record)
            else:
                print('Annotation file not found for %s - Skipping' % filename)
                print('Pattern %s' % annotation_file)

    print ('Nunmber of files: %d' %len(data))
    return data

def read_data_record(data_dir,
                     image_dir,
                     annotation_dir,
                     ftype,
                     validation_len = 10,
                     test_len = 500):
    """
    Initialize list of datapath in data_dir if has not been initialized.
    """
    pickle_filename = 'VOC_datalist.pickle'
    pickle_filepath = os.path.join(data_dir, pickle_filename)
    if not os.path.exists(pickle_filepath):
        data = create_image_lists(data_dir, image_dir, annotation_dir, ftype)
        # Parse data into training and validation
        training_data = data[(validation_len + test_len):]
        validation_data = data[:validation_len]
        test_data = data[validation_len:(validation_len + test_len)]
        result = {'training':training_data,
                  'validation':validation_data,
                  'test':test_data}

        print ('Pickling ...')
        with open(pickle_filepath, 'wb') as f:
            pickle.dump(result, f, pickle.HIGHEST_PROTOCOL)
    else:
        print ('Found pickle file!')

    with open(pickle_filepath, 'rb') as f:
        data_records = pickle.load(f)
    return data_records

def create_BatchDatset(data_dir,
                       image_dir='JPEGImages',
                       annotation_dir='SegmentationClass',
                       ftype='jpg'):
    print(" create BatchDatset from " + data_dir)
    data_record = read_data_record(data_dir, image_dir, annotation_dir, ftype)
    train_dataset = BatchDatset(data_record['training'], True)
    valid_dataset = BatchDatset(data_record['validation'], False)
    test_dataset = BatchDatset(data_record['test'], False)
    return train_dataset, valid_dataset, test_dataset

class BatchDatset:

    images = []
    annotations = []
    batch_offset = 0
    epochs_completed = 0

    def __init__(self, data_records, is_shuffle=False):

        print("Initializing Batch Dataset Reader...")
        self.read_data_to_self(data_records)

        if is_shuffle:
            self.shuffle_data()
        return

    def read_data_to_self(self, data_records, resize_size = 96):
        self.images = np.stack([misc.imresize(imageio.imread(datum['image']),
                        [resize_size, resize_size], interp='bilinear') for datum in data_records])
        self.annotations = np.stack([misc.imresize(imageio.imread(datum['annotation']),
                        [resize_size, resize_size], interp='bilinear') for datum in data_records])
        return

    def shuffle_data(self):
        randperm = np.random.permutation(len(self.images))
        self.images = self.images[randperm]
        self.annotations = self.annotations[randperm]

    def get_records(self):
        return self.images, self.annotations

    def reset_batch_offset(self, offset=0):
        self.batch_offset = offset

    def next_batch(self, batch_size):
        start = self.batch_offset
        self.batch_offset += batch_size
        if self.batch_offset > len(self.images):
            # Finished epoch
            self.epochs_completed += 1
            print("****************** Epochs completed: " + str(self.epochs_completed) + "******************")
            # Shuffle the data
            self.shuffle_data()
            # Start next epoch
            start = 0
            self.batch_offset = batch_size
        end = self.batch_offset
        return self.images[start:end], self.annotations[start:end]

    def get_random_batch(self, batch_size):
        indexes = np.random.randint(0, len(self.images), size=[batch_size])
        return self.images[indexes], self.annotations[indexes]
