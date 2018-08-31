from . import common
import gzip
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

url_train_images = 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz'
url_train_labels = 'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz'
url_test_images = 'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz'
url_test_labels = 'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz'

urls = [url_train_images, url_train_labels, url_test_images, url_test_labels]
download_path='datasets/mnist'
expected_bytes = {
    'train-images-idx3-ubyte.gz': 9912422,
    'train-labels-idx1-ubyte.gz': 28881,
    't10k-images-idx3-ubyte.gz': 1648877,
    't10k-labels-idx1-ubyte.gz': 4542
    }

IMAGE_SIZE = 28
num_training_samples = 60000
num_test_samples = 10000
num_classes = 10

def extract_data(files, force=False):
    train_images = []
    train_labels = []
    test_images = []
    test_labels = []

    for filename in files:
        print('Extracting %s.' % filename)
        if 'images' in filename:
            with gzip.open(filename) as bytestream:
                bytestream.read(16)
                num_images = num_training_samples if 'train' in filename else num_test_samples
                buffer = bytestream.read(IMAGE_SIZE * IMAGE_SIZE * num_images)
                data = np.frombuffer(buffer, dtype=np.uint8).astype(np.float32)
                data = data.reshape(num_images, IMAGE_SIZE*IMAGE_SIZE)
                if 'train' in filename:
                    train_images = data
                else:
                    test_images = data
        else:
            with gzip.open(filename) as bytestream:
                bytestream.read(8)
                num_images = num_training_samples if 'train' in filename else num_test_samples
                buffer = bytestream.read(1 * num_images)
                labels = np.frombuffer(buffer, dtype=np.uint8).astype(np.int64)
                if 'train' in filename:
                    train_labels = labels
                else:
                    test_labels = labels

    return train_images, train_labels, test_images, test_labels

def read_dataset(force_download=False):
    downloaded_files = common.download(urls, download_path, expected_bytes, force=force_download)
    train_images, train_labels, test_images, test_labels = extract_data(downloaded_files)

    return train_images, train_labels, test_images, test_labels

def plot_samples(images, labels, num_classes=num_classes):
    fig = plt.figure(figsize=(num_classes, 10))
    gs = gridspec.GridSpec(10, num_classes)

    for i in np.arange(num_classes):
        idxs = np.where(labels == i)[0]
        np.random.shuffle(idxs)
        idxs = idxs[:10]
        for count, idx in enumerate(idxs):
            sample_img = images[idx, :]
            sample_img = sample_img.reshape(IMAGE_SIZE, IMAGE_SIZE)
            ax = plt.subplot(gs[count, i])
            ax.set_xticks([])
            ax.set_yticks([])
            plt.imshow(sample_img, cmap='gray')
            if ax.is_last_row():
                ax.set_xlabel(i, fontsize=16)
