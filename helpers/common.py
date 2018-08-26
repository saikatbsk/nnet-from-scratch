from __future__ import print_function
import os
import numpy as np
from six.moves.urllib.request import urlretrieve

def path_found(path):
    return os.path.exists(path)

def create_dir(path):
    if not path_found(path):
        os.makedirs(path)

def download(urls, download_path, expected_bytes, force=False):
    create_dir(download_path)

    downloaded_files = []
    for url in urls:
        filename = os.path.join(download_path, os.path.basename(url))
        if force or not path_found(filename):
            print('Attempting to download:', os.path.basename(url))
            filename, _ = urlretrieve(url, filename)
            print('Download Complete!')
        statinfo = os.stat(filename)
        if statinfo.st_size == expected_bytes[os.path.basename(url)]:
            print('Found and verified {}'.format(filename))
            downloaded_files.append(filename)
        else:
            raise Exception('Failed to verify ' + filename + '. Can you get to it with a browser?')

    return downloaded_files

def to_onehot(labels, num_classes):
    onehot_labels = np.zeros((labels.shape[0], num_classes))
    onehot_labels[np.arange(labels.shape[0]), labels] = 1
    return onehot_labels
