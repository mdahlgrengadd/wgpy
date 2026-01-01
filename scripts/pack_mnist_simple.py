#!/usr/bin/env python3
"""
Simplified MNIST dataset downloader and packager.
Downloads MNIST data without requiring Chainer.
"""
import os
import shutil
import numpy as np
import gzip
from urllib.request import urlretrieve
import struct

def download_mnist():
    """Download MNIST dataset from PyTorch mirror."""
    # Use PyTorch's MNIST mirror which is more reliable
    base_url = 'https://ossci-datasets.s3.amazonaws.com/mnist/'
    files = [
        'train-images-idx3-ubyte.gz',
        'train-labels-idx1-ubyte.gz',
        't10k-images-idx3-ubyte.gz',
        't10k-labels-idx1-ubyte.gz'
    ]
    
    download_dir = os.path.expanduser('~/.chainer/dataset/pfnet/chainer/mnist')
    os.makedirs(download_dir, exist_ok=True)
    
    for filename in files:
        filepath = os.path.join(download_dir, filename)
        if not os.path.exists(filepath):
            print(f'Downloading {filename}...')
            urlretrieve(base_url + filename, filepath)
    
    return download_dir

def load_mnist_images(filename):
    """Load MNIST images from gz file."""
    with gzip.open(filename, 'rb') as f:
        magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows, cols)
    return images

def load_mnist_labels(filename):
    """Load MNIST labels from gz file."""
    with gzip.open(filename, 'rb') as f:
        magic, num = struct.unpack('>II', f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels

def create_npz_files(data_dir):
    """Create train.npz and test.npz files."""
    # Load training data
    print('Loading training data...')
    train_images = load_mnist_images(os.path.join(data_dir, 'train-images-idx3-ubyte.gz'))
    train_labels = load_mnist_labels(os.path.join(data_dir, 'train-labels-idx1-ubyte.gz'))
    
    # Load test data  
    print('Loading test data...')
    test_images = load_mnist_images(os.path.join(data_dir, 't10k-images-idx3-ubyte.gz'))
    test_labels = load_mnist_labels(os.path.join(data_dir, 't10k-labels-idx1-ubyte.gz'))
    
    # Save as NPZ files
    print('Saving train.npz...')
    np.savez(os.path.join(data_dir, 'train.npz'), x=train_images, y=train_labels)
    
    print('Saving test.npz...')
    np.savez(os.path.join(data_dir, 'test.npz'), x=test_images, y=test_labels)
    
    print(f'Created {len(train_images)} training samples and {len(test_images)} test samples')

def main():
    # Download MNIST data
    data_dir = download_mnist()
    
    # Create NPZ files
    create_npz_files(data_dir)
    
    # Create zip archive
    print('Creating lib/mnist.zip...')
    shutil.make_archive('lib/mnist', 'zip', data_dir)
    print('lib/mnist.zip created successfully!')
    
    # Create subset for demo (5000 samples)
    print('\nCreating subset for demo (mnist-subset-5k.zip)...')
    train_data = np.load(os.path.join(data_dir, 'train.npz'))
    test_data = np.load(os.path.join(data_dir, 'test.npz'))
    
    # Take first 4000 training samples and 1000 test samples
    train_subset = {'x': train_data['x'][:4000], 'y': train_data['y'][:4000]}
    test_subset = {'x': test_data['x'][:1000], 'y': test_data['y'][:1000]}
    
    # Save subset
    subset_dir = os.path.expanduser('~/.chainer/dataset/pfnet/chainer/mnist_subset')
    os.makedirs(subset_dir, exist_ok=True)
    np.savez(os.path.join(subset_dir, 'train.npz'), **train_subset)
    np.savez(os.path.join(subset_dir, 'test.npz'), **test_subset)
    
    shutil.make_archive('lib/mnist-subset-5k', 'zip', subset_dir)
    print('lib/mnist-subset-5k.zip created successfully!')

if __name__ == '__main__':
    main()
