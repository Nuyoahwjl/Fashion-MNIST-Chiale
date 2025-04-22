import numpy as np
import struct
import gzip
import matplotlib.pyplot as plt
import os
from prettytable import PrettyTable
current_dir = os.path.dirname(os.path.abspath(__file__))



def load_mnist(path, kind='train'):
    """Load MNIST data from `path`"""
    labels_path = f'{path}/{kind}-labels-idx1-ubyte.gz'
    images_path = f'{path}/{kind}-images-idx3-ubyte.gz'

    with gzip.open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II', lbpath.read(8))
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8)

    with gzip.open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
        images = np.frombuffer(imgpath.read(), dtype=np.uint8).reshape(len(labels), 784)

    return images, labels


def show_image(train_images, train_labels, test_images, test_labels):

    table = PrettyTable()
    table.field_names = ["Dataset", "Images Shape", "Labels Shape"]
    table.add_row(["Train", train_images.shape, train_labels.shape])
    table.add_row(["Test ", test_images.shape, test_labels.shape])
    print(table)

    class_names = ['T_shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    # 创建一个 10x10 英寸的画布
    plt.figure(figsize = (10,10))  
    for i in range(0, 30):      
        # 创建子图（6行5列，索引从1开始）      
        plt.subplot(6, 5, i+1)  
        # 隐藏刻度
        plt.xticks([])
        plt.yticks([])
        # 关闭网格线
        plt.grid(False)
        # 显示图像
        # plt.imshow(x_train[i].reshape(28, 28), cmap='gray')
        plt.imshow(train_images[i].reshape(28, 28))
        # 设置标题为对应的类别名称
        plt.title(class_names[train_labels[i]])
    plt.suptitle('Train Images (First 30)', fontsize=16, y=0.95)
    plt.show() 

if __name__ == "__main__":
    data_dir = os.path.join(current_dir, 'data')
    train_images, train_labels = load_mnist(data_dir, kind='train')
    test_images, test_labels = load_mnist(data_dir, kind='t10k')
    show_image(train_images, train_labels, test_images, test_labels)
