import numpy as np
from data_loader import load_mnist
from cnn_model import CNN
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


current_dir = os.path.dirname(os.path.abspath(__file__))


class_names = ['T_shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

def predict(test_images, test_labels):
    test_images = test_images / 255.0
    model = CNN()
    model_names = ['model_step1000.npz', 'model_step2000.npz', 'model_step3000.npz',
                   'model_step4000.npz', 'model_step5000.npz']
    batch_size = 1000
    total_samples = len(test_labels)

    print("model                   accuracy")
    print("================================")

    fig, axs = plt.subplots(1, 5, figsize=(30, 6))  # 一行五列图
    for idx, model_name in enumerate(model_names):
        model_path = os.path.join(current_dir, 'models', model_name)
        model.load_model(model_path)
        all_preds = []
        # 1000个样本一组进行预测
        for i in range(0, total_samples, batch_size):
            batch_images = test_images[i:i+batch_size]
            y_pred = model.forward(batch_images)
            preds = np.argmax(y_pred, axis=1)
            all_preds.extend(preds)
        all_preds = np.array(all_preds)
        accuracy = np.mean(all_preds == test_labels)
        print(f'{model_name.ljust(24)} {accuracy:.4f}')
        # 混淆矩阵
        cm = confusion_matrix(test_labels, all_preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(ax=axs[idx], xticks_rotation='horizontal', cmap='Reds', colorbar=False)
        axs[idx].set_title(model_name.replace('.npz', '')+"_confusion_matrices", fontsize=16)

    plt.tight_layout()  
    plt.show()


if __name__ == '__main__':
    data_dir = os.path.join(current_dir, 'data')
    test_images, test_labels = load_mnist(data_dir, kind='t10k')
    test_mask = np.random.choice(len(test_images), 100, replace=False)
    test_images = test_images[test_mask]
    test_labels = test_labels[test_mask]
    predict(test_images=test_images, test_labels=test_labels)
