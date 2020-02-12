# 基于去噪自动编码器的图像搜索
import numpy as np
from keras.models import Model
from keras.datasets import mnist
import cv2
from keras.models import load_model
from sklearn.metrics import label_ranking_average_precision_score
import time


print('loading mnist dataset')
t0 = time.time()
(x_train,y_train), (x_test,y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1)) # 把一维变成三维图像的形式，28x28x1
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))

noise_factor = 0.5
x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)

x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)
t1 = time.time()
print('mnist dataset loaded in :', t1-t0)

print('Loading model:')
t0 = time.time()
autoencoder = load_model('autoencoder.h5')
encoder = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer('encoder').output)
t1 = time.time()
print('Model loaded in :', t1-t0)

def plot_denoised_images():
    denoised_images = autoencoder.predict(
        x_test_noisy.reshape(x_test_noisy.shape[0], x_test_noisy.shape[1], x_test_noisy.shape[2], 1))
    test_img = x_test_noisy[0]
    resized_test_img = cv2.resize(test_img, (280, 280))
    cv2.imshow('input', resized_test_img)
    # cv2.waitKey(0)
    output = denoised_images[0]
    resized_output = cv2.resize(output, (280, 280))
    cv2.imshow('output', resized_output)
    cv2.waitKey(0)
    cv2.imwrite('test_results/noisy_image.jpg', 255 * resized_test_img)
    cv2.imwrite('test_results/denoised_image.jpg', 255 * resized_output)

def retrieve_closest_images(test_element, test_label, n_samples=10):
    learned_codes = encoder.predict(x_train)
    learned_codes = learned_codes.reshape(learned_codes.shape[0],
                                          learned_codes.shape[1] * learned_codes.shape[2] * learned_codes.shape[3])
    test_code = encoder.predict(np.array([test_element]))
    test_code = test_code.reshape(test_code.shape[1] * test_code.shape[2] * test_code.shape[3])

    distances = []

    for code in learned_codes:
        distance = np.linalg.norm(code - test_code)  #  求ln()
        distances.append(distance)
    nb_elements = learned_codes.shape[0]
    distances = np.array(distances)
    learned_codes_index = np.arange(nb_elements)
    labels = np.copy(y_train).astype('float32')
    labels[labels != test_label] = -1
    labels[labels == test_label] = 1
    labels[labels == 1] = 0
    distance_with_labels = np.stack((distances,labels,learned_codes_index))  ####
    sorted_distance_with_labels = distance_with_labels[distan] ####

    sorted_distances = 28 - sorted_distance_with_labels[:, 0]
    sorted_labels = sorted_distance_with_labels[:, 1]
    sorted_indexes = sorted_distance_with_labels[:, 2]
    kept_indexes = sorted_indexes[:n_samples]

    score = label_ranking_average_precision_score(np.array(sorted_labels), np.array(sorted_distances))  ####
    print('Average precision ranking score for tested element:', score)

    original_image = x_test[0]
    cv2.imshow('original_image',original_image)
    retrieced_images = x_train[int(kept_indexes[0]), :]
    for i in range(1,n_samples):
        retrieced_images = np.hstack(retrieced_images, x_train)
    cv2.imshow('Results',retrieced_images)
    cv2.waitKey(0)

    # cv2.imwrite('test_results/original_image.jpg',255 * cv2.)

retrieve_closest_images(x_test[0], y_test[0])

plot_denoised_images()

