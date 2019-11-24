from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model, load_model
from keras.optimizers import Adam
import cv2
import math
import numpy as np
import os
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
import click

PATH = 'testing_data'
MODEL_PATH = 'models/encoder_cute_09.h5'
PCA_PATH = 'models/pca_model_cute.pkl'
PCA_DATA = 'models/pca_data_cute.npy'


def crop_batches(image, size):

    m, n, _ = image.shape
    m_size = m // size
    n_size = n // size
    
    batches = np.zeros((m_size * n_size, size, size, 3))

    index = 0

    for i in range(m_size):
        for j in range(n_size):
            
            batches[index] = image[i*size:(i+1)*size, j*size:(j+1)*size, :]
            index += 1

    assert index == m_size * n_size

    return batches


def compute_distances(reference, vector):

    result = []
    n_samples = vector.shape[0]

    for i in range(n_samples):

        sample = vector[i]

        data = np.sum((reference - sample) ** 2, axis=1)
        score = min(data)
        result.append(score)


    return result


@click.command()
@click.option('--img_path', prompt='path to images to score',
              help='Path to the original input images.')
@click.option('--store_csv/--not_store_csv', default=False,
              help='True if store scores in csv format')
def measure(img_path='', store_csv=False):
    # check, if the path is valid
    if not os.path.isdir(img_path):
        print(f'"{img_path}" is not a directory.')
        return

    # load PCA
    with open(PCA_PATH, 'rb') as file:
        pca = pickle.load(file)
    reference_data = np.load(PCA_DATA)

    # load encoder
    model = load_model(MODEL_PATH)

    # get image names
    names = os.listdir(img_path)

    result = pd.DataFrame()
    seznam = []

    result_folder = os.path.join('scored')
    if not os.path.isdir(result_folder):
        os.mkdir(result_folder)

    scores = {}

    for name in names:
        if not os.path.isfile(os.path.join(img_path, name)):
            continue

        img = cv2.imread(os.path.join(img_path, name)) / 256 - .5
        batches = crop_batches(img, 256)

        predictions = model.predict(batches, batch_size=8)

        #print(predictions.shape)

        samples, j, k, l = predictions.shape
        df_predictions = pd.DataFrame(predictions.reshape((samples, j * k * l)))

        #print(predictions.shape)

        # pca transformation
        features = df_predictions.columns
        x = df_predictions.loc[:, features].values
        # x = StandardScaler().fit_transform(x)
        pc_predictions = pca.transform(x)

        distances = compute_distances(reference_data, pc_predictions)

        distances = [np.log(d + 1) for d in distances]

        df1 = pd.DataFrame(distances)
        result = pd.concat([result,df1], ignore_index=True, axis=1)


        print('_____________')

        score = f'{np.median(distances):.03f}'
        print(f'SCORE: {score} \t{name}')

        scores[name] = float(score)

        cv2.imwrite(os.path.join(result_folder, f'{score}_{name}'), img)

        #print(distances)
        #print(np.mean(distances))
        #print(np.min(distances), np.max(distances), np.var(distances))

    # print sorted
    # for key in sorted(scores, key=scores.get, reverse=True):
        # print('_____________')
        # print(f'SCORE: {scores[key]} \t{key}')

    if store_csv:
        result.columns = names
        result.T.to_csv('data_v4.csv')



if __name__ == '__main__':
    measure()




