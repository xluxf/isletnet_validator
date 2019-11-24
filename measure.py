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

        print(sample.shape)
        print(reference.shape)
        data = np.sum((reference - sample) ** 2, axis=1)
        score = min(data)
        result.append(score)
        print(sample)
        print(reference[0])
        print(reference[0] - sample)
        print((reference[0] - sample) ** 2)
        print(math.sqrt(sum((reference[0] - sample) ** 2)))

    return result



if __name__ == '__main__':


    # load PCA
    with open(PCA_PATH, 'rb') as file:
        pca = pickle.load(file)

    # load PCA data
    reference_data = np.load(PCA_DATA)

    # load encoder
    model = load_model(MODEL_PATH)

    # get image names
    names = os.listdir(PATH)

    result = pd.DataFrame()
    seznam = []

    for name in names:

        img = cv2.imread(os.path.join(PATH, name)) / 256 - .5
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

        # distances = np.log(distances)

        #result[name] = distances

        # distances. append(name)
        df1 = pd.DataFrame(distances)
        result = pd.concat([result,df1], ignore_index=True, axis=1)


        print('_______________')

        print(name)
        #print(distances)
        #print(np.mean(distances))
        #print(np.min(distances), np.max(distances), np.var(distances))

    result.columns = names

    result.T.to_csv('data_v2.csv')


