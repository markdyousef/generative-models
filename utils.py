import matplotlib.pyplot as plt
import numpy as np
import keras.backend as K
from tqdm import tqdm

def plot_multi(images, dim=(4,4), figsize=(6,6), **kwargs):
    plt.figure(figsize=figsize)
    for i, img in enumerate(images):
        plt.subplot(*((dim) + (i+1,)))
        plt.imshow(img, **kwargs)
        plt.axis('off')
    plt.tight_layout()

def noise(batch_size, length=100):
    return np.random.rand(batch_size, length)

def data_D(size, G, x):
    n_train = x.shape[0]
    real_img = x[np.random.randint(0, n_train, size)]
    fake_img = G.predict(noise(size))

    X = np.concatenate((real_img, fake_img))
    labels = [0]*size + [1]*size
    return X, labels

def make_trainable(net, val):
    net.trainable = val
    for layer in net.layers:
        layer.trainable = val

def train(D, G, m, epochs=5000, bs=128, x=[]):
    dl = []
    gl = []
    for epoch in tqdm(range(epochs)):
        X, y = data_D(bs//2, G, x)
        dl.append(D.train_on_batch(X, y))
        make_trainable(D, False)
        
        preds = m.train_on_batch(noise(bs), np.zeros([bs]))
        gl.append(preds)
        make_trainable(D, True)
    return dl, gl


def limit_mem():
    K.get_session().close()
    cfg = K.tf.ConfigProto()
    cfg.gpu_options.allow_growth = True
    K.set_session(K.tf.Session(config=cfg))