from source.trainer import Trainer
from source.NN import Generator, Discriminator
from sklearn.datasets import fetch_mldata
import numpy as np
import pandas as pd
from chainer import cuda


import pickle

if __name__ == '__main__':
    gen = Generator(100)
    dis = Discriminator()

    data = fetch_mldata('MNIST original')
    X = data['data']
    n_train = X.shape[0]
    X = np.array(X, dtype=np.float32)
    X /= 255.
    X = X.reshape(n_train, 1, 28, 28)

    trainer = Trainer(gen, dis)

    trainer.fit(X, batch_size=1000, epochs=500)

    df_loss = pd.DataFrame(trainer.loss)
    df_loss.to_csv('loss.csv')

    cuda.get_device(0).use()

    gen.to_gpu()
    dis.to_gpu()

    with open('generator.model', 'wb') as w:
        pickle.dump(gen, w)

    with open('discriminator.model', 'wb') as w:
        pickle.dump(dis, w)
