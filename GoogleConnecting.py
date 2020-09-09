import os, sys
from google.colab import drive
drive.mount('/content/mnt')
nb_path = '/content/notebooks'
os.symlink('/content/mnt/My Drive/Colab Notebooks', nb_path)
sys.path.insert(0, nb_path)

cd /content/mnt/My Drive/Colab Notebooks/Fashion_MNIST

import pandas as pd
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
