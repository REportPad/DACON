##1. 필요한 라이브러리를 준비합니다.
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Dense,Flatten
from keras.optimizers import Adam
from keras.callbacks import TensorBoard

##2. 데이터를 불러옵니다.
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

##3. 데이터를 전처리합니다
train_data = np.array(train_df.iloc[:,1:], dtype = 'float32')
test_data = np.array(test_df.iloc[:,1:], dtype='float32')
x_train = train_data[:,1:]/255
y_train = train_data[:,0]
x_test= test_data/255
x_train,x_validate,y_train,y_validate = train_test_split(x_train,y_train,test_size = 0.2,random_state = 12345)
x_train = x_train.reshape(x_train.shape[0],28,28,1)
x_test = x_test.reshape(x_test.shape[0],28,28,1)
x_validate = x_validate.reshape(x_validate.shape[0],28,28,1)

##4. CNN 모델을 구축합니다.
cnn_model = Sequential([
    Conv2D(filters=32,kernel_size=3,activation = 'relu',input_shape = (28,28,1)),
    MaxPooling2D(pool_size=2),
    Flatten(),
    Dense(32,activation = 'relu'),
    Dense(10,activation = 'softmax')
])
cnn_model.compile(loss ='sparse_categorical_crossentropy', optimizer=Adam(lr=0.001),metrics =['accuracy'])

##5. 구축한 모델을 학습합니다.
history = cnn_model.fit(
    x_train,
    y_train,
    batch_size=512,
    epochs=20,
    verbose=1,
    validation_data=(x_validate,y_validate),
)

##6. 학습한 모델로 test 데이터의 범주(클래스)를 예측합니다.
y_pred = cnn_model.predict_classes(x_test)

##7. 예측값을 sample_submission 양식에 맞게 저장합니다.
submission = pd.read_csv('sample_submission.csv', encoding = 'utf-8')
submission['label'] = y_pred
submission.to_csv('fashion_submission.csv', index = False)


