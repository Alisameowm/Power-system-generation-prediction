import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(threshold=np.inf)  ####数字省略不全  输出
import pandas as pd
import math
from tensorflow.keras.models import Sequential
from  tensorflow.keras.layers import Input, Dense, LSTM ,Conv1D,Dropout,Bidirectional,Multiply,Concatenate,BatchNormalization
from tensorflow.keras.layers import Dense, Activation, Dropout, LSTM,LeakyReLU,GRU
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras import optimizers
from sklearn.model_selection import train_test_split
import time
from tensorflow.keras .optimizers import  Adam
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Model
from tensorflow.keras.models import *
from tensorflow.keras.utils import get_custom_objects
import re
import tensorflow as tf
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras import *
import numpy as np
import os
###OS提供了非常丰富的方法用来处理文件和目录
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dense, Input, Dropout ,Bidirectional
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
import re
from tensorflow.keras.callbacks import Callback
import tensorflow
tf.random.set_seed(1234)
def attention_function(inputs, single_attention_vector=False):
    TimeSteps = K.int_shape(inputs)[1]
    input_dim = K.int_shape(inputs)[2]
    a = Permute((2, 1))(inputs)
    a = Dense(TimeSteps, activation='softmax')(a)
    if single_attention_vector:
        a = Lambda(lambda x: K.mean(x, axis=1))(a)
        a = RepeatVector(input_dim)(a)
    a_probs = Permute((2, 1))(a)
    output_attention_mul = Multiply()([inputs, a_probs])
    return output_attention_mul
def creat_dataset(dataset, look_back=30):#根据时间序列构建建模所需要的数据 大致方法为 假如原始的数据是8 5 6 9 3 1 4 2 8 7 3 7 4 6，
    #那么这里的look_back用来规定特征数，则构建的第一个特征为[8 5 6 9 3 1 4 2],该特征的目标值为8(第九个数据)，同时第二个特征为[5 6 9 3 1 4 2 8],
    #, 该特征的目标值为7(第10个数据),类似于滑窗的原理
    ####look_back=8   前8个数据预测第9个数据
    #print("dataset",dataset[1,:],type(dataset),dataset.shape)##### class numpy ndnarry    (5376,10)
    dataX, dataY = [], []
    #print("dataX",dataX,type(dataX))#####list没有形状  dataX [] <class 'list'>
    for i in range(len(dataset)-look_back-1):#5345
        #print(len(dataset)-look_back-1)   5345
        a = dataset[:,0:8][i: (i+look_back)]
        #print(len(a))  30
        # print("a[0]",a[0],len(a[0]))   #30行 10列
        dataX.append(a)   ####list用append
        # print("dataX",dataX[0:5])
        dataY.append(dataset[i+look_back,0])
        #print(len(dataset[i+look_back,0]))  r     浮点型 TypeError: object of type 'numpy.float64' has no len()
      #  print("dataset[i+look_back,0]",dataset[i+look_back,0])
    return np.array(dataX), np.array(dataY)


dataframe = pd.read_excel('testdata.xlsx')


#pd.read_excel   读取excel的数据
#dataframe = dataframe.drop(labels ='time',axis=1)
#print("dataframe",type(dataframe)) #dataframe <class 'pandas.core.frame.DataFrame'>
dataset = dataframe.iloc[3:,2:].values
print("----------------------------------------------------------------------------------------------------dataset ",dataset.shape )









#print("dataset",type(dataset))  #<class 'numpy.ndarray'>
###读取里面的数据
scalerX = MinMaxScaler(feature_range=(0, 1)) ##数据转换的范围是[0,1]，scaler代表min和max

#范围区（0，1）
dataset[:,0:8] = scalerX.fit_transform(dataset[:,0:8])#-1到1归一化
scalerY = MinMaxScaler(feature_range=(0, 1))
#dataset[:,0] = scalerY.fit_transform(dataset[:,0].reshape(-1, 1)).squeeze(1)#-1到1归一化####renshape 转换成1列
dataset[:,8] = scalerY.fit_transform(dataset[:,8].reshape(-1,1)).squeeze(1)#0.69569503 0.65530442
#ValueError: could not broadcast input array from shape (5376,1) into shape (5376)
#ValueError: could not broadcast input array from shape (5376,1) into shape (5376) r   降低维度
#print(dataset[:,0] ,dataset[:,0].shape,type(dataset[:,0]))#(5376,) <class 'numpy.ndarray'>



# from sklearn.externals import joblib
# joblib.dump(scalerX, 'scalerX.pkl')
# joblib.dump(scalerY, 'scalerY.pkl')

look_back = 10
X, Y = creat_dataset(dataset, look_back)#创建训练集的滑窗数据
#print("Y",type(Y),Y.shape) #<class 'numpy.ndarray'>   (5345,)
#trainX,testX,trainY,testY = train_test_split(X,Y,test_size=0.1)
#print("X",type(X),X.shape)#X <class 'numpy.ndarray'> (5345, 30, 10)
all = X.shape[0]
print("all ",type(all))#AttributeError: 'int' object has no attribute 'shape'
#all  <class 'int'>

train_size = int(all*0.8)
trainX = X[0:train_size].astype('float64')
#print("trainX",trainX.shape)#trainX (4810, 30, 10)
testX = X[train_size:].astype('float64')
# print("testX",testX.shape)#testX (535, 30, 10)
trainY = Y[0:train_size].astype('float64')
print("type(trainY)",type(trainY),trainY.shape)#type(trainY) <class 'numpy.ndarray'> (4810,)
testY = Y[train_size:].astype('float64')

def attention_model():
    inputs = Input(shape=(trainX.shape[1],trainX.shape[2] ))#输入归一化后的
    BiLSTM_out = Bidirectional(LSTM(95, return_sequences=True, activation="relu"))(inputs)
    GRU_OUT=GRU(units=32,return_sequences=True,activation="relu")(BiLSTM_out)#先经过一个GRU门控循环单元，目的是改善梯度消失问题，激活函数是Relu函数
   #双向lstm（更新门+遗忘门），激活函数是Relu函数
    Batch_Normalization = BatchNormalization()(GRU_OUT)#标批标准化
    Drop_out = Dropout(0.1)(Batch_Normalization)#随机丢失神经单元，0.1是丢失率
    attention = attention_function(Drop_out)#注意力机制应用在卷积层
    Batch_Normalization = BatchNormalization()(attention)#批标准化
    Drop_out = Dropout(0.1)(Batch_Normalization)#随机丢失
    Flatten_ = Flatten()( Drop_out )#压平，作用于卷积层和全连接层
    output = Dropout(0.1)(Flatten_)#随机丢失
    output = Dense(1, activation='sigmoid')(output)#sigmoid函数作为最后一层一个神经元激活函数
    model = Model(inputs=[inputs], outputs=output)
    return model

model = attention_model()

model.compile(loss='mean_squared_error', optimizer='adam')#优化器adam
model.summary()
history = model.fit(trainX, trainY, batch_size=350, epochs=19,
                    validation_split=0.3, verbose=2)#模型训练

# print('compilatiom time:', time.time()-start)
y_pre = model.predict(testX)#模型预测
testY = testY.reshape(-1, 1)
y_pre= y_pre.reshape(-1, 1)
aaa = pd.DataFrame(testY)
bbb = pd.DataFrame(y_pre)



#结果可视化
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import seaborn as sns
#结果可视化
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
#print((mean_squared_error(testY,y_pre)))
#print("MAE",np.sqrt(mean_absolute_error(testY,y_pre)))
#print("RMSE",(mean_squared_error(testY,y_pre)))
#改

# print("MAE",mean_absolute_error(testY,y_pre))
print("改进MAE",(mean_absolute_error(testY,y_pre)))
print("改进",np.sqrt(mean_squared_error(testY,y_pre)))
print("R^2",r2_score(testY,y_pre))

import matplotlib.pyplot as plt
import seaborn as sns
num =1600
sns.set(style="white")
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 显示中文标签

plt.plot(range(len(y_pre)), y_pre,  'red',label="predicted-value")
plt.plot(range(len(testY)), testY, 'black', label='true-value')
#plt.plot(range(len(y_pre[num :1000])), y_pre[num :1000],  'r--',label="预测值")
#plt.plot(range(len(testY[num :1000])), testY[num :1000], 'b--', label='真实值')
plt.xlabel("样本点", fontsize=20)#预测的自变量
plt.ylabel("功率", fontsize=20)#因变量
plt.legend()
plt.title('facastingmodel')
plt.show()

plt.show()
# plt.grid()
plt.grid()
plt.show()




