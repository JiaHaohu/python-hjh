# -*- coding: utf-8 -*-
"""
Created on Tue Sep  4 16:20:18 2018

@author: hujia
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
tf.reset_default_graph()
#超参数设置
hidden_size=30
num_layers=2
timesteps=10
epochs=3000
batch_size=32
training_num=10000
test_num=1000
gap=0.01
lr=1e-3

def generate_data(seq):
    x=[]
    y=[]
    for i in range(len(seq)-timesteps):
        x.append([seq[i:i+timesteps]])
        y.append([seq[i+timesteps]])
    return np.array(x,dtype=np.float32),np.array(y,dtype=np.float32)

test_start=(training_num+timesteps)*gap
test_end=test_start+(timesteps+test_num)*gap
train_x,train_y=generate_data(np.sin(np.linspace(0,test_start,training_num+timesteps,dtype=np.float32)))
train_x, train_y = generate_data(np.sin(np.linspace(
    0, test_start,training_num+timesteps,dtype=np.float32)))
test_x, test_y = generate_data(np.sin(np.linspace(
    test_start, test_end, test_num+timesteps, dtype=np.float32)))

def lstm_model(x,y,is_training):
    cell=tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.BasicLSTMCell(hidden_size) for _ in range(num_layers)])
    outputs,_=tf.nn.dynamic_rnn(cell,x,dtype=tf.float32)
    output=outputs[:,-1,:]
    pred=tf.contrib.layers.fully_connected(output,1,activation_fn=None)
    if not is_training:
        return pred,None,None
    loss=tf.losses.mean_squared_error(labels=y,predictions=pred)
    train_op=tf.train.AdamOptimizer(lr).minimize(loss)
    return pred,loss,train_op

def run_eval(sess,test_x,test_y):
    ds=tf.data.Dataset.from_tensor_slices((test_x,test_y))
    ds=ds.batch(1)
    x,y=ds.make_one_shot_iterator().get_next()

    with tf.variable_scope('model',reuse=True):
        pred,_,_=lstm_model(x,[0.0],False)
    predictions=[]
    labels=[]
    for i in range(test_num):
        p,l=sess.run([pred,y])
        predictions.append(p)
        labels.append(l)
    predictions=np.array(predictions).squeeze()
    labels=np.array(labels).squeeze()
    rmse=np.sqrt(((predictions-labels)**2).mean(axis=0))
    print("RMSE:%f"%rmse)
    
    plt.figure()
    plt.plot(predictions,label='pred')
    plt.plot(labels,label='real')
    plt.legend()
    plt.show()

    
ds=tf.data.Dataset.from_tensor_slices((train_x,train_y))
ds=ds.repeat().shuffle(1000).batch(batch_size)
x,y=ds.make_one_shot_iterator().get_next()

with tf.variable_scope('model'):
    _,loss,train_op=lstm_model(x,y,True)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print("训练之前")
    run_eval(sess,test_x,test_y)
    for i in range(epochs):
        
        _,l=sess.run([train_op,loss])
        if i%1000==0:
            print('loss',l)

    print('训练之后')
    run_eval(sess,test_x,test_y)