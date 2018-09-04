# -*- coding: utf-8 -*-
"""
Created on Tue Sep  4 09:08:03 2018

@author: hujia
"""

import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
import numpy as np
import os

tf.reset_default_graph()
config=tf.ConfigProto()
config.gpu_options.allow_growth = True


def generate_data():
    train_x=np.random.rand(10000,200)
    train_y=np.ones((10000,136))
    test_x=np.random.rand(1000,200)
    test_y=np.ones((1000,136))
    return train_x,train_y,test_x,test_y
train_x,train_y,test_x,test_y=generate_data()
'''
超参数的定义
'''
lr=1e-3
input_size=20
timestep_size=10
hidden_size=256
layer_num=2
output_num=136
epochs=100
batch_size=100


model_dir='model'
'''
占位符
'''

x_input=tf.placeholder(tf.float32,[None,input_size*timestep_size],name='input_x')
y_input=tf.placeholder(tf.float32,[None,output_num],name='input_y')
keep_prob=tf.placeholder(tf.float32,[])

'''
将输入转换为LSTM的输入
shape=[batch_szie,timesteps_size,input_size]
'''
X=tf.reshape(x_input,[-1,timestep_size,input_size])

'''
创建lstm结构 这里是个双层的网络
'''
def mlstm_model(hidden_size,keep_prob,layer_num):
    cell=tf.contrib.rnn.BasicLSTMCell
    #cell=tf.contrib.rnn.DropoutWrapper(cell,output_keep_prob=keep_prob)
    mlstm=tf.contrib.rnn.MultiRNNCell([cell(hidden_size) for _ in range(layer_num)])
    cell=tf.contrib.rnn.DropoutWrapper(mlstm,output_keep_prob=keep_prob)
    return cell

with tf.variable_scope("LSTM"):
    mlstm=mlstm_model(hidden_size,keep_prob,layer_num)
    '''
    初始化网络
    '''
    '''
    outputs=[]
    mlstm=mlstm_model(hidden_size,keep_prob,layer_num)
    state=mlstm.zero_state(batch_size,dtype=tf.float32)
    with tf.variable_scope('LSTM'):
        for  timestep in range(timestep_size):
            #if i>0:tf.get_variable_scope().reuse_variables()
            cell_output,state=mlstm(X[:,timestep,:],state)
            outputs.append(cell_output)
    h_state = outputs[-1]
    '''
    outputs,_=tf.nn.dynamic_rnn(mlstm,X,dtype=tf.float32)
    output=outputs[:,-1,:]
    w=tf.Variable(tf.random_normal((hidden_size,output_num)),name="w")
    b=tf.Variable(tf.ones((output_num)),name='bais')
    pred=tf.nn.relu(tf.matmul(output,w)+b,name='prd')
    #pred=tf.contrib.layers.fully_connected(output,output_num)
#损失函数
loss=tf.losses.mean_squared_error(labels=y_input,predictions=pred)
train_op=tf.train.AdamOptimizer(lr).minimize(loss)



saver=tf.train.Saver()
init_op=tf.global_variables_initializer()

with tf.Session(config=config) as sess:
    sess.run(init_op)
    checkpoint=tf.train.latest_checkpoint(model_dir)
    if checkpoint:
        saver.restore(sess,checkpoint)
        print("private model has been loaded!")
    else:
        print("No model be load!")
    print("Start training.....")
    for epoch in range(epochs):
        for batch in range(int(len(train_x)/batch_size)):
          
            train_loss,_=sess.run([loss,train_op],feed_dict={x_input:train_x[batch*batch_size:batch*batch_size+batch_size],
                                                         y_input:train_y[batch*batch_size:batch*batch_size+batch_size],
                                                         keep_prob:0.8})
            if batch%100==0:
                predict=sess.run(pred,feed_dict={x_input:test_x,keep_prob:0.8})
                test_loss=sess.run(loss,feed_dict={x_input:test_x,y_input:test_y,keep_prob:0.8})
                print("pred:",predict)
                print("Epoch:%d Step:%d Train loss:%f Test loss:%f"%(epoch,batch,train_loss,test_loss))
        saver.save(sess,os.path.join(model_dir,"test"))
    
    


    








