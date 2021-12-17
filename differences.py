import tensorflow as tf
import numpy as np
import attentions
import sys


def difference_enhanced(input, clips=6):
  diff = []
  for i in range(clips):
    diff_m = tf.nn.conv2d(tf.reduce_mean(input[:,i+1,:,:,:],[1,2],keep_dims=True),
                          filter=tf.Variable(tf.truncated_normal(shape=[1,1,2048,512],stddev=0.1,seed=1)),
                          strides=[1,1,1,1],padding='SAME') -
            tf.nn.conv2d(tf.reduce_mean(input[:,i,:,:,:],[1,2],keep_dims=True),
                          filter=tf.Variable(tf.truncated_normal(shape=[1,1,2048,512],stddev=0.1,seed=1)),
                          strides=[1,1,1,1],padding='SAME'))
    diff_m = tf.nn.sigmoid(tf.nn.conv2d(diff_m,filter=tf.Variable(tf.truncated_normal(shape=[1,1,2048,512],stddev=0.1,seed=1)),
                                       strides=[1,1,1,1],padding='SAME'))
   
    diff_m = tf.multiply(input[:,i,:,:,:],diff_m)
    diff.append(diff_m)
  diff.append(input[:,-1,:,:,:])
  difference = tf.convert_to_tensor(diff)
  difference = tf.transpose(difference,[1,0,2,3,4])
  
  return difference
  
