import tensorflow as tf

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def FC_la(input,out_size=None,name='FC',reuse=False):
  in_channel = int(input.get_shape()[-1])
  x = tf.reshape(input, [-1,in_channel]) if len(input.get_shape())>2 else input
  out = tf.contrib.layers.fully_connected(x, out_size, 
                                          scope=name,reuse=reuse,activation_fn=None,weights_initializer=tf.contrib.layers.xavier_initializer())
  return out

def conv2d(x, W):
  return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

def CA(input,pool=tf.reduce_mean, activation_function=tf.nn.relu, squash=tf.nn.sigmoid,name=None,r=4,reuse=Fales):
  mu = pool(input,reduction_indices=[1,2],keep_dims=True)
  c = int(mu.get_shape()[-1])
  inter_size = int(c/r)
  inter_actives = activation_function(FC_la(
                  input=tf.contrib.layers.flatten(input),out_zier=inter_size, name='CA_inter',reuse=reuse))
  
  output_actives = FC_la(input=inter_actives,out_size=c,name='CA_out',reuse=reuse)
  
  output_actives = squash(output_actives)
  ep_actives = tf.reshape(output_actives,[-1,1,1,c])
  
  output_final = input*ep_actives + input
  
  return output_final

def LSA(input,pool=tf.reduce_mean, activation_function=tf.nn.relu, squash=tf.nn.sigmoid,inter_kernel=1,name=None,r=4,reuse=Fales):
  c = int(input.get_shape()[-1])
  inter_channels = int(c/r)
  inter_actives = tf.layers.conv2d(input=input,filters=inter_channels,kernel_size=inter_kernel,
                                   activation=tf.nn.relu,padding='SAME',use_bias=True,kernel_initializer=tf.variance_scaling_initializer(),
                                   name='LSA_inter',reuse=reuse)
  output_actives = tf.layers.conv2d(input=inter_actives,filters=1,kernel_size=inter_kernel,
                                    padding='SAME',use_bias=True,activation=None,kernel_initializer=tf.variance_scaling_initializer(),
                                    name='LSA_out',reuse=reuse)
  output_actives = squash(output_actives)
  output_final = input*output_actives +input
  
  return output_final

def GSA(input,name=None):
  h = int(input.ger_shape()[-3])
  w = int(input.ger_shape()[-2])
  c = int(input.ger_shape()[-1])
  
  
  theta_w_conv = weight_variable([1,1,c,c])
  theta_b_conv = bias_variable([c])
  theta = conv2d(input,theta_w_conv)+theta_b_conv
  theta = tf.reshape(theta,shape=[-1,h*w,c])
  
  phi_w_conv = weight_variable([1,1,c,c])
  phi_b_conv = bias_variable([c])
  phi = conv2d(input,phi_w_conv)+phi_b_conv
  phi = tf.reshape(phi,shape=[-1,h*w,c])
  f = tf.matmul(theta, phi, transpose_b=True)
  
  phi_shape = phi.get_shape().as_list()
  f = tf.reshape(f, shape=[-1,h*w,phi_shape[-1])
                           
                                      
