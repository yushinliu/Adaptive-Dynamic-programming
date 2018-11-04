import tensorflow as tf
import numpy as np 
import os


Class ADP(object):
      def __init__(self,w_initializer=,b_initializer=):
          

          # implement goal network
          tf.variable_scope("goal_network"):
             weights_1= tf.get_variable("gn_01_w",[input_layer,hidden_layer],initializer=w_initializer)
             bias_1 = tf.get_variable("gn_01_b",[hidden_layer],initializer=b_initializer)
             tensor=tf.add(tf.matmul(x,weights_1),bias_1)
             weights_2= tf.get_variable("gn_02_w",[input_layer,hidden_layer],initializer=w_initializer)
             bias_2 = tf.get_variable("gn_02_b",[hidden_layer],initializer=b_initializer)
             tensor=tf.add(tf.matmul(tensor,weights_1),bias_1)

          # implement critic network
          tf.variable_scope("critic_network"):
             weights_1= tf.get_variable("cn_01_w",[input_layer,hidden_layer],initializer=w_initializer)
             bias_1 = tf.get_variable("cn_01_b",[hidden_layer],initializer=b_initializer)
             tensor=tf.add(tf.matmul(x,weights_1),bias_1)
             weights_2= tf.get_variable("cn_02_w",[input_layer,hidden_layer],initializer=w_initializer)
             bias_2 = tf.get_variable("cn_02_b",[hidden_layer],initializer=b_initializer)
             tensor=tf.add(tf.matmul(tensor,weights_1),bias_1)

          #  implement action network
          tf.variable_scope("action_network"):
             weights_1= tf.get_variable("an_01_w",[input_layer,hidden_layer],initializer=w_initializer)
             bias_1 = tf.get_variable("an_01_b",[hidden_layer],initializer=b_initializer)
             tensor=tf.add(tf.matmul(x,weights_1),bias_1)
             weights_2= tf.get_variable("an_02_w",[input_layer,hidden_layer],initializer=w_initializer)
             bias_2 = tf.get_variable("an_02_b",[hidden_layer],initializer=b_initializer)
             tensor=tf.add(tf.matmul(tensor,weights_1),bias_1)


