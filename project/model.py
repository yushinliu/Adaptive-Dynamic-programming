import tensorflow as tf
import numpy as np 
import os
import gym


Class ADP(object):
    def __init__(self,w_initializer=,b_initializer=):
          

        # implement goal network
        tf.variable_scope("goal_network"):
           weights_1= tf.get_variable("gn_01_w",[input_layer,hidden_layer],initializer=w_initializer)
           bias_1 = tf.get_variable("gn_01_b",[hidden_layer],initializer=b_initializer)
           tensor=tf.add(tf.matmul(x,weights_1),bias_1)
           tensor=tf.nn.relu(tensor) #dont know, assume it is relu
           weights_2= tf.get_variable("gn_02_w",[input_layer,hidden_layer],initializer=w_initializer)
           bias_2 = tf.get_variable("gn_02_b",[hidden_layer],initializer=b_initializer)
           tensor=tf.add(tf.matmul(tensor,weights_1),bias_1)
           s_now=tf.nn.sigmoid(tensor)

        # implement critic network
        tf.variable_scope("critic_network"):
           weights_1= tf.get_variable("cn_01_w",[input_layer,hidden_layer],initializer=w_initializer)
           bias_1 = tf.get_variable("cn_01_b",[hidden_layer],initializer=b_initializer)
           tensor=tf.add(tf.matmul(x,weights_1),bias_1)
           tensor=tf.nn.relu(tensor) #dont know, assume it is relu
           weights_2= tf.get_variable("cn_02_w",[input_layer,hidden_layer],initializer=w_initializer)
           bias_2 = tf.get_variable("cn_02_b",[hidden_layer],initializer=b_initializer)
           tensor=tf.add(tf.matmul(tensor,weights_1),bias_1)
           V_now=tf.nn.sigmoid(tensor)

        #  implement action network
        tf.variable_scope("action_network"):
           weights_1= tf.get_variable("an_01_w",[input_layer,hidden_layer],initializer=w_initializer)
           bias_1 = tf.get_variable("an_01_b",[hidden_layer],initializer=b_initializer)
           tensor=tf.add(tf.matmul(x,weights_1),bias_1)
           tensor=tf.nn.relu(tensor) #dont know, assume it is relu
           weights_2= tf.get_variable("an_02_w",[input_layer,hidden_layer],initializer=w_initializer)
           bias_2 = tf.get_variable("an_02_b",[hidden_layer],initializer=b_initializer)
           tensor=tf.add(tf.matmul(tensor,weights_1),bias_1)
           a_now=tf.nn.sigmoid(tensor)

        loss_goal=0.5*tf.square_difference(V,U_c)
        loss_action=
        loss_critic=


        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate) 
        opt_goal = optimizer.minimize(loss_goal)
        opt_action = optimizer.minimize(loss_action)
        opt_critic = optimizer.minimize(loss_critic)
    def run(self,T_a=,T_c=,T_g=,max_run=,N_a,N_c,N_g):
        init = tf.initialize_all_variables()
        with tf.Session() as sess:
        ob_lst,reward_lst,done_lst,reward_s_lst=[],[],[],[]
            for i in range(max_run):
                sess.run(init)
                observation, reward, done, info = env.step(action)
                # implement goal network
                loss_goal=sess.run(loss_goal,feed_dict={})
                while (loss_goal > T_g) and (i < N):
                    _,s_now=sess.run([opt_goal,s_now],feed_dict={})
					reward_s_lst.append(s_now)
                while (loss_goal > T_g) and (i < N):
                    _,s_now=sess.run([opt_goal,s_now],feed_dict={})
					reward_s_lst.append(s_now)
                while (loss_goal > T_g) and (i < N):
                    _,s_now=sess.run([opt_goal,s_now],feed_dict={})
					reward_s_lst.append(s_now)








if __name__ == "__main__" :
    env = gym.make('CartPole-v0')

    env.reset()
    while random_episodes < 10:
        env.render()
        observation, reward, done, _ = env.step(np.random.randint(0,2))
        reward_sum += reward
        if done:
            random_episodes += 1
            print("Reward for this episode was:",reward_sum)
            reward_sum = 0
            env.reset()
	adp = ADP()
	adp.run()

