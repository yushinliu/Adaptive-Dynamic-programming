import tensorflow as tf
import numpy as np 
import os
import gym

#hyper-parameters definition
GAMMA=1
U_C=0
MAX_RUN=100
N_g,N_c,N_a=10,10,10
e_g,e_c,e_a=-0.001,0.001,0.001

"""
Goal network
"""
class goal_network(object):
    def __init__(self,sess,w_initializer=tf.contrib.layers.xavier_initializer(),b_initializer=tf.zeros_initializer()):
        self.sess=sess
        self.w_initializer=w_initializer
        self.b_initializer=b_initializer
        self.learning_rate=0.1
        with tf.variable_scope("goal_net"):
            """
            weights_1= tf.get_variable("gn_01_w",[5,5],initializer=self.w_initializer) #dimensions : [input layer, hidden_layer]
            bias_1 = tf.get_variable("gn_01_b",[5],initializer=self.b_initializer)#dimensions : [hidden_layer]
            tensor=tf.add(tf.matmul(self.g_input,weights_1),bias_1)
            tensor=tf.nn.relu(tensor) #dont know, assume it is relu
            weights_2= tf.get_variable("gn_02_w",[5,1],initializer=self.w_initializer)#dimensions : [hidden_layer,output_layer]
            bias_2 = tf.get_variable("gn_02_b",[1],initializer=self.b_initializer)#dimensions : [output_layer]
            tensor=tf.add(tf.matmul(tensor,weights_2),bias_2)
            self.s_now=tf.nn.sigmoid(tensor)
            """
            self.g_input = tf.concat([observ, action_input], axis=1)
            hidden= tf.layers.dense(self.g_input, 5, kernel_initializer=self.w_initializer, bias_initializer=self.b_initializer,name="l1",activation=tf.nn.relu)
            self.s=tf.layers.dense(hidden, 1, kernel_initializer=self.w_initializer, bias_initializer=self.b_initializer,name="l2",activation=tf.nn.sigmoid)
        print("goal network init finish")
    def cal_loss(self,J_now,J_last,reward,gamma=GAMMA):
        loss=np.mean(0.5*(gamma*J_now-(J_last-reward)**2))
        return loss
    def update_gradient(self,pass_gradients):
        self.goal_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='goal_net')
        self.goal_grads = tf.gradients(ys=self.s, xs=self.goal_params, grad_ys= pass_gradients)
        opt = tf.train.AdamOptimizer(self.learning_rate)  # (- learning rate) for ascent policy
        self.train_op = opt.apply_gradients(zip(self.goal_grads, self.goal_params))
    def train(self,action,observation):
        _,signal=self.sess.run([self.train_op,self.s],feed_dict={observ:observation,action_input:action})
        return signal

    def test(self):
        #for var in tf.trainable_variables():
         #   print(var.name)
        g_var= tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='goal_net')
        var=self.sess.run(g_var)
            #print(var)

"""
Critic network
"""
class critic_network(object):
    def __init__(self,sess,s_now,a_now,w_initializer=tf.contrib.layers.xavier_initializer(),b_initializer=tf.zeros_initializer(),gamma=GAMMA):
        self.sess=sess
        self.J_last=tf.placeholder(tf.float32,[1,1],name="J_last")
        self.s_now=s_now
        self.a_now=a_now
        self.w_initializer=w_initializer
        self.b_initializer=b_initializer
        self.learning_rate=0.1
        with tf.variable_scope("critic_net"):
            self.g_input=tf.concat([observ,a_now],axis=1)
            self.g_input=tf.concat([self.g_input,s_now],axis=1)
            """
            weights_1= tf.get_variable("gn_01_w",[5,5],initializer=self.w_initializer) #dimensions : [input layer, hidden_layer]
            bias_1 = tf.get_variable("gn_01_b",[5],initializer=self.b_initializer)#dimensions : [hidden_layer]
            tensor=tf.add(tf.matmul(self.g_input,weights_1),bias_1)
            tensor=tf.nn.relu(tensor) #dont know, assume it is relu
            weights_2= tf.get_variable("gn_02_w",[5,1],initializer=self.w_initializer)#dimensions : [hidden_layer,output_layer]
            bias_2 = tf.get_variable("gn_02_b",[1],initializer=self.b_initializer)#dimensions : [output_layer]
            tensor=tf.add(tf.matmul(tensor,weights_2),bias_2)
            self.s_now=tf.nn.sigmoid(tensor)
            """
            hidden= tf.layers.dense(self.g_input, 5, kernel_initializer=self.w_initializer, bias_initializer=self.b_initializer,name="l1",activation=tf.nn.relu)
            self.J_now=tf.layers.dense(hidden, 1, kernel_initializer=self.w_initializer, bias_initializer=self.b_initializer,name="l2",activation=tf.nn.sigmoid)
            self.loss=tf.reduce_mean(0.5*tf.squared_difference(gamma*self.J_now,(self.J_last-s_now)))
            self.critic_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='critic_net')
            self.c2a_grads = tf.gradients(ys=self.J_now, xs=a_now)
            self.c2g_grads = tf.gradients(ys=self.J_now, xs=s_now)
            opt = tf.train.AdamOptimizer(self.learning_rate)  # (- learning rate) for ascent policy
            self.train_op = opt.minimize(self.loss)
        print("critic network init finish")
    def train(self,J_last,action,signal,observation):
        _,value,Loss=self.sess.run([self.train_op,self.J_now,self.loss],feed_dict={observ:observation,action_input:action,self.J_last:J_last})
        return value,Loss

    def test(self):
        #for var in tf.trainable_variables():
         #   print(var.name)
        c_var= tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic_net')
        var=self.sess.run(c_var)
        #print(var)

"""
Action network
"""
class action_network(object):
    def __init__(self,sess,w_initializer=tf.contrib.layers.xavier_initializer(),b_initializer=tf.zeros_initializer()):
        self.sess=sess
        self.w_initializer=w_initializer
        self.b_initializer=b_initializer
        self.learning_rate=0.1
        with tf.variable_scope("action_net"):
            """
            weights_1= tf.get_variable("gn_01_w",[5,5],initializer=self.w_initializer) #dimensions : [input layer, hidden_layer]
            bias_1 = tf.get_variable("gn_01_b",[5],initializer=self.b_initializer)#dimensions : [hidden_layer]
            tensor=tf.add(tf.matmul(self.g_input,weights_1),bias_1)
            tensor=tf.nn.relu(tensor) #dont know, assume it is relu
            weights_2= tf.get_variable("gn_02_w",[5,1],initializer=self.w_initializer)#dimensions : [hidden_layer,output_layer]
            bias_2 = tf.get_variable("gn_02_b",[1],initializer=self.b_initializer)#dimensions : [output_layer]
            tensor=tf.add(tf.matmul(tensor,weights_2),bias_2)
            self.s_now=tf.nn.sigmoid(tensor)
            """
            hidden= tf.layers.dense(observ, 5, kernel_initializer=self.w_initializer, bias_initializer=self.b_initializer,name="l1",activation=tf.nn.relu)
            self.a=tf.layers.dense(hidden, 1, kernel_initializer=self.w_initializer, bias_initializer=self.b_initializer,name="l2",activation=tf.nn.sigmoid)
        print("goal network init finish")
    def cal_loss(self,J_now,U_c=U_C):
        loss=np.mean(0.5*(J_now-U_c)**2)
        return loss
    def update_gradient(self,pass_gradients):
        self.action_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='action_net')
        self.action_grads = tf.gradients(ys=self.a, xs=self.action_params, grad_ys= pass_gradients)
        opt = tf.train.AdamOptimizer(self.learning_rate)  # (- learning rate) for ascent policy
        self.train_op = opt.apply_gradients(zip(self.action_grads, self.action_params))
    def train(self,observation,action):
        _,action=self.sess.run([self.train_op,self.a],feed_dict={observ:observation,action_input:action})
        return action

    def test(self):
        #for var in tf.trainable_variables():
         #   print(var.name)
        a_var= tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='action_network')
        var=self.sess.run(a_var)
            #print(var)


if __name__ == "__main__":
    #initialize the openai env
    env = gym.make('CartPole-v0')
    env.reset();
    random_episodes = 0
    reward_sum = 0
    while random_episodes < 10:
        env.render()
        observation, reward, done, action = env.step(np.random.randint(0,2))
        reward_sum += reward
        if done:
            random_episodes += 1
            print("Reward for this episode was:",reward_sum)
            reward_sum = 0
            env.reset()


    action_lst,signal_lst,value_lst=[],[],[0,0]
    loss_goal,loss_critic,loss_action=1,1,1
    cyc_g,cyc_c,cyc_a=0,0,0
    #set up input tensor
    with tf.variable_scope("Input"):
        observ=tf.placeholder(tf.float32,[1,4],name="observ")
        action_input=tf.placeholder(tf.float32,[1,1],name="action_input")
    #set up agent tensor graph
    sess=tf.Session()
    goal_net=goal_network(sess)
    action_net=action_network(sess)
    critic_net=critic_network(sess,s_now=goal_net.s,a_now=action_net.a)
    goal_net.update_gradient(critic_net.c2g_grads)
    action_net.update_gradient(critic_net.c2a_grads)
    sess.run(tf.global_variables_initializer())

    #initialize the input observation and action
    observation=observation.reshape(1,4)
    action = np.array(env.action_space.sample()).reshape(1,1)
    epoch=0
    done=0
    while epoch < MAX_RUN:
        #tuning the goal network
        while  cyc_g < N_g and loss_goal > e_g:
            signal=goal_net.train(action,observation)
            loss_goal=goal_net.cal_loss(value_lst[-1],value_lst[-2],np.int(done)) #according to paper A three-network architecture for on-line learning and optimization

            cyc_g+=1
            print("goal ",loss_goal)
#        signal_lst.append(signal)
        #tuning the critic network
        value_tmp_lst=[]
        value_tmp_lst.append(value_lst[-1])
        while cyc_c < N_c and loss_critic > e_c:
            value_last=np.array(value_tmp_lst[-1]).reshape(1,1)
            value_now,loss_critic=critic_net.train(value_last,action,signal,observation)
            value_tmp_lst.append(value_now)
            cyc_c+=1
            print("critic ",loss_critic)
        value_lst.append(value_now)
        #tuning the action network
        while cyc_a < N_a and loss_action > e_a:
            action=action_net.train(observation,action)
            loss_action=action_net.cal_loss(value_lst[-1])
            cyc_a+=1
            print("action ",loss_action)
        action=np.int(np.round(action)[0][0])
        #env.render()
        observation, reward, done, info = env.step(action)
        if done:
            env.reset()
            epoch+=1
        print("Epoch :",epoch)#," action :",action," observation :",observation," done :",done)
        action = np.array(action).reshape(1, 1)
