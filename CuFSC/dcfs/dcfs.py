import warnings

import pandas as pd
import numpy as np
import os
import shutil
import sys
import timeit
import tensorflow.keras.backend as K
from collections import deque
import threading
import pickle

sys.path.append(os.path.abspath(''))

from dcfs.agent import Agent
from dcfs.enviorement import Enviorement
from dcfs.general_functions import genreal_func
from dcfs.run_process import run_process


genreal_func = genreal_func()
run_process = run_process()
warnings.filterwarnings('ignore')
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'     
        
class DCFS:
    def __init__(self, number_of_features_in_data, discount_rate, print_log=True, 
                 threads=5,  rnn_unit=10, external_threshold =  0,
                internal_threshold = -1000, learner_model ='DT'):
#         self.dataset = dataset
        self.discount_rate = discount_rate
        self.final_threads = threads
        self.rnn_unit = rnn_unit
        self.copy = threads
        self.print_log = print_log
        
        self.current_lr = 0 
        self.external_threshold =  0
        self.internal_threshold = -1000
        self.learner_model ='DT'        
                
        self.x_memory =  deque()
        self.y_memory =  deque()
        self.a_memory =  deque() 
        
        K.clear_session()

        
        self.agent = Agent(self.internal_threshold, self.external_threshold, self.rnn_unit, number_of_features_in_data)
        
        if self.print_log:
            print ('discount_factor:  ' + str(self.discount_rate))        
            print('number_of_features_in_data:' + str(number_of_features_in_data))
        
    def run_dcfs(self, dataset, episodes, l_r, eps,episode_size):
        
        env = Enviorement(dataset, self.learner_model)
        if self.print_log:
            print('number of episodes: {}'.format(episodes))

        for epi in range(episodes):
            x_for_train = []
            y_for_train = []
            a_for_train = []

            if self.print_log:
                print('episode {} start'.format(epi))


            if l_r != self.current_lr:    
                if self.print_log:
                    print(f"complie model with lr ={l_r}")
#                     print(f"agent parameters ={self.agent.full_model.get_weights()}")
                self.agent.complie(l_r)
                self.current_lr = l_r

            threads = self.final_threads
            n_memory = threads
            batch_size =  threads

            threadLock = threading.Lock()
            jobs = []

            for i in range(0, threads):
                thread = actorthread(i, threadLock, self.discount_rate)
                jobs.append(thread)

            for j in jobs:
                X_batch,a_batch, y_batch =  j.run(env, self.agent, episode_size, eps)
                x_for_train.append(X_batch)
                y_for_train.append(y_batch)
                a_for_train.append(a_batch)


            x_for_train = np.array(x_for_train)

            x_for_train= x_for_train.reshape(x_for_train.shape[0], x_for_train.shape[2],x_for_train.shape[3])

            y_for_train = np.array(y_for_train)

            a_for_train = np.array(a_for_train)

            for raw in x_for_train:
                self.x_memory.append(raw)
                if len(self.x_memory) > n_memory:
                    self.x_memory.popleft()                
            for raw in y_for_train:
                self.y_memory.append(raw)
                if len(self.y_memory) > n_memory:
                    self.y_memory.popleft()
            for raw in a_for_train:
                self.a_memory.append(raw)
                if len(self.a_memory) > n_memory:
                    self.a_memory.popleft()

            self.agent.train([np.array(self.x_memory),
                              np.array(self.a_memory).reshape(np.array(self.a_memory).shape[0],np.array(self.a_memory).shape[1],1)],
                              np.array(self.y_memory).reshape(np.array(self.y_memory).shape[0],np.array(self.y_memory).shape[2],
                              np.array(self.y_memory).shape[1]), min(np.array(self.x_memory).shape[0],batch_size))
            self.agent.updae_model_for_infereance_weights()

            if epi%self.copy == 0:
                if self.print_log:   
                    print ("update traget network")
                self.agent.updae_model_for_infereance_weights()

            ## policy results        
            policy_s = env.action_space      
            policy_selected_actions = env.action_space #new
            policy_done = 0
            policy_order = []
            policy_a_Q_list = []

            self.agent.clear_inference_model_state() 

            while not policy_done:
                policy_a , policy_a_Q = self.agent.act(policy_s, 1, eps, policy_selected_actions)
                policy_order.append(policy_a)
                policy_a_Q_list.append(policy_a_Q)
                if policy_a_Q >= self.agent.external_threshold:
                    policy_s2, policy_selected_actions = env.s2(policy_s, policy_a, policy_selected_actions)
                    policy_done = self.agent.is_done(policy_a_Q, policy_selected_actions, policy_calc=1)
                    policy_s = policy_s2
                else: 
                    policy_done = 1


            policy_columns = np.where(policy_selected_actions==1)[0]

            if self.print_log:                
                print('policy order: {}'.format(policy_order))

#         print(f"agent parameters ={self.agent.full_model.get_weights()}")
        return policy_order



class actorthread(threading.Thread):
    def __init__(self,thread_id,threadLock, discount_rate ):
        threading.Thread.__init__(self)
        self.thread_id = thread_id
        self.threadLock = threadLock
        self.discount_rate =  discount_rate
        
    def run(self, env, agent, episode_size, eps):
        start = timeit.default_timer()

        self.threadLock.acquire()
        
        episode_memory_store = run_process.runprocess(env, agent, episode_size, eps, self.discount_rate)

        if np.array(episode_memory_store).shape[0]>0:
            episode_memory_store = episode_memory_store.reshape(1,episode_memory_store.shape[0], episode_memory_store.shape[1])

            for episode_memory in episode_memory_store:

                X_batch,a_batch, y_batch = genreal_func.create_batch(np.array(episode_memory),self.discount_rate, agent)

        self.threadLock.release()

        return X_batch,a_batch, y_batch



