
from __future__ import absolute_import

from skimage.color import rgb2gray
from skimage.transform import resize
from keras.layers import Dense, Flatten, Input
from keras.layers import LSTM, Dropout


from keras.layers.convolutional import Conv2D
from keras.optimizers import RMSprop
from keras import backend as K
from keras.models import Model
import tensorflow as tf
import numpy as np
import threading
import random
import time

#import gym

import sys


import pickle
import argparse

import skimage.io as io
import skimage.transform as transform

from actions import command2action, generate_bbox, crop_input


import network
import network_vfn as nw

from os import listdir
from os.path import isfile, join

from datetime import datetime


a3c_graph  = tf.get_default_graph()

global_dtype = tf.float32

global_dtype_np = np.float32

#vfn_sess = None

# 멀티쓰레딩을 위한 글로벌 변수
global episode
episode = 0
EPISODES = 8000000
# 환경 생성
env_name = "BreakoutDeterministic-v4"

drop_ratio = 0.5

# This is the definition of helper function
# input : original image
def evaluate_aesthetics_score(images):

    scores = np.zeros(shape=(len(images),))
    features = []
    for i in range(len(images)):
        img = images[i].astype(np.float32)/255
        img_resize = transform.resize(img, (227, 227))-0.5
        img_resize = np.expand_dims(img_resize, axis=0)
        score  , feature = vfn_sess.run([ score_func ], feed_dict={image_placeholder: img_resize})[0]
        scores[i] = score
        features.append( feature)
    return scores , features

def evaluate_aesthetics_score_resized(images):

    scores = np.zeros(shape=(len(images),))
    features = []
    for i in range(len(images)):
        #img = images[i].astype(np.float32)/255
        #img_resize = transform.resize(img, (227, 227))-0.5
        img = images[i].astype(np.float32)
        img_resize = img
        img_resize = np.expand_dims(img_resize, axis=0)
        score  , feature = vfn_sess.run([ score_func ], feed_dict={image_placeholder: img_resize})[0]
        scores[i] = score
        features.append( feature)
    return scores , features



# 브레이크아웃에서의 A3CAgent 클래스(글로벌신경망)
class A3CAgent:
    def __init__(self, action_size):
        global a3c_graph
        # 상태크기와 행동크기를 갖고옴
        self.state_size = (1, 2000)
        self.action_size = action_size
        # A3C 하이퍼파라미터
        self.discount_factor = 0.99
        self.no_op_steps = 30
        self.actor_lr = 2.5e-4
        self.critic_lr = 2.5e-4
        self.beta = 0.05
        # 쓰레드의 갯수
        self.threads = 1
        #self.threads = 8


        # 정책신경망과 가치신경망을 생성
        self.actor, self.critic = self.build_model()
        # 정책신경망과 가치신경망을 업데이트하는 함수 생성
        self.optimizer = [self.actor_optimizer(), self.critic_optimizer()]

        # 텐서보드 설정

        print(' Parent   default graph', tf.get_default_graph())

        self.sess = tf.InteractiveSession()
        K.set_session(self.sess)
        self.sess.run(tf.global_variables_initializer())
        #self.load_model("../save_model/A2RL_a3c_run-20181112124128") 
        #self.load_model("../save_model/A2RL_a3c_run-20181112221820") 
        #self.load_model("../save_model/A2RL_a3c_run-20181113044510") 


        self.summary_placeholders, self.update_ops, self.summary_op = \
            self.setup_summary()
        self.summary_writer = \
            tf.summary.FileWriter('../summary/A2RL_a3c', self.sess.graph)
        #tf.summary.FileWriter(logdir, self.sess.graph)




    # 쓰레드를 만들어 학습을 하는 함수
    def train(self):
        # 쓰레드 수만큼 Agent 클래스 생성
        agents = [Agent(self.action_size, self.state_size,
                        [self.actor, self.critic], self.sess,
                        self.optimizer, self.discount_factor,
                        [self.summary_op, self.summary_placeholders,
                         self.update_ops, self.summary_writer])
                  for _ in range(self.threads)]

        # 각 쓰레드 시작
        for agent in agents:
            time.sleep(1)
            agent.start()

        # 10분(600초)에 한번씩 모델을 저장
        while True:
            time.sleep(  60 * 10)
            #time.sleep(60*5)

            #now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
            now = datetime.now().strftime("%Y%m%d%H%M%S")
            print('now ==' , now )
            # root_logdir = "tf_logs"  # save_model/A2RL_a3c
            root_logdir = "../save_model"  # save_model/A2RL_a3c
            logdir = "{}/A2RL_a3c_run-{}".format(root_logdir, now)
            print('logdir ==', logdir)

            self.save_model(logdir)

            print('sysexit ==', logdir)
            #sys.exit(0)

    # 정책신경망과 가치신경망을 생성
    def build_model(self):

        print(' Parent Model  default graph', tf.get_default_graph())
        K.set_learning_phase(1)  # set learning phase

        input = Input(shape = self.state_size )

        fc1 = Dense(1024, activation = 'relu') (input)
        #drop1 = Dropout(drop_ratio)(fc1)

        fc2 = Dense(1024, activation='relu')(fc1)
        #drop2 = Dropout(drop_ratio)(fc2)

        fc3 = Dense(1024, activation='relu')(fc2)
        #drop3 = Dropout(drop_ratio)(fc3)

        lstm1 = LSTM(1024)( fc3)
        #drop4 = Dropout(drop_ratio)(lstm1)

        policy = Dense(self.action_size, activation='softmax')(lstm1)
        value = Dense(1, activation='linear')(lstm1)

        actor = Model(inputs=input, outputs=policy)
        critic = Model(inputs=input, outputs=value)

        # 가치와 정책을 예측하는 함수를 만들어냄
        actor._make_predict_function()
        critic._make_predict_function()

        actor.summary()
        critic.summary()

        return actor, critic



    # 정책신경망을 업데이트하는 함수
    def actor_optimizer(self):
        action = K.placeholder(shape=[None, self.action_size])
        advantages = K.placeholder(shape=[None, ])

        policy = self.actor.output

        # 정책 크로스 엔트로피 오류함수
        action_prob = K.sum(action * policy, axis=1)
        cross_entropy = K.log(action_prob + 1e-10) * advantages
        cross_entropy = -K.sum(cross_entropy)

        # 탐색을 지속적으로 하기 위한 엔트로피 오류
        entropy = K.sum(policy * K.log(policy + 1e-10), axis=1)
        entropy = K.sum(entropy)

        # 두 오류함수를 더해 최종 오류함수를 만듬
        loss = cross_entropy + self.beta * entropy   # beta is 0.05

        optimizer = RMSprop(lr=self.actor_lr, rho=0.99, epsilon=0.01)
        updates = optimizer.get_updates(self.actor.trainable_weights, [],loss)
        train = K.function([self.actor.input, action, advantages],
                           [loss], updates=updates)
        return train

    # 가치신경망을 업데이트하는 함수
    def critic_optimizer(self):
        discounted_prediction = K.placeholder(shape=(None,))

        value = self.critic.output

        # [반환값 - 가치]의 제곱을 오류함수로 함
        loss = K.mean(K.square(discounted_prediction - value))

        optimizer = RMSprop(lr=self.critic_lr, rho=0.99, epsilon=0.01)
        updates = optimizer.get_updates(self.critic.trainable_weights, [],loss)
        train = K.function([self.critic.input, discounted_prediction],
                           [loss], updates=updates)
        return train

    def load_model(self, name):
        self.actor.load_weights(name + "_actor.h5")
        self.critic.load_weights(name + "_critic.h5")

    def save_model(self, name):

        self.actor.save_weights(name + "_actor.h5")
        self.critic.save_weights(name + "_critic.h5")

    # 각 에피소드 당 학습 정보를 기록
    def setup_summary(self):
        episode_total_reward = tf.Variable(0.)
        episode_avg_max_q = tf.Variable(0.)
        episode_duration = tf.Variable(0.)

        tf.summary.scalar('Total Reward/Episode', episode_total_reward)
        tf.summary.scalar('Average Max Prob/Episode', episode_avg_max_q)
        tf.summary.scalar('Duration/Episode', episode_duration)

        summary_vars = [episode_total_reward,
                        episode_avg_max_q,
                        episode_duration]

        summary_placeholders = [tf.placeholder(tf.float32)
                                for _ in range(len(summary_vars))]
        update_ops = [summary_vars[i].assign(summary_placeholders[i])
                      for i in range(len(summary_vars))]
        summary_op = tf.summary.merge_all()
        return summary_placeholders, update_ops, summary_op


# 액터러너 클래스(쓰레드)
class Agent(threading.Thread):
    def __init__(self, action_size, state_size, model, sess,
                 optimizer, discount_factor, summary_ops):
        threading.Thread.__init__(self)

        # A3CAgent 클래스에서 상속
        self.action_size = action_size
        self.state_size = state_size
        self.actor, self.critic = model
        self.sess = sess
        self.optimizer = optimizer
        self.discount_factor = discount_factor
        [self.summary_op, self.summary_placeholders,
         self.update_ops, self.summary_writer] = summary_ops

        # 지정된 타임스텝동안 샘플을 저장할 리스트
        self.states, self.actions, self.rewards = [], [], []

        # 로컬 모델 생성
        self.local_actor, self.local_critic = self.build_local_model()

        self.avg_p_max = 0
        self.avg_loss = 0

        # 모델 업데이트 주기
        self.t_max = 10

        self.t = 0

        self.T_max = 50

        self.step_penalty = 0.001

        self.epoch_size   = 100000   # 20
        self.train_size = 9000
        self. batch_size = 32    #32

        # VFN Preload

        #with open('vfn_rl.pkl', 'rb') as f:
            #self.var_dict = pickle.load(f)





    #  This is the      definition      of      helper      function

    def train_ (self  , TrainPath ):

        global episode
        global global_dtype
        global a3c_graph

        trainfiles = [f for f in listdir(TrainPath) if isfile(join(TrainPath, f))]

        # print(TrainPath)
        # print(trainfiles)

        rand_index = np.random.choice(len(trainfiles), size=self.batch_size)
        print('rand_index = ', rand_index)

        trainfiles_batch = [trainfiles[index] for index in rand_index]

        #trainfiles_batch = ['12775.jpg' , '338809.jpg', '33018.jpg' , '2233.jpg']
        trainfiles_batch_fullname =  [ join( TrainPath, x )   for x in trainfiles_batch     ]

        print( 'trainfiles_batch = ' ,  trainfiles_batch_fullname)
        #print ( ' jo in path  ='  , join(TrainPath, '337580.jpg')  )
        # add the following codes in the main function
        images = [  ]
        for x in trainfiles_batch_fullname:   # remember to replace with the filename of your test image
            #print ( ' train_full_name  = ' , x)

            y = io.imread(  x )
            #print (' y == ' , y.shape , 'y.dim =' ,  y.ndim   )
            if y.ndim != 3 :
                print(' train_full_name  = ', x)
                continue
            images.append (     y[:, :, :3] )
            # io.imread('test1.jpg')[:, :, :3]  # remember to replace with the filename of your test image


        #print(images )

        #sys.exit(0)
        print(len(images))

        #sys.exit(0)

        for j in range(len(images)):

            step = 0
            self.t = 0

            scores, features = evaluate_aesthetics_score([images[j]])
            print('Poorly Cropped Image vs Well Cropped Image')
            print(scores)
            # print(features )

            print(' feature shape ', features[0].shape)

            batch_size = 1
            terminals = np.zeros(batch_size)
            ratios = np.repeat([[0, 0, 20, 20]], batch_size, axis=0)

            if self.t == 0:
                global_score = scores[0]
                global_feature = features[0]

            # print(' global feature shape = ', global_feature )
            print(' global score = ', global_score)

            #parser = argparse.ArgumentParser(description='A2RL: Auto Image Cropping')
            #parser.add_argument('--image_path', required=True, help='Path for the image to be cropped')
            #parser.add_argument('--save_path', required=True, help='Path for saving cropped image')
            #args = parser.parse_args()

            score = 0
            done = False
            # while episode < EPISODES:
            while step < self.T_max and not done:  # T_max = 50

                if self.t == 0:
                    local_feature = global_feature
                    local_score = global_score
                else:
                    local_feature = new_features[0]
                    local_score = new_scores[0]

                observe = np.concatenate((global_feature, local_feature), axis=1)
                print(' observe shape  = ', observe.shape)

                history = np.expand_dims(observe, axis=0)

                print(' history shape  = ', history.shape)
                print(' history   = ', history)
                # print ( ' history = '  , history[ history > 0])

                #if self.t < 3 :  # 5번까지  action은 무조건 Vaild action으로 취급하자
                    #action, policy = self.get_action_pretest(history)
                #else :
                    #action, policy = self.get_action_test(history)
                    #action, policy = self.get_action(history)

                action, policy = self.get_action(history)


                print('action = ', action)
                print('policy = ', policy)


                if action == 13 and  step > 1 :
                    done = True
                    print(' done is True ')
                else  :
                    print (' terminals = ' , terminals )
                    terminals[0] = 0 


                #sys.exit(0)

                    # 선택한 행동으로 한 바운딩 박스 생성

                ratios, terminals = command2action([action], ratios, terminals)

                print(' ratios =', ratios)

                im = images[j].astype(np.float32) / 255 - 0.5
                bbox = generate_bbox([im], ratios)

                #print ( 'bbox == ' , bbox )


                # New Cropped Image 생성 && Resize to ( 227, 227 )
                img = crop_input([im], bbox)

                # Score , Feature of newly Cropped image
                new_scores, new_features = evaluate_aesthetics_score_resized(img)

                print(' new_scores = ', new_scores)
                # print ( ' new_features = ' , new_features )
                # Reward 계산
                reward = np.sign(new_scores[0] - local_score) - self.step_penalty * (self.t + 1)

                # Check Aspect Ratio
                x_width = bbox[0][2] - bbox[0][0]
                y_height = bbox[0][3] - bbox[0][1]




                asratio = x_width/y_height

                #print(' x_width   == ', x_width)
                #print(' y_height  == ', y_height)
                #print(' asratio   ==' , asratio)

                #sys.exit(0)

                if asratio < 0.5 or asratio > 2 :
                    reward = reward - 5
                    
                print('reward = ', reward)

                # sys.exit(0)

                # 각 타임스텝마다 Cropped 이미지 생성

                # print( 'bbox = ' , bbox)
                # 정책의 최대값
                self.avg_p_max += np.amax(self.actor.predict(
                    np.float32(history)))

                score += reward

                # 샘플을 저장
                self.append_sample(history, action, reward)
                print(' {}  ### reward  222 = {} '.format(self.t, reward))

                #if self.t == 1000:
                    #sys.exit(0)

                step += 1
                self.t += 1

                if  step == self.T_max :
                   done = True

                # 에피소드가 끝나거나 최대 타임스텝 수에 도달하면 학습을 진행
                if self.t % self.t_max or done:
                    self.train_model(done)
                    self.update_local_model()

                
                if done :
                    # 각 에피소드 당 학습 정보를 기록
                    episode += 1
                    print("episode:", episode, "  score:", score, "  step:",
                          step)

                    stats = [score, self.avg_p_max / float(step),
                             step]
                    for i in range(len(stats)):
                        self.sess.run(self.update_ops[i], feed_dict={
                            self.summary_placeholders[i]: float(stats[i])
                        })
                    summary_str = self.sess.run(self.summary_op)
                    self.summary_writer.add_summary(summary_str, episode + 1)
                    self.avg_p_max = 0
                    self.avg_loss = 0
                    step = 0

    def train9000_(self, TrainPath ):

        global episode
        global global_dtype
        global a3c_graph

        trainfiles = [f for f in listdir(TrainPath) if isfile(join(TrainPath, f))]

        # print(TrainPath)
        print( len (trainfiles ) )


        episode_step = int(9000/32)


        for i_step  in range( episode_step  ) :

            rand_index = np.random.choice(len(trainfiles), size=self.batch_size)
            print('rand_index = ', rand_index)

            trainfiles_batch = [trainfiles[index] for index in rand_index]
            #print ( range( int ( i_step*32)   , int ( (i_step+1)* 32 )  ))
            #trainfiles_batch = [trainfiles[index] for index in range( int ( i_step*32)   , int ( (i_step+1)* 32 )  )]

            # trainfiles_batch = ['12775.jpg' , '338809.jpg', '33018.jpg' , '2233.jpg']
            trainfiles_batch_fullname = [join(TrainPath, x) for x in trainfiles_batch]

            print('trainfiles_batch = ', trainfiles_batch_fullname)

            # print ( ' jo in path  ='  , join(TrainPath, '337580.jpg')  )
            # add the following codes in the main function
            images = []
            images_filename = [] 
            for x in trainfiles_batch_fullname:  # remember to replace with the filename of your test image
                # print ( ' train_full_name  = ' , x)

                y = io.imread(x)
                # print (' y == ' , y.shape , 'y.dim =' ,  y.ndim   )
                if y.ndim != 3:
                    print(' train_full_name  = ', x)
                    continue
                images.append(y[:, :, :3])
                # io.imread('test1.jpg')[:, :, :3]  # remember to replace with the filename of your test image
                images_filename.append(x)
            # print(images )

            # sys.exit(0)
            print(len(images))

            # sys.exit(0)

            for j in range(len(images)):

                step = 0
                self.t = 0

                scores, features = evaluate_aesthetics_score([images[j]])
                print('Poorly Cropped Image vs Well Cropped Image', images_filename[j])
                print(scores)
                # print(features )

                print(' feature shape ', features[0].shape)

                batch_size = 1
                terminals = np.zeros(batch_size)
                ratios = np.repeat([[0, 0, 20, 20]], batch_size, axis=0)

                if self.t == 0:
                    global_score = scores[0]
                    global_feature = features[0]

                # print(' global feature shape = ', global_feature )
                print(' global score = ', global_score)

                #parser = argparse.ArgumentParser(description='A2RL: Auto Image Cropping')
                #parser.add_argument('--image_path', required=True, help='Path for the image to be cropped')
                #parser.add_argument('--save_path', required=True, help='Path for saving cropped image')
                #args = parser.parse_args()

                score = 0
                done = False
                # while episode < EPISODES:
                while step < self.T_max and not done:  # T_max = 50

                    if self.t == 0:
                        local_feature = global_feature
                        local_score = global_score
                    else:
                        local_feature = new_features[0]
                        local_score = new_scores[0]

                    observe = np.concatenate((global_feature, local_feature), axis=1)
                    print(' observe shape  = ', observe.shape)

                    history = np.expand_dims(observe, axis=0)

                    print(' history shape  = ', history.shape)
                    print(' history   = ', history)
                    # print ( ' history = '  , history[ history > 0])

                    # if self.t < 3 :  # 5번까지  action은 무조건 Vaild action으로 취급하자
                    # action, policy = self.get_action_pretest(history)
                    # else :
                    # action, policy = self.get_action_test(history)
                    # action, policy = self.get_action(history)

                    action, policy = self.get_action(history)

                    print('action = ', action)
                    print('policy = ', policy)

                    if action == 13 and step > 1:
                        done = True
                        print(' done is True ')
                    else:
                        print(' terminals = ', terminals)
                        terminals[0] = 0

                        # sys.exit(0)

                        # 선택한 행동으로 한 바운딩 박스 생성

                    ratios, terminals = command2action([action], ratios, terminals)

                    print(' ratios =', ratios)

                    im = images[j].astype(np.float32) / 255 - 0.5
                    bbox = generate_bbox([im], ratios)

                    # print ( 'bbox == ' , bbox )

                    # New Cropped Image 생성 && Resize to ( 227, 227 )
                    img = crop_input([im], bbox)

                    # Score , Feature of newly Cropped image
                    new_scores, new_features = evaluate_aesthetics_score_resized(img)

                    print(' new_scores = ', new_scores)
                    # print ( ' new_features = ' , new_features )
                    # Reward 계산
                    reward = np.sign(new_scores[0] - local_score) - self.step_penalty * (self.t + 1)

                    # Check Aspect Ratio
                    x_width = bbox[0][2] - bbox[0][0]
                    y_height = bbox[0][3] - bbox[0][1]

                    asratio = x_width / y_height

                    # print(' x_width   == ', x_width)
                    # print(' y_height  == ', y_height)
                    # print(' asratio   ==' , asratio)

                    # sys.exit(0)

                    if asratio < 0.5 or asratio > 2:
                        reward = reward - 5

                    print('reward = ', reward)

                    # sys.exit(0)

                    # 각 타임스텝마다 Cropped 이미지 생성

                    # print( 'bbox = ' , bbox)
                    # 정책의 최대값
                    self.avg_p_max += np.amax(self.actor.predict(
                        np.float32(history)))

                    score += reward

                    # 샘플을 저장
                    self.append_sample(history, action, reward)
                    print(' {}  ### reward  222 = {} '.format(self.t, reward))

                    # if self.t == 1000:
                    # sys.exit(0)

                    step += 1
                    self.t += 1

                    if step == self.T_max:
                        done = True

                    # 에피소드가 끝나거나 최대 타임스텝 수에 도달하면 학습을 진행
                    if self.t % self.t_max or done:
                        self.train_model(done)
                        self.update_local_model()

                    if done:
                        # 각 에피소드 당 학습 정보를 기록
                        episode += 1
                        print("episode:", episode, "  score:", score, "  step:", step ,  "  avg_p_max:" , self.avg_p_max , 
                              "  avg_p_max/step:" , self.avg_p_max / float(step) )

                        stats = [score, self.avg_p_max / float(step),
                                 step]
                        for i in range(len(stats)):
                            self.sess.run(self.update_ops[i], feed_dict={
                                self.summary_placeholders[i]: float(stats[i])
                            })
                        summary_str = self.sess.run(self.summary_op)
                        self.summary_writer.add_summary(summary_str, episode + 1)
                        self.avg_p_max = 0
                        self.avg_loss = 0
                        step = 0

    def run(self):
        global episode
        global global_dtype
        global a3c_graph

        #env = gym.make(env_name)



        # 입력 이미지


        for  epoch_step  in range ( self.epoch_size) :

            #TrainPath = '../AVA/Train'
            #self.train_( TrainPath )
            print( ' epoch_step  == ' , epoch_step ) 
            TrainPath = '../AVA/Train8954'
            self.train9000_( TrainPath )

        #sys.exit(0)

    # k-스텝 prediction 계산
    def discounted_prediction(self, rewards, done):
        discounted_prediction = np.zeros_like(rewards)
        running_add = 0

        if not done:
            running_add = self.critic.predict(np.float32(
                self.states[-1] ))[0]

        for t in reversed(range(0, len(rewards))):
            running_add = running_add * self.discount_factor + rewards[t]
            discounted_prediction[t] = running_add
        return discounted_prediction

    # 정책신경망과 가치신경망을 업데이트
    def train_model(self, done):
        print(' train model start !!! ')
        discounted_prediction = self.discounted_prediction(self.rewards, done)

        states = np.zeros((len(self.states), 1, 2000))
        for i in range(len(self.states)):
            states[i] = self.states[i]

        states = np.float32(states )

        values = self.critic.predict(states)
        values = np.reshape(values, len(values))

        advantages = discounted_prediction - values

        self.optimizer[0]([states, self.actions, advantages])
        self.optimizer[1]([states, discounted_prediction])
        self.states, self.actions, self.rewards = [], [], []

        print( ' train  model end!!! ')

    # 로컬신경망을 생성하는 함수
    def build_local_model(self):

        print(' Child build_local_model  graph', tf.get_default_graph())
        K.set_learning_phase(1)  # set learning phase

        input = Input(shape=self.state_size)

        #input = Input(shape=( 1 , 2000  ))
        fc1 = Dense(1024, activation = 'relu') (input)
        #drop1 = Dropout(drop_ratio)(fc1)

        fc2 = Dense(1024, activation='relu')(fc1)
        #drop2 = Dropout(drop_ratio)(fc2)

        fc3 = Dense(1024, activation='relu')(fc2)
        #drop3 = Dropout(drop_ratio)(fc3)

        lstm1 = LSTM(1024)( fc3)
        #drop4 = Dropout(drop_ratio)(lstm1)

        policy = Dense(self.action_size, activation='softmax')(lstm1)
        value = Dense(1, activation='linear')(lstm1)


        local_actor = Model(inputs=input, outputs=policy)
        local_critic = Model(inputs=input, outputs=value)

        local_actor._make_predict_function()
        local_critic._make_predict_function()

        local_actor.set_weights(self.actor.get_weights())
        local_critic.set_weights(self.critic.get_weights())

        local_actor.summary()
        local_critic.summary()

        return local_actor, local_critic




    # 로컬신경망을 글로벌신경망으로 업데이트
    def update_local_model(self):
        self.local_actor.set_weights(self.actor.get_weights())
        self.local_critic.set_weights(self.critic.get_weights())

    # 정책신경망의 출력을 받아서 확률적으로 행동을 선택


    def get_action_pretest(self, history):
        # 이미 정규화 in evaluate_aesthetics_score
        #history = np.float32(history / 255.)

        action_array = [0, 1, 2, 3, 4, 9, 10 ]

        choice  = np.random.choice( action_array , 1)
        print ( ' choice =' , choice )
        action_index = choice[0]
        print(' pre get_action == ' , action_index )
        policy = self.local_actor.predict(history)[0]

        print(' pre get_action 2 ' , policy)
        #print(' pre get_action 3 ', np.argmax(policy) )
        #action_index = np.argmax(policy)


        return action_index, policy


    def get_action(self, history):
        # 이미 정규화 in evaluate_aesthetics_score
        #history = np.float32(history / 255.)
        print(' get_action')
        policy = self.local_actor.predict(history)[0]

        print(' get_action 2 ')
        action_index = np.random.choice(self.action_size, 1, p=policy)[0]
        return action_index, policy

    # 샘플을 저장
    def append_sample(self, history, action, reward):
        self.states.append(history)
        act = np.zeros(self.action_size)
        act[action] = 1
        self.actions.append(act)
        self.rewards.append(reward)


# 학습속도를 높이기 위해 흑백화면으로 전처리
def pre_processing(next_observe, observe):
    processed_observe = np.maximum(next_observe, observe)
    processed_observe = np.uint8(
        resize(rgb2gray(processed_observe), (84, 84), mode='constant') * 255)
    return processed_observe

if __name__ == "__main__":
    # 입력 이미지

    # In the following example, we are going to feed only one image into the network
    batch_size = 1

    # TODO: Change this if your model file is located somewhere else
    snapshot = '../a2rl_model/model-spp-max'

    tf.reset_default_graph()
    embedding_dim = 1000
    ranking_loss = 'svm'
    net_data = np.load('alexnet.npy', encoding='bytes').item()
    image_placeholder = tf.placeholder(dtype=global_dtype, shape=[batch_size, 227, 227, 3])
    var_dict = nw.get_variable_dict(net_data)
    SPP = True
    pooling = 'max'
    with tf.variable_scope("ranker") as scope:
        feature_vec = nw.build_alexconvnet(image_placeholder, var_dict, embedding_dim, SPP=SPP, pooling=pooling)
        score_func = nw.score(feature_vec)
    # load pre-trained model
    saver = tf.train.Saver(tf.global_variables())
    vfn_sess = tf.Session(config=tf.ConfigProto())
    vfn_sess.run(tf.global_variables_initializer())
    saver.restore(vfn_sess, snapshot)




    global_agent = A3CAgent(action_size=14)
    global_agent.train()


