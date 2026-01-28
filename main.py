# -*- coding: UTF-8 -*-
import numpy as np
import random
import os
import parl
from parl.core.fluid.algorithm import Algorithm
from parl.core.fluid import layers 
import copy
import paddle.fluid as fluid
import collections
import math

MEMORY_SIZE = 20000  
MEMORY_WARMUP_SIZE = 200  
BATCH_SIZE = 32 
LEARNING_RATE = 0.001  
GAMMA = 0.9 

pre_time=0.0
pre_stat_succ=0
pre_stat_coll=0

_n=10 # number of nodes
_simTime=50000 # sec

# Initialize packet time tracking
packet_time_enter = np.zeros(_n)  # Time when packet enters the system for each station
packet_time_exit = np.zeros(_n)  # Time when packet successfully exits the system

rate=1730 # 11, 5.5, 2 or 1 Mbps
_cwmin=16
_cwmax=1024
rtsmode=0 #0: data->ack; 1:rts->cts->data->ack
cwthreshold=16

SIFS=16
DIFS=34
EIFS=SIFS+DIFS+128+112
SLOT=9
M=1000000

cw_list=[]
_pktSize=3895 # bytes
stat_succ=0
stat_coll=0
stat_pkts=np.zeros(_n)
cw=np.zeros(_n)
bo=np.zeros(_n)
fairness_index = 0
now=0.0

f = open("data/50000_simtime/10_nodes/output.txt", "w")
f2 = open("data/50000_simtime/10_nodes/loss.txt", "w")
f3 = open("data/50000_simtime/10_nodes/thr.txt", "w")
f4 = open("data/50000_simtime/10_nodes/latency.txt", "w")

class Model(parl.Model):
    def __init__(self, act_dim):
        hid1_size = 128
        hid2_size = 128
        # 3层全连接网络
        self.fc1 = layers.fc(size=hid1_size, act='relu')
        self.fc2 = layers.fc(size=hid2_size, act='relu')
        self.fc3 = layers.fc(size=act_dim, act=None)

    def value(self, obs):
        h1 = self.fc1(obs)
        h2 = self.fc2(h1)
        Q = self.fc3(h2)
        return Q

class DDQN(Algorithm):
    def __init__(self, model, act_dim=None, gamma=None, lr=None):
        """ Double DQN algorithm
        Args:
            model (parl.Model): model defining forward network of Q function
            act_dim (int): dimension of the action space
            gamma (float): discounted factor for reward computation.
            lr (float): learning rate.
        """
        self.model = model
        self.target_model = copy.deepcopy(model)

        assert isinstance(act_dim, int)
        assert isinstance(gamma, float)

        self.act_dim = act_dim
        self.gamma = gamma
        self.lr = lr

    def predict(self, obs):
        """ use value model self.model to predict the action value
        """
        return self.model.value(obs)

    def learn(self,
              obs,
              action,
              reward,
              next_obs,
              terminal,
              learning_rate=None):
        """ update value model self.model with DQN algorithm
        """
        # Support the modification of learning_rate
        if learning_rate is None:
            assert isinstance(
                self.lr,
                float), "Please set the learning rate of DQN in initializaion."
            learning_rate = self.lr

        pred_value = self.model.value(obs)
        action_onehot = layers.one_hot(action, self.act_dim)
        action_onehot = layers.cast(action_onehot, dtype='float32')
        pred_action_value = layers.reduce_sum(
            layers.elementwise_mul(action_onehot, pred_value), dim=1)

        # calculate the target q value
        next_action_value = self.model.value(next_obs)
        greedy_action = layers.argmax(next_action_value, axis=-1)
        greedy_action = layers.unsqueeze(greedy_action, axes=[1])
        greedy_action_onehot = layers.one_hot(greedy_action, self.act_dim)
        next_pred_value = self.target_model.value(next_obs)
        max_v = layers.reduce_sum(
            greedy_action_onehot * next_pred_value, dim=1)
        max_v.stop_gradient = True

        target = reward + (
            1.0 - layers.cast(terminal, dtype='float32')) * self.gamma * max_v
        cost = layers.square_error_cost(pred_action_value, target)
        cost = layers.reduce_mean(cost)
        optimizer = fluid.optimizer.Adam(
            learning_rate=learning_rate, epsilon=1e-3)
        optimizer.minimize(cost)
        return cost

    def sync_target(self):
        """ sync weights of self.model to self.target_model
        """
        self.model.sync_weights_to(self.target_model)

class Agent(parl.Agent):
    def __init__(self,
                 algorithm,
                 obs_dim,
                 act_dim,
                 e_greed=0.1,
                 e_greed_decrement=0):
        assert isinstance(obs_dim, int)
        assert isinstance(act_dim, int)
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        super(Agent, self).__init__(algorithm)

        self.global_step = 0
        self.update_target_steps = 200  # 每隔200个training steps再把model的参数复制到target_model中

        self.e_greed = e_greed  # 有一定概率随机选取动作，探索
        self.e_greed_decrement = e_greed_decrement  # 随着训练逐步收敛，探索的程度慢慢降低

    def build_program(self):
        self.pred_program = fluid.Program()
        self.learn_program = fluid.Program()

        with fluid.program_guard(self.pred_program):  # 搭建计算图用于 预测动作，定义输入输出变量
            obs = layers.data(
                name='obs', shape=[self.obs_dim], dtype='float32')
            self.value = self.alg.predict(obs)

        with fluid.program_guard(self.learn_program):  # 搭建计算图用于 更新Q网络，定义输入输出变量
            obs = layers.data(
                name='obs', shape=[self.obs_dim], dtype='float32')
            action = layers.data(name='act', shape=[1], dtype='int32')
            reward = layers.data(name='reward', shape=[], dtype='float32')
            next_obs = layers.data(
                name='next_obs', shape=[self.obs_dim], dtype='float32')
            terminal = layers.data(name='terminal', shape=[], dtype='bool')
            self.cost = self.alg.learn(obs, action, reward, next_obs, terminal)

    def sample(self, obs):
        sample = np.random.rand()  # 产生0~1之间的小数
        if sample < self.e_greed:
            act = np.random.randint(self.act_dim)  # 探索：每个动作都有概率被选择
        else:
            act = self.predict(obs)  # 选择最优动作
        self.e_greed = max(
            0.01, self.e_greed - self.e_greed_decrement)  # 随着训练逐步收敛，探索的程度慢慢降低
        return act

    def predict(self, obs):  # 选择最优动作
        obs = np.expand_dims(obs, axis=0)
        pred_Q = self.fluid_executor.run(
            self.pred_program,
            feed={'obs': obs.astype('float32')},
            fetch_list=[self.value])[0]
        pred_Q = np.squeeze(pred_Q, axis=0)
        act = np.argmax(pred_Q)  # 选择Q最大的下标，即对应的动作
        return act

    def learn(self, obs, act, reward, next_obs, terminal):
        # 每隔200个training steps同步一次model和target_model的参数
        if self.global_step % self.update_target_steps == 0:
            self.alg.sync_target()
        self.global_step += 1

        act = np.expand_dims(act, -1)
        feed = {
            'obs': obs.astype('float32'),
            'act': act.astype('int32'),
            'reward': reward,
            'next_obs': next_obs.astype('float32'),
            'terminal': terminal
        }
        cost = self.fluid_executor.run(
            self.learn_program, feed=feed, fetch_list=[self.cost])[0]  # 训练一次网络
        return cost

class ReplayMemory(object):
    def __init__(self, max_size):
        self.buffer = collections.deque(maxlen=max_size)

    def append(self, exp):
        self.buffer.append(exp)

    def sample(self, batch_size):
        mini_batch = random.sample(self.buffer, batch_size)
        obs_batch, action_batch, reward_batch, next_obs_batch, done_batch = [], [], [], [], []

        for experience in mini_batch:
            s, a, r, s_p, done = experience
            obs_batch.append(s)
            action_batch.append(a)
            reward_batch.append(r)
            next_obs_batch.append(s_p)
            done_batch.append(done)

        return np.array(obs_batch).astype('float32'), \
            np.array(action_batch).astype('float32'), np.array(reward_batch).astype('float32'),\
            np.array(next_obs_batch).astype('float32'), np.array(done_batch).astype('float32')

    def __len__(self):
        return len(self.buffer)

def init_bo():
    for i in range(0,_n):
        cw[i]=_cwmin
        bo[i]=random.randint(0,_cwmax)%cw[i]
        #print("cw[",i,"]=",cw[i]," bo[",i,"]=",bo[i])

def Trts():
    time=128+(20*8)/1
    return time

def Tcts():
    time=128+(14*8)/1
    return time

def Tdata():
    global rate
    time=128+((_pktSize+28)*8.0)/rate
    return time

def Tack():
    time=128+(14*8.0)/1
    return time

def getMinBoAllStationsIndex():
    index=0
    min=bo[index]
    for i in range(0,_n):
        if bo[i]<min:
            index=i
            min=bo[index]

    return index

def getCountMinBoAllStations(min):
    count=0
    for i in range(_n):
        if(bo[i]==min):
            count+=1

    return count

def subMinBoFromAll(min,count):
    global _cwmin, _cwmax, cwthreshold, bo_list, cw_times
    equcount=0
    for i in range(0,_n):
        if bo[i]<min:
            print("<Error> min=",min," bo=",bo[i])
            exit(1)
        if(bo[i]>min):
            bo[i]-=min
        elif bo[i]==min: #這邊min指有所節點最小
            #print("before",i,"cw[i]:",cw[i],"bo[i]",bo[i],"cwthreshold",cwthreshold,"cwmin",_cwmin)
            if count==1:
                if (cw[i]<= cwthreshold):
                    cw[i]=cw[i]/2
                else:
                    cw[i]-=16
                if cw[i]<=_cwmin:
                    cw[i]=_cwmin
                bo[i] = random.randint(0, _cwmax) % cw[i]
            elif count>1:
                if (cw[i]<=cwthreshold):
                    cw[i]=cw[i]*2
                else:
                    cw[i]+=16
                if cw[i]>=_cwmax:
                    cw[i]=_cwmax

                if cw[i]<=_cwmin:
                    cw[i]=_cwmin
                bo[i] = random.randint(0, _cwmax) % cw[i]
            else:
                print("<Error> count=",count)
                exit(1)


def setStats(min,index,count):
    global stat_succ,stat_coll
    if count==1:
        stat_pkts[index]+=1
        stat_succ+=1
    else:
        stat_coll+=1
        for i in range(0,_n):
            if bo[i]<min:
                print("<Error> min=", min, " bo=", bo[i])
                exit(1)
            #elif bo[i]==min:
            #    print("Collision with min=", min)

def setNow(min, count, index):
    global M, now, SIFS, DIFS, EIFS, SLOT, packet_time_enter, packet_time_exit

    if rtsmode==1:
        now+=Trts()/M;

    if count==1:
        if(rtsmode==1):
            now+=SIFS/M
            now+=Tcts()/M
            now+=SIFS/M
        now+=DIFS/M
        now+=min*SLOT/M
        packet_time_enter[index] = now  # Packet handling starts
        now+=Tdata()/M
        packet_time_exit[index] = now  # Packet handling ends
        now+=Tack()/M
        # # Calculate latency and write to file
        # latency = (packet_time_exit[index] - packet_time_enter[index]) * 1000  # Convert to milliseconds
        # f4.write(f"{latency}\n")
    elif count>1:
        if rtsmode==1:
            now+=EIFS/M
            now+=min*SLOT/M
        else:
            now+=EIFS/M
            now+=min*SLOT/M
            now+=Tdata()/M
    else:
          print("<Error> count=", count)
          exit(1)

def new_resolve(new_cwmin,new_cwthreshold):
    global cwthreshold,_cwmin,_cwmax

    cwthreshold=new_cwthreshold
    _cwmin=new_cwmin
    #_cwmax=new_cwmin 
    #print("cwmin",_cwmin,"cwmax",_cwmax)
    # index=getMinBoAllStationsIndex()
    index = getMinBoAllStationsIndex()  # Retrieve index of the station
    #print("index:",index)
    min=bo[index]
    #print("min:",min)
    count=getCountMinBoAllStations(min)
    #print(count)
    setNow(min, count, index)
    setStats(min, index, count)
    subMinBoFromAll(min, count)

def printStats():
    global _n, fairness_index, _pktSize, stat_pkts, now
    print("\nGeneral Statistics\n")
    print("Stations: ",_n)
    print("-"*50)
    print("stat_succ: ",stat_succ,"stat_coll:",stat_coll)
    print(_n, stat_coll/(stat_succ+stat_coll)*100, ((stat_succ)*(_pktSize*8.0)/now)/100000000)
    print("Collision rate: ", stat_coll/(stat_succ+stat_coll)*100, "%")
    print("Aggregate Throughput: ", ((stat_succ)*(_pktSize*8.0)/now) / 100000000)

    sum_thr=0
    sum_square_thr=0
    for i in range(0,_n):
        sum_thr=sum_thr+stat_pkts[i]*(_pktSize*8.0)/now
        sum_square_thr=sum_square_thr+stat_pkts[i]*(_pktSize*8.0)/now*stat_pkts[i]*(_pktSize*8.0)/now
    fair_index=sum_thr*sum_thr/(_n*sum_square_thr)
    print("Fairness Index: ", fair_index)


def printLatency():
    global packet_time_enter, packet_time_exit, _n
    latencies = packet_time_exit - packet_time_enter
    avg_latency = np.mean(latencies[latencies > 0])  # Average only non-zero latencies
    print("Average Latency: {:.5f} ms".format(avg_latency * 1000))  # Convert to milliseconds


def main():
    global _n, now, _simTime, stat_succ, stat_coll, pre_stat_succ, pre_stat_coll, _pktSize, pre_time,new_cwthreshold,stat_pkts
    pre_collision_rate=0.0
    random.seed(1)
    np.random.seed(1)
    init_bo()
    obs_dim=2
    act_dim=9
    print("obs_dim=",obs_dim,"act_dim=",act_dim)

    # 
    model = Model(act_dim=act_dim)
    algorithm = DDQN(model, act_dim=act_dim, gamma=GAMMA, lr=LEARNING_RATE)
    agent = Agent(
        algorithm,
        obs_dim=obs_dim,
        act_dim=act_dim,
        e_greed=0.1,  # 
        e_greed_decrement=1e-6)  #

    rpm = ReplayMemory(MEMORY_SIZE)  # 

    #save_path = './ddnq_model.ckpt'
    #if os.path.isfile(save_path):
    #    agent.restore(save_path)

    step=0
    reward=0.0
    state = [0.0, 0.0]
    show=0
    episode=0
    new_cwmin=16
    new_cwthreshold=256

    while now < _simTime:
        while len(rpm) < MEMORY_WARMUP_SIZE:
            obs = np.array(state)
            action = agent.sample(obs)
            act_cwmin=action//3
            act_cwthreshold=action%3
            #print(new_cwmin,new_cwthreshold)                        
            #print("act_cwmin",act_cwmin,"act_cwthreshold",act_cwthreshold)
            if act_cwmin==0:
                pre_cwmin=new_cwmin
            elif act_cwmin==1:
                if new_cwmin<1024:
                    pre_cwmin=new_cwmin
                    new_cwmin+=1
                else:
                    pre_cwmin=new_cwmin                    
            elif act_cwmin==2:
                if new_cwmin>16:
                    pre_cwmin=new_cwmin
                    new_cwmin-=1
                else:
                    pre_cwmin=new_cwmin


            if act_cwthreshold==0:
                pre_cwthrashold=new_cwthreshold
            elif act_cwthreshold==1:
                if new_cwthreshold<1024:
                    pre_cwthrashold=new_cwthreshold
                    new_cwthreshold+=1
                else:
                    pre_cwthrashold=new_cwthreshold                  
            elif act_cwthreshold==2:
                if new_cwthreshold>16:
                    pre_cwthrashold=new_cwthreshold
                    new_cwthreshold-=1
                else:
                    pre_cwthrashold=new_cwthreshold           
            #print(new_cwmin,new_cwthreshold)
            #print()  

            t1 = now
            while True:
                new_resolve(new_cwmin,new_cwthreshold)
                if now - t1 > 0.1:
                    break

            collision_rate = (stat_coll - pre_stat_coll) / (stat_succ + stat_coll - pre_stat_succ - pre_stat_coll) * 100
            thr = (stat_succ - pre_stat_succ) * (_pktSize * 8.0) / (now - pre_time)
            reward = thr / rate / M
            # if collision_rate < 10 and reward <0.45:
            #     reward=0
            # print(episode,"\t", reward, file = f)
            next_state=[]
            next_state.append(collision_rate)
            next_state.append(pre_collision_rate)
            # step += 1
            pre_stat_succ = stat_succ
            pre_stat_coll = stat_coll
            pre_collision_rate = collision_rate
            pre_time = now
            #print("now=", now, " collison rate=",collision_rate," thr=", thr)
            next_obs=np.array(next_state)
            done = False
            rpm.append((obs, action, reward, next_obs, done))
            #print("len(rpm)=", len(rpm), "obs=", obs, " action=", action, " next_obs=", next_obs, " reward=", reward)
            state = next_state
            #if step>=5:
            #    exit()

        if step%5==0:
          (batch_obs, batch_action, batch_reward, batch_next_obs, batch_done) = rpm.sample(BATCH_SIZE)
          train_loss = agent.learn(batch_obs, batch_action, batch_reward, batch_next_obs, batch_done)
        #   print(episode,"\t", train_loss[0], file = f2)
        #   print(episode,train_loss,"\n",file=f)

        obs = np.array(state)
        action = agent.sample(obs)
        act_cwmin=action//3
        act_cwthreshold=action%3
        #print(new_cwmin,new_cwthreshold)                        
        #print("act_cwmin",act_cwmin,"act_cwthreshold",act_cwthreshold)
        if act_cwmin==0:
            pre_cwmin=new_cwmin
        elif act_cwmin==1:
            if new_cwmin<1024:
                pre_cwmin=new_cwmin
                new_cwmin+=1
            else:
                pre_cwmin=new_cwmin                    
        elif act_cwmin==2:
            if new_cwmin>16:
                pre_cwmin=new_cwmin
                new_cwmin-=1
            else:
                pre_cwmin=new_cwmin

        if act_cwthreshold==0:
            pre_cwthrashold=new_cwthreshold
        elif act_cwthreshold==1:
            if new_cwthreshold<1024:
                pre_cwthrashold=new_cwthreshold
                new_cwthreshold+=1
            else:
                pre_cwthrashold=new_cwthreshold                  
        elif act_cwthreshold==2:
            if new_cwthreshold>16:
                pre_cwthrashold=new_cwthreshold
                new_cwthreshold-=1
            else:
                pre_cwthrashold=new_cwthreshold           
        #print(new_cwmin,new_cwthreshold)
        #print()

        t1 = now
        episode+=1
        while True:
            new_resolve(new_cwmin,new_cwthreshold)
            if now - t1 > 0.1:
                break

        collision_rate = (stat_coll - pre_stat_coll) / (stat_succ + stat_coll - pre_stat_succ - pre_stat_coll) * 100
        thr = (stat_succ - pre_stat_succ) * (_pktSize * 8.0) / (now - pre_time)
        reward = thr / rate / M
        # if collision_rate < 10 and reward <0.45:
        #     reward=0
        # print(episode,"\t", reward, file = f)
        next_state = []
        next_state.append(collision_rate)
        next_state.append(pre_collision_rate)
        step += 1
        pre_stat_succ = stat_succ
        pre_stat_coll = stat_coll
        pre_collision_rate = collision_rate
        pre_time = now
        # print("now=", now, " collison rate=",collision_rate," thr=", thr)
        next_obs = np.array(next_state)
        done = False
        rpm.append((obs, action, reward, next_obs, done))
        if now > show:
            print("now=", now, "obs=", obs, " action=", action, " next_obs=", next_obs, " reward=", reward)
            print(now, "\t", reward, file = f)
            print(now, "\t", train_loss[0], file = f2)
            print(now, "\t", thr, file = f3)
            # Calculate average latency since last log
            valid_latencies = [packet_time_exit[i] - packet_time_enter[i] for i in range(_n) if packet_time_exit[i] > packet_time_enter[i]]
            if valid_latencies:
                average_latency = sum(valid_latencies) / len(valid_latencies)
                print(now, "\t", average_latency * 1000, file=f4)  # Convert to milliseconds and print
            show+=500
        state = next_state

    printStats()
    # train_stat_pkts=stat_pkts
    # print(stat_pkts)
    # agent.save(save_path)
    f.close()

    now=pre_time=0.0
    state = [0.0, 0.0]
    stat_coll=pre_stat_coll=0
    stat_succ=pre_stat_succ=0
    stat_pkts=np.zeros(_n)
    while now < 5:
        obs = np.array(state)
        action = agent.predict(obs)
        act_cwmin=action//3
        act_cwthreshold=action%3
        #print(new_cwmin,new_cwthreshold)                        
        #print("act_cwmin",act_cwmin,"act_cwthreshold",act_cwthreshold)
        if act_cwmin==0:
            pre_cwmin=new_cwmin
        elif act_cwmin==1:
            if new_cwmin<1024:
                pre_cwmin=new_cwmin
                new_cwmin+=1
            else:
                pre_cwmin=new_cwmin                    
        elif act_cwmin==2:
            if new_cwmin>16:
                pre_cwmin=new_cwmin
                new_cwmin-=1
            else:
                pre_cwmin=new_cwmin

        if act_cwthreshold==0:
            pre_cwthrashold=new_cwthreshold
        elif act_cwthreshold==1:
            if new_cwthreshold<1024:
                pre_cwthrashold=new_cwthreshold
                new_cwthreshold+=1
            else:
                pre_cwthrashold=new_cwthreshold                  
        elif act_cwthreshold==2:
            if new_cwthreshold>16:
                pre_cwthrashold=new_cwthreshold
                new_cwthreshold-=1
            else:
                pre_cwthrashold=new_cwthreshold           
        print("new_cwmin: ", new_cwmin, "new_cwthreshold: ", new_cwthreshold)
        #print()  
        # cw_list.append(new_cwmin)    

        t1 = now
        while True:
            new_resolve(new_cwmin,new_cwthreshold)
            if now - t1 > 0.1:
                break

        collision_rate = (stat_coll - pre_stat_coll) / (stat_succ + stat_coll - pre_stat_succ - pre_stat_coll) * 100
        thr = (stat_succ - pre_stat_succ) * (_pktSize * 8.0) / (now - pre_time)
        next_state = []
        next_state.append(collision_rate)
        next_state.append(pre_collision_rate)
        pre_stat_succ = stat_succ
        pre_stat_coll = stat_coll
        pre_collision_rate = collision_rate
        pre_time = now
        print("now=", now, " collison rate=",collision_rate," thr=", thr)
        state = next_state
    print("="*25, " Evaluation Result:")
    printStats()

main()
printLatency()
f.close()
f2.close()
f3.close()
f4.close()