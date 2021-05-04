import DQN as DQNAgent
import gym
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import copy

import matplotlib.pyplot as plt

envName='CartPole-v1'
env=gym.make(envName)

torch.set_num_threads(4)

class DeepQNetwork(nn.Module):
    def __init__(self, lr, n_actions, input_dims, name):
        super(DeepQNetwork, self).__init__()
        self.checkpoint_file = os.path.join("./", name)

        self.fc1 = nn.Linear(*input_dims, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, n_actions, bias=False)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)

        self.loss = nn.MSELoss()
        self.device = 'cpu'
        self.to(self.device)

    def forward(self, state):
        fc1 = F.relu(self.fc1(state))
        fc2 = F.relu(self.fc2(fc1))
        actions = self.fc3(fc2)

        return actions

    def save_checkpoint(self):
        pass

    def load_checkpoint(self):
        pass


class BaseLine(nn.Module):
    def __init__(self, lr, action_dims, input_dims, name):
        super(BaseLine, self).__init__()
        self.checkpoint_file = os.path.join("./", name)

        l = list(input_dims)
        l[0] += 1
        input_dims = tuple(l)
        self.fc1 = nn.Linear(*input_dims, 20)
        self.fc2 = nn.Linear(20, 1)

        self.optimizer = optim.RMSprop(self.parameters(), lr=lr)

        self.loss = nn.MSELoss()
        self.device = 'cpu'
        self.to(self.device)

    def forward(self, state, action):
        fc1 = F.tanh(self.fc1(torch.cat((state, torch.unsqueeze(action.float(), dim=-1)), dim=1)))
        fc2 = self.fc2(fc1)

        return fc2

    def save_checkpoint(self):
        pass

    def load_checkpoint(self):
        pass



nEpochs=1000
lr=1e-3
nRuns =10
nameAppendix=str(np.random.randint(10050000))

logsDQN=np.zeros([20,nEpochs])
 
for runId in np.arange(nRuns):
    print("RUN ",runId)

    mem_size = 1000000
    Q1 = DeepQNetwork(lr, env.action_space.n, (env.observation_space.shape), "q_eval")
    Q2 = DeepQNetwork(lr, env.action_space.n, (env.observation_space.shape), "q_next")
    BL = BaseLine(lr, (env.action_space.shape), (env.observation_space.shape), "baseline")
    Q2.load_state_dict(Q1.state_dict())
    # time0=time.time() 
    agent1 = DQNAgent.DQN_VR(Q1, Q2, BL, env, gamma=0.95, epsilon=0.99, lr=lr, input_dims=(env.observation_space.shape),
                     n_actions=env.action_space.n, mem_size=mem_size, eps_min=0.01,
                     batch_size=32, replace=50, eps_dec=0.998, save_var = True, save_norm = True)
    Q1 = DeepQNetwork(lr, env.action_space.n, (env.observation_space.shape), "q_eval")
    Q2 = DeepQNetwork(lr, env.action_space.n, (env.observation_space.shape), "q_next")
    agent2 = DQNAgent.DQN(Q1, Q2, env, gamma=0.95, epsilon=0.99, lr=lr, input_dims=(env.observation_space.shape),
                     n_actions=env.action_space.n, mem_size=mem_size, eps_min=0.01,
                     batch_size=32, replace=50, eps_dec=0.998, save_var = True)

    logs=agent1.train(n_epochs=nEpochs,verbose=20)
    # timesDQN[runId]=time.time()-time0
    logsDQN[runId,:]=logs['meanRewards']
    with open('.\DQNNNNNN_VR'+envName+'TestStab_' + 'DQN_VR' +'_nEpochs'+str(nEpochs)+'_lr'+str(lr)+'_'+nameAppendix+'_'+str(runId)+'.pkl','wb') as f:
        pickle.dump({'meanRewards':logs['meanRewards'], 'variances':logs['variances'], 'qv':logs['Qvalues'], 'norms':logs['norms']},f)


    logs=agent2.train(n_epochs=nEpochs,verbose=20)
    # timesDQN[runId]=time.time()-time0
    logsDQN[10+runId,:]=logs['meanRewards']
    with open('.\DQNNNNNN'+envName+'TestStab_' + 'DQN_VR' +'_nEpochs'+str(nEpochs)+'_lr'+str(lr)+'_'+nameAppendix+'_'+str(runId)+'.pkl','wb') as f:
        pickle.dump({'meanRewards':logs['meanRewards'], 'variances':logs['variances'], 'qv':logs['Qvalues']},f)

f,ax = plt.subplots(figsize=(10,6))

ax.grid()
ax.set_title("Rewards During Training",fontsize=16)
ax.set_xlabel("Epochs",fontsize=14)
ax.set_ylabel("Mean Reward",fontsize=14)
ax.tick_params(axis='x', labelsize=14)
ax.tick_params(axis='y', labelsize=14)
ax.plot(np.arange(nEpochs),np.mean(logsDQN[0:10,:], axis=0))
ax.plot(np.arange(nEpochs),np.mean(logsDQN[10:20,:], axis=0))
ax.legend(['DQN VR', 'DQN'])

plt.savefig('Cart Pole rewards.pdf',bbox_inches='tight')

plt.show()


