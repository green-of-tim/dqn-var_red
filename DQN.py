#!/usr/bin/env python
# coding: utf-8

import torch
import numpy as np
import tqdm

import copy


class Error(Exception):
    pass

class AgentConfigError(Error):
    """
        Exception raised when agent config is invalid
    """

    def __init__(self, expression, message):
        self.expression = expression
        self.message = message


class DQN():

    def __init__(self, DeepQNetwork1, DeepQNetwork2, env, mem_size, gamma, epsilon, lr, n_actions, input_dims, 
        batch_size, eps_min, eps_dec, replace, device='cpu', save_var=False):
        """
            DeepQNetwork1, DeepQNetwork2 -- identical convolutional nets for Q-function
            simulator Simulator -- simulator object
            mem_size -- max size of memory buffer for experience replay
            gamma -- discount factor
            epsilon -- starting exploration rate
            ls -- learning rate
            n_actions -- number of actions in the environment
            input_dims -- dimensions of the imput
            batch_size -- size of batch for learning
            eps_min -- minimum value of exploration rate
            eps_dec -- explorations rate decay parameter
            replace -- number of games before updating target network
            save_var -- save mean variances for each game
            save_norm -- save norms of expected value of baseline
        """
        #self.simulator = simulator
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.batch_size = batch_size
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.replace_target_cnt = replace
        self.action_space = [i for i in range(n_actions)]
        self.learn_step_counter = 0
        self.save_var = save_var
        

        if self.save_var:
            self.var_tmp = []
            self.variances = []

        if self.save_var:
            self.Qvalues = []
            self.test_states = torch.tensor([[-0.061586, -0.75893141, 0.05793238, 1.15547541], #states for CartPole
                                [-0.07676463, -0.95475889, 0.08104189, 1.46574644],
                                [-0.0958598, -1.15077434, 0.11035682, 1.78260485],
                                [-0.11887529, -0.95705275, 0.14600892, 1.5261692 ],
                                [-0.13801635, -0.7639636, 0.1765323, 1.28239155],
                                [-0.15329562, -0.57147373, 0.20218013, 1.04977545],
                                [-0.02786724, 0.00361763, -0.03938967, -0.01611184],
                                [-0.02779488, -0.19091794, -0.03971191, 0.26388759],
                                [-0.03161324, 0.00474768, -0.03443415, -0.04105167]])
            # self.test_states = torch.tensor([[0.3, 0.5, 0.4, -0.4, -0.3], # states for Snake
            #                                 [0.4, 0.6, 0.3, 0.3, -0.3],
            #                                 [0.5, 0., 0.4, 0.3, 0.2],
            #                                 [0.4, -0.1, 0.5, -0.2, 0.4]])

        class ReplayBuffer(object):
            def __init__(self, max_size, input_shape, n_actions):
                self.mem_size = max_size
                self.mem_cntr = 0
                self.state_memory = np.zeros((self.mem_size, *input_shape),
                                            dtype=np.float32)
                self.new_state_memory = np.zeros((self.mem_size, *input_shape),
                                                dtype=np.float32)

                self.action_memory = np.zeros(self.mem_size, dtype=np.int64)
                self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
                self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

            def store_transition(self, state, action, reward, state_, done):
                index = self.mem_cntr % self.mem_size
                self.state_memory[index] = state
                self.new_state_memory[index] = state_
                self.action_memory[index] = action
                self.reward_memory[index] = reward
                self.terminal_memory[index] = done
                self.mem_cntr += 1

            def sample_buffer(self, batch_size):
                max_mem = min(self.mem_cntr, self.mem_size)
                batch = np.random.choice(max_mem, batch_size, replace=False)

                states = self.state_memory[batch]
                actions = self.action_memory[batch]
                rewards = self.reward_memory[batch]
                states_ = self.new_state_memory[batch]
                terminal = self.terminal_memory[batch]

                return states, actions, rewards, states_, terminal

        if len(input_dims) != 1:
            self.type = "pic"
            self.memory = ReplayBuffer(mem_size, input_dims[-1:] + input_dims[0:-1], n_actions)
        else:
            self.type = "vec"
            self.memory = ReplayBuffer(mem_size, input_dims, n_actions)

        self.q_eval = DeepQNetwork1

        self.q_next = DeepQNetwork2

        
    def SampleActionDiscrete(self, state):
        """
            Samples actions based on the given state

            state float [batch,stateShape] -- batch of states
                  or float [stateShape] -- one state
        """
        if len(state.shape) == len(self.input_dims):
            # state = torch.tensor([state],dtype=torch.float).to(self.q_eval.device)
            state = torch.unsqueeze(torch.Tensor(state),0)
            actions = self.q_eval.forward(state.to(self.q_eval.device))
            action = torch.argmax(actions, dim=-1).item()
            return action
        else:
            actions = self.q_eval.forward(state.to(self.q_eval.device))
            action = torch.argmax(actions, dim=-1).detach().numpy()
            return action

    def store_transition(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)

    def sample_memory(self):
        state, action, reward, new_state, done = \
                                self.memory.sample_buffer(self.batch_size)

        states = torch.tensor(state).to(self.q_eval.device)
        rewards = torch.tensor(reward).to(self.q_eval.device)
        dones = torch.tensor(done).to(self.q_eval.device)
        actions = torch.tensor(action).to(self.q_eval.device)
        states_ = torch.tensor(new_state).to(self.q_eval.device)

        return states, actions, rewards, states_, dones

    def replace_target_network(self):
        if self.learn_step_counter % self.replace_target_cnt == 0:
            self.q_next.load_state_dict(self.q_eval.state_dict())

    # def decrement_epsilon(self):
    #     self.epsilon = self.epsilon - self.eps_dec \
    #                        if self.epsilon > self.eps_min else self.eps_min

    def decrement_epsilon(self):
        self.epsilon = self.epsilon * self.eps_dec \
                           if self.epsilon > self.eps_min else self.eps_min

    def save_models(self):
        self.q_eval.save_checkpoint()
        self.q_next.save_checkpoint()

    def load_models(self):
        self.q_eval.load_checkpoint()
        self.q_next.load_checkpoint()

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        self.q_eval.optimizer.zero_grad()

        self.replace_target_network()

        states, actions, rewards, states_, dones = self.sample_memory()
        indices = np.arange(self.batch_size)

        q_pred = self.q_eval.forward(states)[indices, actions]
        q_next = self.q_next.forward(states_).max(dim=1)[0]

        q_next[dones] = 0.0
        q_target = rewards + self.gamma*q_next

        if self.save_var:
            autogradGrads=[torch.autograd.grad(q_pred[k],self.q_eval.parameters(), retain_graph=True, create_graph=True)\
                        for k in np.arange(self.batch_size)]
                
            q_Gradients = torch.cat([ \
                                    torch.unsqueeze(torch.cat([ 
                                                torch.flatten(grad) for grad in\
                                                    autogradGrads[k]\
                                    ],0),0)\
                                for k in np.arange(self.batch_size)],0).detach().numpy()

            weights = torch.from_numpy(np.array([np.linalg.norm(q_Gradients[k]) for k in np.arange(self.batch_size)]))
        
            vector_tmp = (q_target.detach() - q_pred.detach()) * weights
            firstTerm = torch.mean(torch.sum(vector_tmp*vector_tmp,-1))
            secondTerm = torch.sum(torch.mean(vector_tmp,0)**2)
            self.var_tmp.append(firstTerm - secondTerm)

        loss = self.q_eval.loss(q_target, q_pred).to(self.q_eval.device)
        loss.backward()
        self.q_eval.optimizer.step()
        self.learn_step_counter += 1

        self.decrement_epsilon()

    
    def train(self, n_epochs=2000, max_step=1000, load_checkpoint=False, verbose=0):   
        """
            Trains the agent
            n_epochs int -- number of epochs
            max_step int -- maximum length of sampled trajectory from the simulator
            load_checkpoint -- whether use old or new networks
            verbose int -- verbosity parameter, set this positive and print meanRewards every *verbose* epochs
        """

        stats = {'meanRewards': np.zeros([n_epochs]), 'nSteps': np.zeros([n_epochs])}

        #policy handler for the simulator
        def policyHandler(state):      
            if np.random.random() > self.epsilon:
                if self.type == "pic":
                    action = np.array(self.SampleActionDiscrete(np.transpose(state, (2, 0, 1))))
                else:
                    action = np.array(self.SampleActionDiscrete(state))
            else:
                action = np.random.choice(self.action_space)
            return action
            

        print("Training DQN.....")

        best_score = -np.inf
        n_steps = 0

        if load_checkpoint:
            agent.load_models()

        # fname = 'DQN_' + '_lr' + str(self.lr) +'_' + str(n_epochs) + 'games'
        # figure_file = 'plots/' + fname + '.png'

        for epochId in tqdm.tqdm(np.arange(n_epochs)):

            if self.save_var:
                self.var_tmp = []

            stat_steps = 0
            rewards = 0
            s = self.env.reset()
            while True:
                # env.render()
                a = policyHandler(s)
                s_, r, terminal, _ = self.env.step(a)
                d = 0
                rewards += r
                stat_steps += 1
                if terminal:
                    d = 1
                self.store_transition(s, a, r, s_, d)
                self.learn()
                s = s_
                if terminal:
                    break

            if self.save_var:
                mean_var = torch.mean(torch.tensor(self.var_tmp))
                if np.isnan(mean_var):
                    mean_var = torch.mean(torch.tensor([0.]))
                self.variances.append(mean_var)

            if self.save_var:
                qv = self.q_eval.forward(self.test_states).max(dim=1)[0]
                self.Qvalues.append(torch.mean(qv))

            avg_score = np.mean(stats['meanRewards'][-100:])

            if avg_score > best_score:
                if not load_checkpoint:
                    self.save_models()
                best_score = avg_score

            if load_checkpoint and n_steps >= 18000:
                break
            
            stats['meanRewards'][epochId]=rewards
            stats['nSteps'][epochId]=stat_steps

            if(verbose>0):
                if(epochId % verbose==0):
                    print("......","meanReward:",stats['meanRewards'][epochId])

        print("DONE")

        if self.save_weights:
            stats['weights'] = self.weights
        if self.save_var:
            stats['variances'] = self.variances
            stats['Qvalues'] = self.Qvalues

        return stats
    
    def evaluate(self, n_samples=2000, max_step=1000):
        """
            Evaluates the agent by sampling from the environment
            n_samples int -- number of trajectories for estimation
            max_step int -- maximum length of trajectory allowed in simulator
        """
        
        stats = {'rewardMean': 0,'rewardStd': 0}

        #policy handler for the simulator
        def policyHandler(state):
            if self.type == "pic":      
                action = np.array(self.SampleActionDiscrete(np.transpose(state, (2, 0, 1))))
            else:
                action = np.array(self.SampleActionDiscrete(state))
            return action


        stateSampler= pySim.GymResetSampler(self.simulator.gymInstance)
        if self.type == 'pic':
            rewards = \
                    self.simulator.SampleTrajectoriesFromStateSampler( stateSampler, policyHandler,n_samples,\
                        returnRewards=True, maxIterations=max_step, stateMemorySize=self.frames_in_state, grayscale=self.grayscale, downsample=self.downsample)
        else:
            rewards = \
                    self.simulator.SampleTrajectoriesFromStateSampler( stateSampler, policyHandler,n_samples,\
                        returnRewards=True, maxIterations=max_step)
        stats['rewardMean']=np.mean(np.sum(rewards[:,0,:],axis=1))
        stats['rewardStd']=np.std(np.sum(rewards[:,0,:],axis=1))
            
        return stats

##################################################################

class DQN_VR():

    def __init__(self, DeepQNetwork1, DeepQNetwork2, BaseLineNetwork, env, mem_size, gamma, epsilon, lr, n_actions, input_dims, 
        batch_size, eps_min, eps_dec, replace, device='cpu', save_var=False, save_norm=False):
        """
            DeepQNetwork1, DeepQNetwork2 -- identical convolutional nets for Q-function
            simulator Simulator -- simulator object
            mem_size -- max size of memory buffer for experience replay
            gamma -- discount factor
            epsilon -- starting exploration rate
            ls -- learning rate
            n_actions -- number of actions in the environment
            input_dims -- dimensions of the imput
            batch_size -- size of batch for learning
            eps_min -- minimum value of exploration rate
            eps_dec -- explorations rate decay parameter
            replace -- number of games before updating target network
            save_var -- save mean variances for each game
            save_norm -- save norms of expected value of baseline
        """
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.batch_size = batch_size
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.replace_target_cnt = replace
        self.action_space = [i for i in range(n_actions)]
        self.learn_step_counter = 0
        self.VR_counter = 0

        self.save_var = save_var
        self.save_norm = save_norm

        if self.save_var:
            self.var_tmp = []
            self.variances = []

        if self.save_norm:
            self.norm_tmp = []
            self.norms = []
        
        if self.save_var:
            self.Qvalues = []
            # self.test_states = torch.tensor([[0.3, 0.5, 0.4, -0.4, -0.3], # states for Snake
            #                                 [0.4, 0.6, 0.3, 0.3, -0.3],
            #                                 [0.5, 0., 0.4, 0.3, 0.2],
            #                                 [0.4, -0.1, 0.5, -0.2, 0.4]])
            self.test_states = torch.tensor([[-0.061586, -0.75893141, 0.05793238, 1.15547541], #states for CartPole
                                [-0.07676463, -0.95475889, 0.08104189, 1.46574644],
                                [-0.0958598, -1.15077434, 0.11035682, 1.78260485],
                                [-0.11887529, -0.95705275, 0.14600892, 1.5261692 ],
                                [-0.13801635, -0.7639636, 0.1765323, 1.28239155],
                                [-0.15329562, -0.57147373, 0.20218013, 1.04977545],
                                [-0.02786724, 0.00361763, -0.03938967, -0.01611184],
                                [-0.02779488, -0.19091794, -0.03971191, 0.26388759],
                                [-0.03161324, 0.00474768, -0.03443415, -0.04105167]])

        class ReplayBuffer(object):
            def __init__(self, max_size, input_shape, n_actions):
                self.mem_size = max_size
                self.mem_cntr = 0
                self.state_memory = np.zeros((self.mem_size, *input_shape),
                                            dtype=np.float32)
                self.new_state_memory = np.zeros((self.mem_size, *input_shape),
                                                dtype=np.float32)

                self.action_memory = np.zeros(self.mem_size, dtype=np.int64)
                self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
                self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

            def store_transition(self, state, action, reward, state_, done):
                index = self.mem_cntr % self.mem_size
                self.state_memory[index] = state
                self.new_state_memory[index] = state_
                self.action_memory[index] = action
                self.reward_memory[index] = reward
                self.terminal_memory[index] = done
                self.mem_cntr += 1

            def sample_buffer(self, batch_size):
                max_mem = min(self.mem_cntr, self.mem_size)
                batch = np.random.choice(max_mem, batch_size, replace=False)

                states = self.state_memory[batch]
                actions = self.action_memory[batch]
                rewards = self.reward_memory[batch]
                states_ = self.new_state_memory[batch]
                terminal = self.terminal_memory[batch]

                return states, actions, rewards, states_, terminal

        if len(input_dims) != 1:
            self.type = "pic"
            self.memory = ReplayBuffer(mem_size, input_dims[-1:] + input_dims[0:-1], n_actions)
        else:
            self.type = "vec"
            self.memory = ReplayBuffer(mem_size, input_dims, n_actions)

        self.q_eval = DeepQNetwork1

        self.q_next = DeepQNetwork2

        self.baseline = BaseLineNetwork

        
    def SampleActionDiscrete(self, state):
        """
            Samples actions based on the given state

            state float [batch,stateShape] -- batch of states
                  or float [stateShape] -- one state
        """
        if len(state.shape) == len(self.input_dims):
            # state = torch.tensor([state],dtype=torch.float).to(self.q_eval.device)
            state = torch.unsqueeze(torch.Tensor(state),0)
            actions = self.q_eval.forward(state.to(self.q_eval.device))
            action = torch.argmax(actions, dim=-1).item()
            return action
        else:
            actions = self.q_eval.forward(state.to(self.q_eval.device))
            action = torch.argmax(actions, dim=-1).detach().numpy()
            return action

    def store_transition(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)

    def sample_memory(self):
        state, action, reward, new_state, done = \
                                self.memory.sample_buffer(self.batch_size)

        states = torch.tensor(state).to(self.q_eval.device)
        rewards = torch.tensor(reward).to(self.q_eval.device)
        dones = torch.tensor(done).to(self.q_eval.device)
        actions = torch.tensor(action).to(self.q_eval.device)
        states_ = torch.tensor(new_state).to(self.q_eval.device)

        return states, actions, rewards, states_, dones

    def replace_target_network(self):
        if self.learn_step_counter % self.replace_target_cnt == 0:
            self.q_next.load_state_dict(self.q_eval.state_dict())

    # def decrement_epsilon(self):
    #     self.epsilon = self.epsilon - self.eps_dec \
    #                        if self.epsilon > self.eps_min else self.eps_min

    def decrement_epsilon(self):
        self.epsilon = self.epsilon * self.eps_dec \
                           if self.epsilon > self.eps_min else self.eps_min

    def save_models(self):
        self.q_eval.save_checkpoint()
        self.q_next.save_checkpoint()

    def load_models(self):
        self.q_eval.load_checkpoint()
        self.q_next.load_checkpoint()


    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        # if self.VR_counter % 4 != 0:
        #     self.VR_counter = (self.VR_counter + 1) % 4
        #     return
        # self.VR_counter = (self.VR_counter + 1) % 4

        self.q_eval.optimizer.zero_grad()
        self.baseline.optimizer.zero_grad()

        self.replace_target_network()

        states, actions, rewards, states_, dones = self.sample_memory()
        indices = np.arange(self.batch_size)

        q_pred = self.q_eval.forward(states)[indices, actions]
        q_next = self.q_next.forward(states_).max(dim=1)[0]

        q_next[dones] = 0.0
        q_target = rewards + self.gamma*q_next

        autogradGrads=[torch.autograd.grad(q_pred[k],self.q_eval.parameters(), retain_graph=True, create_graph=True)\
                    for k in np.arange(self.batch_size)]
            
        q_Gradients = torch.cat([ \
                                torch.unsqueeze(torch.cat([ 
                                            torch.flatten(grad) for grad in\
                                                autogradGrads[k]\
                                ],0),0)\
                              for k in np.arange(self.batch_size)],0).detach().numpy()

        weights = torch.from_numpy(np.array([np.linalg.norm(q_Gradients[k]) for k in np.arange(self.batch_size)]))

        baselines_batch = self.baseline(states, actions)
        
        if self.save_var:
            vector_tmp = (q_target.detach() - q_pred.detach() - baselines_batch.reshape([self.batch_size]).detach()) * weights
            firstTerm = torch.sum(vector_tmp*vector_tmp,0)
            secondTerm = torch.sum(torch.mean(vector_tmp,0)**2)
            self.var_tmp.append(firstTerm - secondTerm)

        if self.save_norm:
            self.norm_tmp.append(torch.norm(torch.matmul(torch.from_numpy(np.transpose(q_Gradients)), baselines_batch.detach())))


        loss1 = torch.norm(torch.matmul(torch.from_numpy(np.transpose(q_Gradients)), baselines_batch))
        loss2 = self.baseline.loss(q_target.detach() * weights, (q_pred.detach() + baselines_batch.reshape([self.batch_size])) * weights)
        loss3 = (torch.norm(torch.matmul(q_target.detach() - q_pred.detach() - baselines_batch.reshape([self.batch_size]), torch.from_numpy(q_Gradients))) / self.batch_size) ** 2
        loss_bl = loss1 + loss2 + loss3
        loss_bl.backward()
        self.baseline.optimizer.step()

        self.q_eval.optimizer.zero_grad()
        self.baseline.optimizer.zero_grad()
        
        loss = self.q_eval.loss(q_target, q_pred + baselines_batch.detach()).to(self.q_eval.device)
        loss.backward()
        self.q_eval.optimizer.step()
        self.learn_step_counter += 1

        self.q_eval.optimizer.zero_grad()
        self.baseline.optimizer.zero_grad()

        self.decrement_epsilon()

    
    def train(self, n_epochs=2000, max_step=1000, load_checkpoint=False, verbose=0):   
        """
            Trains the agent
            n_epochs int -- number of epochs
            max_step int -- maximum length of sampled trajectory from the simulator
            load_checkpoint -- whether use old or new networks
            verbose int -- verbosity parameter, set this positive and print meanRewards every *verbose* epochs
        """

        stats = {'meanRewards': np.zeros([n_epochs]), 'nSteps': np.zeros([n_epochs])}

        #policy handler for the simulator
        def policyHandler(state):      
            if np.random.random() > self.epsilon:
                if self.type == "pic":
                    action = np.array(self.SampleActionDiscrete(np.transpose(state, (2, 0, 1))))
                else:
                    action = np.array(self.SampleActionDiscrete(state))
            else:
                action = np.random.choice(self.action_space)
            return action

        print("Training DQN VR.....")

        best_score = -np.inf
        n_steps = 0

        if load_checkpoint:
            agent.load_models()

        # fname = 'DQN_' + '_lr' + str(self.lr) +'_' + str(n_epochs) + 'games'
        # figure_file = 'plots/' + fname + '.png'

        for epochId in tqdm.tqdm(np.arange(n_epochs)):

            if self.save_norm:
                self.norm_tmp = []

            if self.save_var:
                self.var_tmp = []

            stat_steps = 0
            rewards = 0
            s = self.env.reset()
            while True:
                # env.render()
                a = policyHandler(s)
                s_, r, terminal, _ = self.env.step(a)
                d = 0
                rewards += r
                stat_steps += 1
                if terminal:
                    d = 1
                self.store_transition(s, a, r, s_, d)
                self.learn()
                s = s_
                if terminal:
                    break

            if self.save_norm:
                mean_norm = torch.mean(torch.tensor(self.norm_tmp))
                if np.isnan(mean_norm):
                    mean_norm = torch.mean(torch.tensor([0.]))
                self.norms.append(mean_norm)

            if self.save_var:
                mean_var = torch.mean(torch.tensor(self.var_tmp))
                if np.isnan(mean_var):
                    mean_var = torch.mean(torch.tensor([0.]))
                self.variances.append(mean_var)
            
            if self.save_var:
                qv = self.q_eval.forward(self.test_states).max(dim=1)[0]
                self.Qvalues.append(torch.mean(qv))

            avg_score = np.mean(stats['meanRewards'][-100:])

            if avg_score > best_score:
                if not load_checkpoint:
                    self.save_models()
                best_score = avg_score

            if load_checkpoint and n_steps >= 18000:
                break
            
            stats['meanRewards'][epochId]=rewards
            stats['nSteps'][epochId]=stat_steps

            if(verbose>0):
                if(epochId % verbose==0):
                    print("......","meanReward:",stats['meanRewards'][epochId])

        print("DONE")

        if self.save_norm:
            stats['norms'] = self.norms
        if self.save_var:
            stats['variances'] = self.variances
            stats['Qvalues'] = self.Qvalues

        return stats
    
    def evaluate(self, n_samples=2000, max_step=1000):
        """
            Evaluates the agent by sampling from the environment
            n_samples int -- number of trajectories for estimation
            max_step int -- maximum length of trajectory allowed in simulator
        """
        
        stats = {'rewardMean': 0,'rewardStd': 0}

        #policy handler for the simulator
        def policyHandler(state):
            if self.type == "pic":      
                action = np.array(self.SampleActionDiscrete(np.transpose(state, (2, 0, 1))))
            else:
                action = np.array(self.SampleActionDiscrete(state))
            return action


        stateSampler= pySim.GymResetSampler(self.simulator.gymInstance)
        if self.type == 'pic':
            rewards = \
                    self.simulator.SampleTrajectoriesFromStateSampler( stateSampler, policyHandler,n_samples,\
                        returnRewards=True, maxIterations=max_step, stateMemorySize=self.frames_in_state, grayscale=self.grayscale, downsample=self.downsample)
        else:
            rewards = \
                    self.simulator.SampleTrajectoriesFromStateSampler( stateSampler, policyHandler,n_samples,\
                        returnRewards=True, maxIterations=max_step)
        stats['rewardMean']=np.mean(np.sum(rewards[:,0,:],axis=1))
        stats['rewardStd']=np.std(np.sum(rewards[:,0,:],axis=1))
            
        return stats

##################################################################

