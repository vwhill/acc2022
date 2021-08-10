# -*- coding: utf-8 -*-
"""
Created on Sat Aug 7 2021

@author: Vincent W. Hill
Utilities for "Deep Reinforcement Learning Control for a Chaotic Dynamical System"
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
from scipy import signal as sig
import math
from gym import Env
from gym.spaces import Discrete, Box
from tensorflow.keras.models import Sequential
import tensorflow.keras.layers as layers
from rl.agents import DQNAgent, SARSAAgent
from rl.policy import BoltzmannQPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory

rng = np.random.default_rng(69)  # seeded sim
# rng = np.random.default_rng()  # unseeded sim

#%% Classes

class DoublePendulum():
    def __init__(self):
        self.m0 = 1.5   # mass of cart
        self.m1 = 0.5   # mass of pendulum 1
        self.m2 = 0.75  # mass of pendulum 2
        self.L1 = 0.5   # length of pendulum 1
        self.L2 = 0.75  # length of pendulum 2
        self.g = 9.81   # gravitational constant
        
        self.x0 = 0.    # initial value for x
        self.xd0 = 0.   # initial value for x_dot
        self.t1 = 0.    # initial value for theta_1
        self.td1 = 0.   # initial value for theta_dot_1
        self.t2 = 0.    # initial value for theta_2
        self.td2 = 0.   # initial value for theta_dot_2
        self.statevec = np.array([])  # state vector
        
        self.u = 0.  # control input
        
        self.x1 = self.x0 + self.L1 * math.sin(self.t1)
        self.x1init = self.x1
        
        self.y1 = self.L1 * math.cos(self.t1)
        self.y1init = self.y1
        
        self.x2 = self.x0 + self.L1 * math.sin(self.t1) + self.L2 * math.sin(self.t2)
        self.x2init = self.x2
        
        self.y2 = self.L1 * math.cos(self.t1) + self.L2 * math.cos(self.t2)
        self.y2init = self.y2
        
        self.position = np.array([self.x1, self.y1, self.x2, self.y2])
        
        self.state_history = [[], [], [], [], [], []]
        self.position_history = [[], [], [], []]
        self.control_history = []
        
        self.get_mats()
    
    def get_state(self, xin):
        self.statevec = xin.copy()
        self.statevec[1] = normalize(self.statevec[1].item())
        self.statevec[2] = normalize(self.statevec[2].item())
        
        self.x0 = xin[0].item()
        self.t1 = normalize(xin[1].item())
        self.t2 = normalize(xin[2].item())
        self.xd0 = xin[3].item()    
        self.td1 = xin[4].item()
        self.td2 = xin[5].item()
        
        self.x1 = self.x0 + self.L1 * math.sin(self.t1)
        self.y1 = self.L1 * math.cos(self.t1)
        
        self.x2 = self.x0 + self.L1 * math.sin(self.t1) + self.L2 * math.sin(self.t2)
        self.y2 = self.L1 * math.cos(self.t1) + self.L2 * math.cos(self.t2)
        
        self.position = np.array([self.x1, self.y1, self.x2, self.y2])
    
    def save_state(self, xin):
        for i in range(0, len(xin)):
            self.state_history[i].append(xin[i].item())
        
        self.position_history[0].append(self.x1)
        self.position_history[1].append(self.y1)
        self.position_history[2].append(self.x2)
        self.position_history[3].append(self.y2)
        
        self.control_history.append(self.u)
        
    def get_mats(self):
        self.D = np.array([[self.m0 + self.m1 + self.m2, 
                            (.5 * self.m1 + self.m2) * self.L1 * math.cos(self.t1), 
                            .5 * self.m2 * self.L2 * math.cos(self.t2)], 
                           
                           [(.5 * self.m1 + self.m2) * self.L1 * math.cos(self.t1), 
                            (0.33 * self.m1 + self.m2) * self.L1**2,
                            .5 * self.m2 * self.L1 * self.L2 * math.cos(self.t1 - self.t2)], 
                           [.5 * self.m2 * self.L2 * math.cos(self.t2),
                            .5 * self.m2 * self.L1 * self.L2 * math.cos(self.t1 - self.t2),
                            .33 * self.m2 * self.L2**2]])
        
        self.Dinv = la.inv(self.D)
        
        self.C = np.array([[0., 
                            -(.5 * self.m1 + self.m2) * self.L1 * math.sin(self.t1) * self.td1,
                           -.5 * self.m2 * self.L2 * math.sin(self.t2) * self.td2],
                           [0.,
                            0.,
                            .5 * self.m2 * self.L1 * self.L2 * math.sin(self.t1 - self.t2) * self.td2],
                           [0.,
                            -.5 * self.m2 * self.L1 * self.L2 * math.sin(self.t1 - self.t2) * self.td1,
                            0.]])
        
        self.G = np.array([[0.],
                           [-0.5 * (self.m1 + self.m2) * self.L1 * self.g * math.sin(self.t1)],
                           [-0.5 * self.m2 * self.g * self.L1 * math.sin(self.t2)]])
        
        self.H = np.array([[1.],
                           [0.],
                           [0.]])
        
        self.A = np.vstack((np.concatenate((np.zeros((3, 3)), np.eye(3)), axis=1), 
                           np.concatenate((np.zeros((3, 3)), -self.Dinv @ self.C), axis=1)))
        
        self.Lx = np.vstack((np.zeros((3, 1)), -self.Dinv @ self.G))
        
        self.B = np.vstack((np.zeros((3, 1)), self.Dinv @ self.H))
        
    def get_linear_model(self):
        ddet = 1 / (4 * self.m0 * self.m1 + 3 * self.m0 * self.m2 + self.m1**2 + self.m1 * self.m2)
    
        a42 = -1.5 * ddet * self.g * (2 * self.m1**2 + 5 * self.m1 * self.m2 + 2 * self.m2**2)
        a43 = 1.5 * ddet * self.m1 * self.m2 * self.g
        a52 = 1.5 * (ddet / self.L1) * self.g *  (4 * self.m0 * self.m1 + 8 * self.m0 * self.m2 + \
                                                  4 * self.m1**2 + \
                                                  9 * self.m1 * self.m2 + 2 * self.m2**2)
        a53 = -4.5 * (ddet / self.L1) * self.g * (2 * self.m0 * self.m2 + self.m1 * self.m2)
        a62 = -4.5 * (ddet / self.L2) * (2 * self.m0 * self.m1 + 4 * self.m0 * self.m2 + \
                                         self.m1 ** 2 + 2 * self.m1 * self.m2)
        a63 = 1.5 * (ddet / self.L2) * (self.m1 ** 2 + 4 * self.m0 * self.m1 + \
                                        12 * self.m0 * self.m2 + \
                                        4 * self.m1 * self.m2)
        
        self.Alin = np.array([[0., 0., 0., 1., 0., 0.],
                              [0., 0., 0., 0., 1., 0.],
                              [0., 0., 0., 0., 0., 1.],
                              [0., a42, a43, 0., 0., 0.],
                              [0., a52, a53, 0., 0., 0.],
                              [0., a62, a63, 0., 0., 0.]])
        
        b4 = ddet * (4 * self.m1 + 3 * self.m2)
        b5 = (-3 * ddet / self.L1) * (2 * self.m1 + self.m2)
        b6 = (2 * ddet * self.m2) / self.L2
            
        self.Blin = np.array([[0.], [0.], [0.], [b4], [b5], [b6]])
        
        self.Clin = np.identity(6)
        self.Dlin = np.zeros((6, 1))
            
    def plot_results(self, **kwargs):
        plt.figure()
        t = np.ndarray.tolist(np.linspace(0, kwargs["tend"], int(kwargs["tend"]/kwargs["dt"])+1))
        plt.plot(t, self.position_history[1], label='Midpoint')
        plt.plot(t, self.position_history[3], label='Tip')
        plt.xlabel("Time (sec)")
        plt.ylabel("Pendulum y positions (m)")
        plt.legend()
        plt.title(kwargs["title"])
        
        plt.figure()
        t = np.ndarray.tolist(np.linspace(0, kwargs["tend"], int(kwargs["tend"]/kwargs["dt"])+1))
        plt.plot(t, self.control_history)
        plt.xlabel("Time (sec)")
        plt.ylabel("Control input (N)")
        plt.title(kwargs["title"])
        
        plt.figure()
        plt.plot(self.state_history[0], np.ndarray.tolist(np.zeros(len(self.state_history[0]))), linewidth=2, label='Cart')
        plt.plot(self.position_history[0], self.position_history[1], linewidth=2, label='Midpoint')
        plt.plot(self.position_history[2], self.position_history[3], linewidth=2, label='Tip')
        plt.xlabel('x-position (m)')
        plt.ylabel('y-position (m)')
        plt.legend()
        plt.title(kwargs["title"])

class DoubPendEnv(DoublePendulum, Env):
    def __init__(self):
        self.dt = 0.01
        self.tend = 5.
        self.time = 0.
        self.dp = DoublePendulum()
        x = np.array([[0.], [np.deg2rad(0.1)], [0.], [0.], [0.], [0.]])   # initial state
        self.dp.get_state(x)
        self.dp.save_state(x)
        self.state = np.concatenate((self.dp.statevec.flatten(), self.dp.position), axis=0)
        self.actsize = 51
        self.actmag = [-5, 5]
        self.action_space = Discrete(self.actsize)
        self.observation_space = Box(low = -50 * np.ones(10),
                                     high = 50 * np.ones(10))
        
        ds_low = np.array([-100.,                  # x
                           -np.deg2rad(5.),        # t1
                           -np.deg2rad(5.),        # t2
                           -100.,                  # xdot
                           -np.deg2rad(10.),       # t1dot
                           -np.deg2rad(10.),       # t2dot
                           self.dp.x1init - 100,   # x1
                           self.dp.y1init - 0.1,  # y1
                           self.dp.x2init - 100,   # x2
                           self.dp.y2init - 0.1]) # y2
        
        ds_high = np.array([100.,                  # x
                           np.deg2rad(5.),         # t1
                           np.deg2rad(5.),         # t2
                           100.,                   # xdot
                           np.deg2rad(10.),        # t1dot
                           np.deg2rad(10.),        # t2dot
                           self.dp.x1init + 100,   # x1
                           self.dp.y1init + 0.1,  # y1
                           self.dp.x2init + 100,   # x2
                           self.dp.y2init + 0.1]) # y2
        
        self.desired_state = [ds_low.copy(), ds_high.copy()]
        
    def step(self, action):
        self.get_action(self.actmag, action)
        x = rk4(doubpend, self.dp.statevec, self.dt, dp = self.dp)
        self.dp.get_state(x)
        self.dp.save_state(x)
        self.dp.get_mats()
        self.state = np.concatenate((self.dp.statevec.flatten(), self.dp.position), axis=0)
        
        if (self.state >= self.desired_state[0]).all() and (self.state <= self.desired_state[1]).all():
            reward = 1
        else:
            reward = -1
        info = {}
        
        self.time += self.dt
        
        if self.time >= self.tend - self.dt:
            done = True
        else:
            done = False
        
        return self.state, reward, done, info
    
    def reset(self):
        self.dp = DoublePendulum()
        x = np.array([[0.], [np.deg2rad(0.1)], [0.], [0.], [0.], [0.]])   # initial state
        self.dp.get_state(x)
        self.dp.save_state(x)
        self.state = np.concatenate((self.dp.statevec.flatten(), self.dp.position), axis=0)
        self.time = 0.
        return self.state
    
    def get_action(self, mag, action):
        actspace = np.linspace(mag[0], mag[1], self.actsize)
        self.dp.u =  actspace[action].item()

class LQR:
    def __init__(self, F, G, statedim, indim):
        self.dt = 0.01
        C = np.identity(statedim)
        D = np.zeros((statedim, statedim))
        Q = np.identity(statedim)
        Q[0, 0] = 5.
        Q[1, 1] = 50.
        Q[2, 2] = 50.
        Q[3, 3] = 20.
        Q[4, 4] = 700.
        Q[5, 5] = 700.
        R = np.identity(indim)
        P = la.solve_discrete_are(F, G, Q, R, e=None, s=None, balanced=True)
        self.K = la.inv(G.T@P@G+R)@(G.T@P@F)

        self.sysout = sig.StateSpace(F-G@self.K, G@self.K, C, D, dt=self.dt)

        self.F = self.sysout.A
        self.G = self.sysout.B

#%% Functions

def doubpend(x, **kwargs):
    t0, td0, t1, td1, t2, td1 = x
    dp = kwargs["dp"]
    u = dp.u
    A = dp.A
    Lx = dp.Lx
    B = dp.B
    return A @ x + B * u + Lx

def rk4(f, x, h, **kwargs):
    """ Implements a classic Runge-Kutta integration RK4.

    Args:
        f (function): function to integrate, must take x as the first argument
            and arbitrary kwargs after
        x (numpy array, or float): state needed by function
        h (float): step size
    Returns:
        (numpy array, or float): Integrated state
    """
    k1 = h * f(x, **kwargs)
    k2 = h * f(x + 0.5 * k1, **kwargs)
    k3 = h * f(x + 0.5 * k2, **kwargs)
    k4 = h * f(x + k3, **kwargs)
    return x + (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

def discretize(dt, A, B, C, D):
    sys = sig.StateSpace(A, B, C, D)
    sysd = sys.to_discrete(dt)
    F = sysd.A
    G = sysd.B
    return F, G

def build_dense_model(state_shape, action_shape):
    model = Sequential()
    density = 1000
    numlayer = 10
    # model.add(layers.Flatten(input_shape = (1, state_shape[0])))
    model.add(layers.Dense(density, activation='relu', input_shape = state_shape))
    for i in range(0, numlayer):
        model.add(layers.Dense(density, activation='relu'))
    
    model.add(layers.Dense(action_shape, activation='linear'))
    return model

def build_conv_model(state_shape, action_shape):
    model = Sequential()
    density = 24
    numlayer = 1
    model.add(layers.Conv1D(32, 5, activation='relu', input_shape = (state_shape[0], 1)))
    for i in range(0, numlayer):
        model.add(layers.Dense(density, activation='relu'))
    
    model.add(layers.Flatten())
    model.add(layers.Dense(action_shape, activation='linear'))
    return model

def build_lstm_model(state_shape, action_shape):
    model = Sequential()
    epochs = 128
    model.add(layers.Reshape((state_shape[1], state_shape[2]), input_shape=state_shape))
    model.add(layers.LSTM(epochs, activation='tanh', recurrent_activation='hard_sigmoid', 
                          return_sequences=True))
    model.add(layers.Dense(action_shape, activation='linear'))
    return model

def build_dqn_agent(model, actions):
    policy = BoltzmannQPolicy()
    memory = SequentialMemory(limit=10000000, window_length=1)
    dqn = DQNAgent(model=model, memory=memory, policy=policy, 
                  nb_actions=actions, nb_steps_warmup=1000, 
                  enable_double_dqn=False, enable_dueling_network=True, 
                  target_model_update=1e-3)
    return dqn

def build_sarsa_agent(model, actions):
    policy = EpsGreedyQPolicy()    
    sarsa = SARSAAgent(model=model, policy=policy, nb_actions=actions)
    return sarsa
    
def train_agent(agent, env):
    agent.fit(env, nb_steps=10000000, visualize=False, verbose=1) # edit made to L56 of dqn.py!!!!!
    agent.save_weights('dqn_weights.h5f', overwrite=True)
    # agent.test(env, nb_episodes = 5, visualize=False)

def normalize(num):
    lower = np.deg2rad(-180.0)
    upper = np.deg2rad(180.0)
    res = num
    if num > upper or num == lower:
        num = lower + abs(num + upper) % (abs(lower) + abs(upper))
    if num < lower or num == upper:
        num = upper - abs(num - lower) % (abs(lower) + abs(upper))

    res = lower if res == upper else num

    return res