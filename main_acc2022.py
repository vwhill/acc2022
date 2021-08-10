# -*- coding: utf-8 -*-
"""
Created on Sat Aug 7 2021

@author: Vincent W. Hill
Main script for "Deep Reinforcement Learning Control for a Chaotic Dynamical System"
"""
#%% Imports

import numpy as np
import utilities as util
from tensorflow.keras.optimizers import Adam

rng = np.random.default_rng(69)  # seeded sim
# rng = np.random.default_rng()  # unseeded sim

#%% Uncontrolled Simuation

plant_unc = util.DoublePendulum()

dt = 0.01
tend = 5.
maxiter = int(tend/dt)

plant_unc.u = 0.
x = np.array([[0.], [0.], [0.], [0.1], [np.deg2rad(1.)], [0.]])  # initial state
plant_unc.get_state(x)
plant_unc.save_state(x)

for i in range(0, maxiter):
    x = util.rk4(util.doubpend, x, dt, dp = plant_unc)
    plant_unc.get_state(x)
    plant_unc.save_state(x)
    plant_unc.get_mats()

plant_unc.plot_results(dt = dt, tend = tend, title = "Uncontrolled")

#%% LQR Control

plant_lqr = util.DoublePendulum()

x = np.array([[0.], [0.], [0.], [0.1], [np.deg2rad(1.)], [0.]])  # initial state
plant_lqr.get_state(x)
plant_lqr.save_state(x)
plant_lqr.get_linear_model()

F_lqr, G_lqr = util.discretize(dt, plant_lqr.Alin, plant_lqr.Blin, plant_lqr.Clin, plant_lqr.Dlin)

control = util.LQR(F_lqr, G_lqr, 6, 1)

plant_lqr.u = -control.K @ x

for i in range(0, 100):
    x = util.rk4(util.doubpend, x, dt, dp = plant_lqr)
    plant_lqr.get_state(x)
    plant_lqr.save_state(x)
    plant_lqr.get_mats()
    plant_lqr.u = (-control.K @ x).item()

plant_lqr.plot_results(dt = dt, tend = 1., title = "LQR Control")

#%% Deep Reinforcement Learning Control

env = util.DoubPendEnv()
env.dt = dt
env.tend = tend

state_shape = env.observation_space.shape
action_shape = env.action_space.n

model = util.build_dense_model(state_shape, action_shape)
model.summary()

agent = util.build_agent(model, action_shape)

agent.compile(Adam(lr=0.0001), metrics=['mae'])

dotrain = 1

if dotrain == 1:
    util.train_agent(agent, env)
else:
    agent.load_weights("dqn_weights.h5f")
    
agent.test(env, nb_episodes = 1, visualize=False)

plant_drl = env.dp

plant_drl.plot_results(dt = dt, tend = tend, title = "DRL Control")
