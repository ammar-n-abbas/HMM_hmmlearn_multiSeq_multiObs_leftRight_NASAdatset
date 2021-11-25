# ********************************************* NASA *****************************************************

# -*- coding: utf-8 -*-
"""
Created on Mon Aug 30 13:03:23 2021

@author: ammar@scch
"""

############################################################################################################
#                                           IMPORTING LIBRARIES
# ##########################################################################################################

import itertools

import time
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import gym
import random
import sklearn.preprocessing
import sklearn.pipeline
import torch
import random
import warnings
import copy
import seaborn as sns
import tensorflow as tf
import keras
import math

from IPython.display import clear_output
from gym import spaces
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.deepq.policies import LnMlpPolicy, LnCnnPolicy
from stable_baselines import DQN
from stable_baselines.common.env_checker import check_env
from IPython.display import display
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.kernel_approximation import RBFSampler
from sklearn.decomposition import PCA
from lib import plotting
from collections import deque
from sklearn.preprocessing import StandardScaler, MinMaxScaler

warnings.filterwarnings("ignore", category=DeprecationWarning)
standard = StandardScaler()
minmax = MinMaxScaler()

############################################################################################################
#                                           DATASET PREPARATION
# ##########################################################################################################

dir_path = os.getcwd()
dataset = 'train_FD001.txt'
df = pd.read_csv(dir_path + r'/CMAPSSData/' + dataset, sep=" ", header=None, skipinitialspace=True).dropna(axis=1)
df = df.rename(columns={0: 'unit', 1: 'cycle', 2: 'W1', 3: 'W2', 4: 'W3'})
df_A = df[df.columns[[0, 1]]]
df_W = df[df.columns[[2, 3, 4]]]
df_S = df[df.columns[list(range(5, 26))]]
df_X = pd.concat([df_W, df_S], axis=1)

'''RUL as sensor reading'''
df_A['RUL'] = 0
for i in range(1, 101):
    df_A['RUL'].loc[df_A['unit'] == i] = df_A[df_A['unit'] == i].cycle.max() - df_A[df_A['unit'] == i].cycle

'''Standardization'''
df_X = standard.fit_transform(df_X)

'''train_test split'''
engine_unit = 1

'''##
# %% ENGINE UNIT SPECIFIC DATA
engine_unit = 1
engine_df_A = df_A[df_A['unit'] == engine_unit]
engine_df_X = df_X.iloc[engine_df_A.index[0]:engine_df_A.index[-1] + 1]
engine_df_W = df_W.iloc[engine_df_A.index[0]:engine_df_A.index[-1] + 1]

##
# %% NORMALIZE DATA
X = scaler.fit_transform(engine_df_X)
# X = (((engine_df_X - engine_df_X.mean()) / engine_df_X.std()).fillna(0))
# X = ((engine_df_X - engine_df_X.min()) / (engine_df_X.max() - engine_df_X.min())).fillna(0)).values'''

'''
##
# %% READ RUL & APPEND

# df_RUL = pd.read_csv(dir_path + '/CMAPSSData/RUL_FD001.txt', sep=" ", header=None, skipinitialspace=True).dropna(axis=1)
# df_RUL.columns = ['RUL']
# df_z_scaled_RUL = df_z_scaled.join(df_RUL, 1)

##
# %% REGRESSION TO GET "RUL distribution"

# x = df_z_scaled_RUL.iloc[:,list(range(5, 26))]
# y = df_RUL

##
# %% DIMENSIONALITY REDUCTION TO GET "HEALTH INDICATOR"

sensor_data = df_z_scaled.iloc[:, list(range(5, 26))].dropna(axis=1)
pca = PCA(n_components=1)
principalComponents = (1 - pca.fit_transform(sensor_data))

pdf = pd.DataFrame(data=principalComponents, columns=['health indicator'])
pdf_normalized = (pdf - pdf.min()) / (pdf.max() - pdf.min()) * 100

df_scaled_principal = df_z_scaled.join(pdf_normalized, 1)
df_scaled_principal = df_scaled_principal.rename(columns={0: 'engine unit', 1: 'cycle'})


##
# %% VISUALIZATION
engine_unit = 76
engine_df = df_scaled_principal[df_scaled_principal['engine unit'] == engine_unit]
# engine_df.plot.line('cycle', 'health indicator')
# plt.show()

HI = np.array(engine_df['health indicator'])[0:191].astype(np.float32)
# plt.plot(HI)
# plt.show()
'''

############################################################################################################
# **********************************************************************************************************
#                                       HIDDEN MARKOV MODEL (LIBRARY)
# **********************************************************************************************************
# ##########################################################################################################


from hmmlearn import hmm
from random import randint
import pickle

# df_S = df[df.columns[[6, 8, 11, 12, 15, 16, 19]]]
# df_hmm = pd.concat([df_A['cycle'], df_S], axis=1)
df_hmm = minmax.fit_transform(df_S)

df_hmm = pd.DataFrame(df_hmm)
cols_to_drop = df_hmm.nunique()[df_hmm.nunique() == 1].index
df_hmm = df_hmm.drop(cols_to_drop, axis=1)
cols_to_drop = df_hmm.nunique()[df_hmm.nunique() == 2].index
df_hmm = df_hmm.drop(cols_to_drop, axis=1).to_numpy()

lengths = [df[df['unit'] == i].cycle.max() for i in range(1, df_A['unit'].max() + 1)]
# o = df_X[df_A[df_A['unit'] == 1].index[0]:df_A[df_A['unit'] == 1].index[-1] + 1]

num_states = 15
remodel = hmm.GaussianHMM(n_components=num_states,
                          n_iter=500,
                          verbose=True,
                          init_params="cm", params="cmt")

transmat = np.zeros((num_states, num_states))
# Left-to-right: each state is connected to itself and its
# direct successor.
for i in range(num_states):
    if i == num_states - 1:
        transmat[i, i] = 1.0
    else:
        transmat[i, i] = transmat[i, i + 1] = 0.5

# Always start in first state
startprob = np.zeros(num_states)
startprob[0] = 1.0

remodel.startprob_ = startprob
remodel.transmat_ = transmat
remodel = remodel.fit(df_hmm, lengths)

# with open("HMM_model.pkl", "wb") as file: pickle.dump(remodel, file)
# with open("filename.pkl", "rb") as file: pickle.load(file)

state_seq = remodel.predict(df_hmm)
pred = [state_seq[df[df['unit'] == i].index[0]:df[df['unit'] == i].index[-1] + 1] for i in
        range(1, df_A['unit'].max() + 1)]

prob = remodel.predict_proba(df_hmm, lengths)
prob_next_step = remodel.transmat_[state_seq, :]

HMM_out = [prob[df[df['unit'] == i].index[0]:df[df['unit'] == i].index[-1] + 1]
           for i in range(1, df_A['unit'].max() + 1)]
failure_states = [pred[i][-1] for i in range(df_A['unit'].max())]

'''RUL Prediction - Monte Carlo Simulation'''
from sklearn.utils import check_random_state

transmat_cdf = np.cumsum(remodel.transmat_, axis=1)
random_state = check_random_state(remodel.random_state)

predRUL = []
for i in range(df_A[df_A['unit'] == 1]['cycle'].max()):
    RUL = []
    for j in range(1):
        cycle = 0
        pred_obs_seq = [df_hmm[i]]
        pred_state_seq = remodel.predict(pred_obs_seq)
        while pred_state_seq[-1] not in set(failure_states):
            cycle += 1
            prob_next_state = (transmat_cdf[pred_state_seq[-1]] > random_state.rand()).argmax()
            prob_next_obs = remodel._generate_sample_from_state(prob_next_state, random_state)
            pred_obs_seq = np.append(pred_obs_seq, [prob_next_obs], 0)
            pred_state_seq = remodel.predict(pred_obs_seq)
        RUL.append(cycle)
    # noinspection PyTypeChecker
    predRUL.append(round(np.mean(RUL)))

plt.plot(predRUL)
plt.plot(df_A[df_A['unit'] == 1].RUL)
plt.show()

plt.figure(0)
plt.plot(pred[0])
plt.xlabel('# Flights')
plt.ylabel('HMM states')

plt.figure(1)
E = [randint(1, df_A['unit'].max()) for p in range(0, 10)]
for e in E:
    plt.plot(pred[e - 1])
plt.xlabel('# Flights')
plt.ylabel('HMM states')
plt.legend(E, title='engine unit')

plt.figure(2)
plt.scatter(list(range(1, len(failure_states) + 1)), failure_states)
plt.xlabel('Engine units')
plt.ylabel('HMM states')
plt.legend(title='failure states')

plt.figure(3)
pca = PCA(n_components=2).fit_transform(df_hmm)
for class_value in range(num_states):
    # get row indexes for samples with this class
    row_ix = np.where(state_seq == class_value)
    plt.scatter(pca[row_ix, 0], pca[row_ix, 1])
plt.xlabel('PCA Feature 1')
plt.ylabel('PCA Feature 2')
plt.legend(list(range(0, num_states)), title='HMM states')

plt.show()

print('done')
