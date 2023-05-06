#!/usr/bin/env python
# coding: utf-8

## NECESSARY UPDATES
# 1 - ONE STEP GAMMA OPT
# 2 - OPT GAMMA TILL CONVERGENCE

## NOTE THERE MAY BE BIAS FROM OUR INITIALIZATION E.G. OF BETA

## LOOK AT POPULATION PARAMETERS OVER 100 RUNS (SEEDS)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import pickle

NUM_RUNS = 100
NUM_SUBJECTS = 10000
MAX_EPOCHS = 200


def sigmoid(z):
    return 1 / (1 + np.exp(-1 * z))


def generate_data(
    random_seed=0, num_subjects=NUM_SUBJECTS, num_obs_range=[50, 100],
    beta=[-2., 2.], sigma=[1., 1.]):

    rng = np.random.RandomState(random_seed)

    subj_specific_effects = [s * rng.randn(num_subjects) for s in sigma]

    # observed features

    n_obs_per_subject = rng.randint(
        *num_obs_range,
        size=num_subjects
    )

    sid = np.array([i for i, n_obs in enumerate(n_obs_per_subject) for _ in range(n_obs)]).astype(np.int32)
    sid_nobs = np.array([n_obs for i, n_obs in enumerate(n_obs_per_subject) for _ in range(n_obs)]).astype(np.int32)

    ss_coefs = np.array([
        [c0, c1]
        for n_obs, c0, c1 in zip(n_obs_per_subject, *subj_specific_effects)
        for _ in range(n_obs)
    ])

    X = np.stack([np.ones(len(sid)), rng.randn(len(sid))]).T.astype(np.float32)

    # outcomes

    y_logits = np.sum(X * np.array(beta) + X * ss_coefs, axis=-1)

    y_probs = sigmoid(y_logits)
    y = (rng.rand(len(y_probs)) < y_probs).astype(np.float32)

    # summary of outcomes

    return X, sid, sid_nobs, y


class NeuralFeaturesGLMM(nn.Module):
    def __init__(self, num_population_effects, num_subjects, num_subject_specific_effects):
        super(NeuralFeaturesGLMM, self).__init__()
        
        self.pop_params_layer = nn.Linear(num_population_effects, 1, bias=False)
        
        self.ss_params_layer = nn.Embedding(num_subjects, num_subject_specific_effects)
        self.ss_params_mu = 0#nn.Parameter(torch.zeros(num_subject_specific_effects, 1))
        self.ss_params_logsigma = nn.Parameter(torch.zeros(num_subject_specific_effects, 1))
        
        self.pop_nll = nn.BCEWithLogitsLoss(reduction='none')
        self.ssp_nll = nn.GaussianNLLLoss(reduction='none')
        
        self.ss_params = self.ss_params_layer.parameters()
        #self.pop_params = (*self.pop_params_layer.parameters(), self.ss_params_mu, self.ss_params_logsigma)
        self.pop_params = (*self.pop_params_layer.parameters(), self.ss_params_logsigma)

    def forward(self, x, z, sid):
        logits = torch.squeeze(self.pop_params_layer(x))
        ss_params = self.ss_params_layer(sid)
        logits += torch.sum(ss_params * z, axis=-1)
        
        return logits
        
    def nll(self, x, z, sid, sid_nobs, y):
        
        y_pred = self.forward(x, z, sid)
        
        # cross entropy loss
        nll = self.pop_nll(y_pred, y)
        
        # subject specific parameters
        ssp = torch.t(self.ss_params_layer(sid))
        
        # subject specific effects likelihood
        # we divide by num obs for each subject to avoid repeating for each observation
        nll += torch.sum(self.ssp_nll(ssp, self.ss_params_mu, torch.exp(2 * self.ss_params_logsigma)), 0) / sid_nobs
        
        return torch.sum(nll)


class BatchLoader:
    def __init__(self, idx_arr, *arrs):
        self.idx_arr = idx_arr.copy()
        self.arrs = [arr.copy() for arr in arrs]
        self.num_subjects = len(np.unique(idx_arr))
    def get_batches(self, batch_size=1):
        for ndx in range(0, self.num_subjects, batch_size):
            idx = (self.idx_arr >= ndx) & (self.idx_arr < min(ndx + batch_size, self.num_subjects))
            yield (torch.from_numpy(arr[idx]) for arr in (self.idx_arr, *self.arrs))

    
def train(model, batchloader, max_epochs, ss_steps=1, pop_steps=1, eps=1e-4):
    
    ss_opt = optim.Adam(model.ss_params)
    pop_opt = optim.Adam(model.pop_params)
    
    all_loss = []
    best_loss = np.inf
    
    for epoch in range(max_epochs):
        
        epoch_loss = []
        
        for batch_idx, (sid_batch, sid_nobs_batch, x_batch, z_batch, y_batch) in enumerate(
            batchloader.get_batches(batch_size=50)):
            
            #print(x_batch.dtype, sid_batch.dtype, sid_nobs_batch.dtype, y_batch.dtype)
            
            for step in range(ss_steps):
            
                ss_opt.zero_grad()              
                
                loss = model.nll(x_batch, z_batch, sid_batch, sid_nobs_batch, y_batch)
                loss.backward()
                ss_opt.step()
                
            for step in range(pop_steps):
                
                pop_opt.zero_grad()
                
                loss = model.nll(x_batch, z_batch, sid_batch, sid_nobs_batch, y_batch)
                loss.backward()
                pop_opt.step()
                
            current_loss = loss.detach().numpy()
            epoch_loss.append(current_loss)
            
        epoch_loss = np.mean(epoch_loss)
        all_loss.append(epoch_loss)
        print('Epoch %i; Loss = %.3f' % (epoch, epoch_loss))

        if (epoch_loss / best_loss) > (1. - eps):
            break

        best_loss = min(best_loss, epoch_loss)
        
    return all_loss


def main():

    results = []
    loss_plots = []

    for i in range(NUM_RUNS):

        X, sid, sid_nobs, y = generate_data(num_subjects=NUM_SUBJECTS, random_seed=i)

        loader = BatchLoader(sid, sid_nobs, X, X, y)

        mdl = NeuralFeaturesGLMM(2, NUM_SUBJECTS, 2)

        all_loss = train(mdl, loader, MAX_EPOCHS, ss_steps=1)

        beta1, beta2 = mdl.pop_params_layer.weight.detach().numpy()[0]
        logsigma1, logsigma2 = mdl.ss_params_logsigma.detach().numpy()

        results.append({
            'beta1': beta1,
            'beta2': beta2,
            'logsigma1': logsigma1[0],
            'logsigma2': logsigma2[0]
            })

        loss_plots.append(all_loss)

    pd.DataFrame(results).to_csv('sim_1_results.csv')

    with open('sim_1_loss_plots.pickle', 'wb') as f:
        pickle.dump(all_loss, f)


if __name__ == '__main__':
    main()


