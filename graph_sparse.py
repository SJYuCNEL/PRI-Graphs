#  *          NAME OF THE PROGRAM THIS FILE BELONGS TO
#  *
#  *   file: graph_sparse.py
#  *
#  *    NEC Laboratories Europe GmbH. PROPRIETARY INFORMATION
#  *
#  * This software is supplied under the terms of a license agreement
#  * or nondisclosure agreement with NEC Laboratories Europe GmbH. and 
#  * may not becopied or disclosed except in accordance with the terms of that
#  * agreement. The software and its source code contain valuable 
#  * trade secrets and confidential information which have to be 
#  * maintained in confidence. 
#  * Any unauthorized publication, transfer to third parties or 
#  * duplication of the object or source code - either totally or in 
#  * part - is prohibited. 
#  *

#  *
#  *   Copyright (c) 2022 NEC Laboratories Europe GmbH. All Rights Reserved.
#  *
#  * Authors: Francesco Alesiani  francesco.alesiani@neclab.eu
#  *
#  * 2022 NEC Laboratories Europe GmbH. DISCLAIMS ALL 
#  * WARRANTIES, EITHER EXPRESS OR IMPLIED,
#  * INCLUDING BUT NOT LIMITED TO IMPLIED WARRANTIES OF MERCHANTABILITY
#  * AND FITNESS FOR A PARTICULAR PURPOSE AND THE WARRANTY AGAINST LATENT
#  * DEFECTS, WITH RESPECT TO THE PROGRAM AND THE ACCOMPANYING
#  * DOCUMENTATION.
#  *
#  * No Liability For Consequential Damages IN NO EVENT SHALL 2019 NEC 
#  * Laboratories Europe GmbH, NEC Corporation 
#  * OR ANY OF ITS SUBSIDIARIES BE LIABLE FOR ANY
#  * DAMAGES WHATSOEVER (INCLUDING, WITHOUT LIMITATION, DAMAGES FOR LOSS
#  * OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF INFORMATION, OR
#  * OTHER PECUNIARY LOSS AND INDIRECT, CONSEQUENTIAL, INCIDENTAL,
#  * ECONOMIC OR PUNITIVE DAMAGES) ARISING OUT OF THE USE OF OR INABILITY
#  * TO USE THIS PROGRAM, EVEN IF NEC Europe Ltd. HAS BEEN ADVISED OF THE
#  * POSSIBILITY OF SUCH DAMAGES.
#  *
#  *     THIS HEADER MAY NOT BE EXTRACTED OR MODIFIED IN ANY WAY.
#  */

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import networkx as nx
import torch
torch.set_default_tensor_type('torch.DoubleTensor')
import warnings
warnings.filterwarnings('ignore')
import torch.nn.functional as F
from torch import autograd
import math
   
      
def incidence(g1):
    '''
    get incidence matrix
    '''
    E = nx.incidence_matrix(g1, oriented=True)
    E = E.todense()
    return E       


def get_graph_from_incidence(E, w=None):
    
    if w is None:
        am = (E @ E.T !=0).int()
    else:
        am = (E @ torch.diag(w) @ E.T !=0).int()
    am = am.numpy()
    np.fill_diagonal(am, 0)
    g_ = nx.from_numpy_matrix(am)
    return g_


def vn_entropy(k, eps=1e-8):

    k = k/torch.trace(k)  # normalization

    eigv = torch.abs(torch.symeig(k, eigenvectors=True)[0])
    entropy = - torch.sum(eigv[eigv>0]*torch.log(eigv[eigv>0]+eps))
    return entropy


def entropy_loss(sigma, rho, beta, alpha, entropy_fn = vn_entropy):
    assert(beta>=0), "beta shall be >=0"
    sigma_diag = torch.diag(sigma)
    connectivity_loss = alpha * torch.sum(torch.log(sigma_diag[sigma_diag>0]))
    #print(connectivity_loss)
    if beta>0:
        loss = 0.5*(1-beta)/beta * entropy_fn(sigma) + entropy_fn(0.5 * (sigma+rho))
        return loss - connectivity_loss 
    else:
        return entropy_fn(sigma) - connectivity_loss 

   
# ---------------------------------------    
def sparsification_stochastic(G, tau, n_samples, epochs, lr, beta, alpha, loss_type = 'vn', seed=42, verbose=True,  hard=True, plot_flag = False):    
    
    torch.manual_seed(seed) 

    E = torch.from_numpy(incidence(G).astype(np.double)) # incidence matrix

    rho = E@E.T # lap matrix 
    m, n = G.number_of_edges(), G.number_of_nodes()    
    # init
    theta = torch.randn(m, 2, requires_grad=True) # parameters  

    if loss_type == 'vn':
        entropy_fn = vn_entropy
    # Optimization loop

    optimizer = torch.optim.Adam([theta], lr=lr)

    history = []
    cost_vec = np.zeros((epochs, n_samples))
    for epoch in range(epochs):
        cost = 0       
        for sample in range(n_samples):
            # Sampling
            z = F.gumbel_softmax(theta, tau, hard = hard)
            w = z[:, 1].squeeze()
            probs = torch.exp(theta[:,1].squeeze()-theta[:,0].squeeze()) / (1 + torch.exp(theta[:,1].squeeze()-theta[:,0].squeeze()))

            # Cost evaluation
            sigma = E@torch.diag(w)@E.T
            _loss = entropy_loss(sigma,rho,beta, alpha, entropy_fn)
                  
            cost = cost + _loss
            cost_vec[epoch,sample] = _loss            
        cost = cost/n_samples        
        # Gradient update
        cost.backward()
        optimizer.step()
        optimizer.zero_grad()
        # logging
        history.append(cost.item())
        if verbose and (epoch==0 or (epoch+1) % 100 == 0):
            print('[Epoch %4d/%d] loss: %f - w(mean): %f %% - theta(mean): %f(%f)' % (epoch+1, epochs, cost.item(), w.detach().mean()*100, theta.detach().mean(), theta.detach().std()))
    
    z = F.gumbel_softmax(theta, tau, hard=True)
    w = z[:,1].squeeze()
    probs = torch.exp(theta[:,1].squeeze()-theta[:,0].squeeze()) / (1 + torch.exp(theta[:,1].squeeze()-theta[:,0].squeeze()))
    
    sigma = E@torch.diag(w)@E.T    
    # some plot
    if plot_flag:
        plt.plot(history, label="loss, beta=%f"%beta)
        plt.legend()
        plt.show()    
    #w, sigma, E, theta, probs = w.cpu(), sigma.cpu(), E.cpu(), theta.cpu(), probs.cpu()    
    w,sigma,E,theta,probs,cost_vec,history = w.detach(),sigma.detach(),E.detach(),theta.detach(),probs.detach(),cost_vec,history
    
    return w,sigma,E,theta,probs,cost_vec,history    
# ---------------------------------------  
def plot_cost_vec1(cost_vec):
    y = cost_vec.mean(axis=1)
    y_min = cost_vec.min(axis=1)
    y_max = cost_vec.max(axis=1)
    x = np.arange(cost_vec.shape[0])
    noise = -cost_vec.std(axis=1)
    plt.plot(x, y)
    plt.fill_between(x, y_min, y_max, alpha = .5, color='green')
    plt.fill_between(x, y + noise, y - noise, alpha = .7, color='red')

def plot_cost_vec2(cost_vec):
    plt.plot(cost_vec.mean(axis=1)+cost_vec.std(axis=1))
    plt.plot(cost_vec.mean(axis=1)-cost_vec.std(axis=1))
    plt.plot(cost_vec.mean(axis=1))   

def plot_graphs(E,w, name, save_flag=False):
    g1 = get_graph_from_incidence(E)
    g2 = get_graph_from_incidence(E,w)    
    pos = nx.kamada_kawai_layout(g1)      
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle('Original and Sprasified graphs - %s'%name)        
    plt.axis('off')
    nx.draw(g1,pos,ax=ax1)    
    nx.draw(g2,pos,ax=ax2)
    if save_flag: fig.savefig('graphs_%s.pdf'%name, dpi=fig.dpi, pad_inches=0, bbox_inches='tight')    