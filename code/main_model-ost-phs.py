#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 12:38:15 2023

@author: Yu

Explore the model with oscillated input/output + phase changes.
"""
#%%
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# solving odes
import scipy.integrate as spi
import pylab as pl

#%%
###############################################################################
#                       Networks construction
###############################################################################
# Cycle graphs
G = nx.cycle_graph(5)
# G = nx.cycle_graph(6)
# G.add_edge(3,5)
# G = nx.cycle_graph(7)
N = G.number_of_nodes()
print("{} nodes with {} edges".format(N, G.number_of_edges()))
gtype = 'cycle-n{}'.format(N)

# relabel nodes
dict_i2n = {n:n+1 for n in G.nodes()}
# H = nx.relabel_nodes(G, mapping)

# visualisation
pos = nx.spring_layout(G)
nx.draw_networkx(G, pos=pos, with_labels=False)
nx.draw_networkx_labels(G, pos=pos, labels=dict_i2n)
print("Nodes:", G.nodes())

#%%
# Choose source and sink
souri = [0]
sinki = [2]

print("Sources:", [dict_i2n[i] for i in souri])
print("Sinks:", [dict_i2n[i] for i in sinki])

#%%
# Plot the network
# where source and sink has different colours
node_size = 500
font_size = 16
lw = 1.5

# nodes
chigh = ['b', 'y', 'r']
nodec = []
for i in range(N):
    if i in souri:
        nodec.append(chigh[0])
    elif i in sinki:
        nodec.append(chigh[2])
    else:
        nodec.append(chigh[1])

plt.figure(figsize=(4.5, 3))
# pos = nx.spring_layout(G)
nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color=nodec, alpha=0.6)
nx.draw_networkx_labels(G, pos, labels=dict_i2n, font_size=font_size, font_family='STIXGeneral')   
nx.draw_networkx_edges(G, pos, edge_color='grey', width=lw, alpha=0.8)

plt.grid(False)
plt.box(False)
# plt.savefig("{}_Graph.pdf".format(gtype), dpi=300, bbox_inches='tight')
plt.show()

#%%
# shortest paths and others
path_sesti = [(0,1), (1,2)]

#%%
# Network characteristics
# Adjacency matrix
A = nx.to_numpy_array(G)
W = A.copy()

# Distances
dist = 10. # Distance for other edges
dist_s = 10. # Distance for the edges in the path of the least #edges
if dist_s != dist:
    gtype += '-l{}'.format(int(dist_s))
else:
    gtype += '-uni'
print("Graph name:", gtype)

L = W.copy()
L[L > 0] = 1/L[L > 0]
L = L*dist
# reduce the length for some edges to dist_s
# edges = [(13,18)]
# edges = [(0,1), (1,6), (6,7), (7,12), (12,13), (13,18), (18,19), (19,24)]
# edges = [(0,4), (3,4)]
edges = path_sesti.copy()
for u,v in edges:
    L[u,v] = dist_s
    L[v,u] = dist_s

# Conductivity
D = A.copy()*1.0

plt.imshow(L, cmap='Blues')
plt.colorbar()
plt.title("Distance matrix")
plt.show()

#%%
###############################################################################
#                           Dynamics
###############################################################################
### initialisation ###
# input rate
at = 1.
# output percentage rate
bt = .5
# reinforcement strength (of conductancy)
q = .1
# decaying rate (of conductancy)
lam = .01

#%%
### Continuous ###
# Initialisation
import math

# amplitude
# a = 2.
theta = [0.01, 0.1][0]
magr = [0.5, 1.][1]

# Oscillation 
def oscillation(t, rate):
  '''input to the source nodes'''
  # return rate*(np.sin(2*math.pi*f*t + math.pi/2) + 1)
  return rate*(np.sin(2*math.pi*theta*t) + 1)

N = G.number_of_nodes()
x0 = 1.0*np.ones(N);
D0 = 1.0*A.copy()
ND=MaxTime=10000.0;
TS=0.01

INPUT=np.hstack((x0,D0.flatten()))

#%%
# Differential equations - for scipy.integrate.odeint
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']

# difference in phase = 2*pi*f*t_lag
phs = np.arange(-1, 1.05, 0.05)
print("phases/pi considered:", phs)
res_min = []
res_mid = []
res_max = []
T_res = int(100/TS) # record the min/mas of the last T_res steps

# record the results for specific phase differences
phs_rec = [-1., -0.5, 0., 0.5, 1.]
RESs = []

for ph in phs:
    t_lag = ph/(2*theta)
    print("current phase/pi:", ph)
    
    res_min.append([])
    res_max.append([])
    res_mid.append([])
    
    def diff_eqs(INP,t):  
        '''The main set of equations'''
        Y = np.zeros((N + N*N))
        V = INP   
        for i in range(N):
            if i in souri:
                Y[i] = oscillation(t, at) - bt*V[i]
            elif i in sinki:
                Y[i] = oscillation(t+t_lag, at*magr) - bt*V[i]
            else: 
                Y[i] = 0
            
            for j in range(N):
                if L[j,i] > 0:
                    # ODE for the state value
                    Y[i] += (V[j] - V[i])/L[j,i]*V[(j+1)*N+i] # from each neighbours
                    # ODE for the conductivity
                    Y[(j+1)*N+i] = q*abs(V[i] - V[j])/L[j,i]*V[(j+1)*N+i] - lam*V[(j+1)*N+i] # conductancy D[j,i]
                    # Y[(j+1)*N+i] = q*(V[j] - V[i])/L[j,i]*V[(j+1)*N+i] - lam*V[(j+1)*N+i]
                else:
                    Y[(j+1)*N+i] = 0
        return Y
    
    t_start = 0.0; t_end = ND; t_inc = TS
    t_range = np.arange(t_start, t_end+t_inc, t_inc)
    RES = spi.odeint(diff_eqs,INPUT,t_range)

    # Record the results
    for i in range(N):
        for j in range(i+1, N):
            res_min[-1].append(min(RES[-T_res:,j+N*(1+i)]))
            res_max[-1].append(max(RES[-T_res:,j+N*(1+i)]))
            res_mid[-1].append((res_min[-1][-1] + res_max[-1][-1])/2)
    
    if round(ph,2) in phs_rec:
        RESs.append(RES[-T_res:])
    
    # # Visualisation
    # pl.figure(figsize=(24, 16))
    
    # # Visualisation - the state value
    # pl.subplot(411)
    # for i in range(N):
    #   if i in souri:
    #       pl.plot(t_range, oscillation(t_range, at) - bt*RES[:,i], color='b', label='initial source {}'.format(dict_i2n[i]), alpha=0.8)
    #   elif i in sinki:
    #       pl.plot(t_range, oscillation(t_range+t_lag, at) - bt*RES[:,i], color='r', label='initial sink {}'.format(dict_i2n[i]), alpha=0.8)
    # pl.xlabel('Time')
    # pl.ylabel('Input vs Output')
    # pl.legend(loc='upper right')
    
    # pl.subplot(412)
    # for i in range(N):
    #   if i in souri:
    #       pl.plot(t_range, RES[:,i], color='b', label='node {}'.format(dict_i2n[i]), alpha=0.8)
    #   elif i in sinki:
    #       pl.plot(t_range, RES[:,i], color='r', label='node {}'.format(dict_i2n[i]), alpha=0.8)
    #   else:
    #  	    pl.plot(t_range, RES[:,i], color=colors[i], linestyle='dashed', label='node {}'.format(dict_i2n[i]), alpha=0.8)
    # pl.xlabel('Time')
    # pl.ylabel('# Particles')
    # pl.legend(loc='upper right')
    
    # # Visualisation - the conductivity of the edges in the shortest paths
    # pl.subplot(413)
    # for i in range(N):
    #     for j in range(i+1, N):
    #         if (i,j) in path_sesti:
    #             pl.plot(t_range, RES[:,j+N*(1+i)], color=colors[i], label='({},{})'.format(dict_i2n[i],dict_i2n[j]), alpha=0.6)
    #       #plot steady states
    #       # pl.hlines(y=D_steady[node, i], xmin=0, xmax=t_range[-1], color=colors[i], linestyles='--', label='({},{})'.format(node,i))
    # pl.ylabel('Conductivity')
    # pl.xlabel('Time')
    # pl.legend(loc='upper left')

    # # Visualisation - the conductivity of the edges that are not in the shortest path
    # pl.subplot(414)
    # for i in range(N):
    #     for j in range(i+1, N):
    #         if (L[i,j]>0) and ((i,j) not in path_sesti):
    #             pl.plot(t_range, RES[:,j+N*(1+i)], color=colors[i], label='({},{})'.format(dict_i2n[i],dict_i2n[j]), alpha=0.6)
    #       #plot steady states
    #       # pl.hlines(y=D_steady[node, i], xmin=0, xmax=t_range[-1], color=colors[i], linestyles='--', label='({},{})'.format(node,i))
    # pl.ylabel('Conductivity')
    # pl.xlabel('Time')
    # pl.legend(loc='upper left')
    # pl.savefig(gtype+"_continuous-1ost-p2-diff{:.2f}pi.pdf".format(round(ph,2)), dpi=200, bbox_inches='tight')
    # pl.show()

#%%
# plot
res_min = np.array(res_min)
res_max = np.array(res_max)
res_mid = np.array(res_mid)

#%%
# Visualisation
colors = ['tab:blue', 'tab:red', 'tab:green', 'tab:orange', 'tab:purple', 'tab:orange']

# plot for the shortest path
plt.figure(figsize=(6, 3.8))
idx = 0
for i in range(N):
    for j in range(i+1, N):
        if (i,j) in path_sesti:
            plt.plot(phs, res_mid[:,idx], marker='*', linestyle='dashed',
                     color=colors[i], label='({},{})'.format(dict_i2n[i],dict_i2n[j]))
            plt.fill_between(phs, res_min[:,idx], res_max[:,idx],
                             color=colors[i], alpha=0.5)
        idx += 1
plt.xlabel("Phase difference"+r"$/\pi$")
plt.ylabel("Conductivity")
plt.ylim([-0.1, 1.38])
plt.grid()
plt.legend()
plt.savefig(gtype+"_ost-pis-sp.pdf", dpi=300, bbox_inches='tight')
plt.show()    

# plot for other edges
plt.figure(figsize=(6, 3.8))
idx = 0
for i in range(N):
    for j in range(i+1, N):
        if (L[i,j]>0) and ((i,j) not in path_sesti):
            plt.plot(phs, res_mid[:,idx], marker='*', linestyle='dashed',
                     color=colors[i+2], label='({},{})'.format(dict_i2n[i],dict_i2n[j]), alpha=0.8)
            plt.fill_between(phs, res_min[:,idx], res_max[:,idx],
                             color=colors[i+2], alpha=0.5)
        idx += 1
plt.xlabel("Phase difference"+r"$/\pi$")
plt.ylabel("Conductivity")
plt.ylim([-0.1, 1.38])
plt.grid()
plt.legend()
plt.savefig(gtype+"_ost-pis-nsp.pdf", dpi=300, bbox_inches='tight')
plt.show() 

#%%
# plot in plot
# plot for the shortest path
colors = ['tab:blue', 'tab:red', 'tab:green', 'tab:orange', 'tab:purple']

fig, ax = plt.subplots(1,1,figsize=(8.5, 3.8))
ins1 = ax.inset_axes([0.86,0.65,0.12,0.18])
ins2 = ax.inset_axes([0.05,0.65,0.12,0.18])
ins3 = ax.inset_axes([0.45,0.5,0.12,0.18])
ins4 = ax.inset_axes([0.1,0.25,0.12,0.18])
ins5 = ax.inset_axes([0.81,0.25,0.12,0.18])

idx = 0
for i in range(N):
    for j in range(i+1, N):
        if (i,j) in path_sesti:
            ax.plot(phs, res_mid[:,idx], marker='*', linestyle='dashed',
                     color=colors[i], label='({},{})'.format(dict_i2n[i],dict_i2n[j]))
            ax.fill_between(phs, res_min[:,idx], res_max[:,idx],
                             color=colors[i], alpha=0.5)
            
            # \phi/\pi=1
            ins1.plot(list(t_range)[-T_res:], RESs[4][-T_res:,j+N*(1+i)], color=colors[i], alpha=0.6)
            ins1.set_xticks([])
            ins1.set_title(r"$\phi/\pi=1$")
            
            # \phi\pi=-1
            ins2.plot(list(t_range)[-T_res:], RESs[0][-T_res:,j+N*(1+i)], color=colors[i], alpha=0.6)
            ins2.set_xticks([])
            ins2.set_title(r"$\phi/\pi=-1$")
            
            # \phi/\pi=0
            ins3.plot(list(t_range)[-T_res:], RESs[2][-T_res:,j+N*(1+i)], color=colors[i], alpha=0.6)
            ins3.set_xticks([])
            ins3.set_title(r"$\phi/\pi=0$")
            
            # \phi\pi=-0.5
            ins4.plot(list(t_range)[-T_res:], RESs[1][-T_res:,j+N*(1+i)], color=colors[i], alpha=0.6)
            ins4.set_xticks([])
            ins4.set_title(r"$\phi/\pi=-0.5$")
            
            # \phi/\pi=0.5
            ins5.plot(list(t_range)[-T_res:], RESs[3][-T_res:,j+N*(1+i)], color=colors[i], alpha=0.6)
            ins5.set_xticks([])
            ins5.set_title(r"$\phi/\pi=0.5$")
            
            
        idx += 1
plt.xlabel("Phase difference"+r"$/\pi$")
plt.ylabel("Conductivity")
plt.ylim([-0.1, 1.38])
plt.xlim([-1.55, 1.55])
plt.grid()
plt.legend(loc='upper center')
plt.savefig(gtype+"_ost-pis-sp_more.pdf", dpi=300, bbox_inches='tight')
plt.show()

#%%
# plot for edges not in the shortest path
colors = ['tab:purple', 'tab:red', 'tab:green', 'tab:orange']

fig, ax = plt.subplots(1,1,figsize=(8.5, 3.8))
ins1 = ax.inset_axes([0.876,0.17,0.12,0.18])
ins2 = ax.inset_axes([0.062,0.17,0.12,0.18])
ins3 = ax.inset_axes([0.45,0.48,0.12,0.18])
ins4 = ax.inset_axes([0.25,0.4,0.12,0.18])
ins5 = ax.inset_axes([0.65,0.4,0.12,0.18])

idx = 0
for i in range(N):
    for j in range(i+1, N):
        if (L[i,j]>0) and ((i,j) not in path_sesti):
            ax.plot(phs, res_mid[:,idx], marker='*', linestyle='dashed',
                     color=colors[i], label='({},{})'.format(dict_i2n[i],dict_i2n[j]))
            ax.fill_between(phs, res_min[:,idx], res_max[:,idx],
                             color=colors[i], alpha=0.5)
            
            # \phi/\pi=1
            ins1.plot(list(t_range)[-T_res:], RESs[4][-T_res:,j+N*(1+i)], color=colors[i], alpha=0.6)
            ins1.set_xticks([])
            ins1.set_title(r"$\phi/\pi=1$")
            
            # \phi/\pi=-1
            ins2.plot(list(t_range)[-T_res:], RESs[0][-T_res:,j+N*(1+i)], color=colors[i], alpha=0.6)
            ins2.set_xticks([])
            ins2.set_title(r"$\phi/\pi=-1$")
            
            # \phi/\pi=0
            ins3.plot(list(t_range)[-T_res:], RESs[2][-T_res:,j+N*(1+i)], color=colors[i], alpha=0.6)
            ins3.set_xticks([])
            ins3.set_title(r"$\phi/\pi=0$")
            
            # \phi/\pi=-0.5
            ins4.plot(list(t_range)[-T_res:], RESs[1][-T_res:,j+N*(1+i)], color=colors[i], alpha=0.6)
            ins4.set_xticks([])
            ins4.set_title(r"$\phi/\pi=-0.5$")
            
            # \phi/\pi=0.5
            ins5.plot(list(t_range)[-T_res:], RESs[3][-T_res:,j+N*(1+i)], color=colors[i], alpha=0.6)
            ins5.set_xticks([])
            ins5.set_title(r"$\phi/\pi=0.5$")
            
            
        idx += 1
plt.xlabel("Phase difference"+r"$/\pi$")
plt.ylabel("Conductivity")
plt.ylim([-0.1, 1.38])
plt.xlim([-1.55, 1.55])
plt.grid()
plt.legend(loc='upper center')
plt.savefig(gtype+"_ost-pis-nsp_more.pdf", dpi=300, bbox_inches='tight')
plt.show()
