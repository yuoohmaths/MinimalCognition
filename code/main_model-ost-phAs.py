#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Yu Tian

This is to explore the model on cycle graphs: 
   (i) two oscillatory nodes (1 and 3);
   (ii) phase difference between -\pi and \pi;
   (iii) amplitude ratio between 1/10 and 10.
"""
#%%
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# solving odes
import scipy.integrate as spi

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

# Frequency
theta = [0.01, 0.1][0]

# Oscillation 
def oscillation(t, rate):
  '''input to the source nodes'''
  return rate*(np.sin(2*math.pi*theta*t) + 1)

N = G.number_of_nodes()
x0 = 1.0*np.ones(N);
D0 = 1.0*A.copy()
ND=MaxTime=10000.0;
TS=0.01

INPUT=np.hstack((x0,D0.flatten()))
    
#%%
###### Table preparation ######
# Differential equations
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
# draw the graph
figsize = (1.3, 1.2)
node_size = 100
font_size = 8
lw = [1.5, 0.1][1]
fac = 0.8
tol = 1e-8

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

# contract the position
pos_new = pos.copy()
for i,val in pos.items():
    pos_new[i] = val*fac

### Parameters ###
# Amplitude ratios
magrs = [0.1, 1/3., 0.5, 0.8, 0.9, 1., 10/9., 10/8., 2., 3., 5., 10.]

# difference in phase = 2*pi*f*t_lag
# phs = np.arange(-1, 1.05, 0.05)
phs = np.arange(-0.75, 1.05, 0.25)
print("phases/pi considered:", phs)
T_res = int(100/TS) # record the min/mas of the last T_res steps


for magr in magrs:
    if magr >= 10:
        magr_lab = str(magr)[:2]+'-'+str(magr)[3:5]
    else:
        magr_lab = str(magr)[0]+'-'+str(magr)[2:5]
    print("Amplitude ratio:", magr, magr_lab)
    
    res_min = []
    res_mid = []
    res_max = []
    
    for ph in phs:
        t_lag = ph/(2*theta)
        if ph > -1e-4:
            ph_lab = str(ph)[0]+'-'+str(ph)[2:4]
        else:
            ph_lab = 'n'+str(ph)[1]+'-'+str(ph)[3:5]
        print("current phase_diff/pi:", ph, ph_lab)
        mp_lab = 'A'+ magr_lab + '-p' + ph_lab
        print("current comb:", magr, ph, mp_lab)
        
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
        # RES = spi.odeint(diff_eqs,INPUT,t_range)
        RES = spi.solve_ivp(diff_eqs, [t_start, t_end], INPUT, method='RK45', t_eval=t_range)
        
        # Record the results
        for i in range(N):
            # for j in range(i+1, N):
            for j in range(N):
                res_min[-1].append(min(RES.y[j+N*(1+i),-T_res:]))
                res_max[-1].append(max(RES.y[j+N*(1+i),-T_res:]))
                res_mid[-1].append((res_min[-1][-1] + res_max[-1][-1])/2)
                
                 
        # Draw the graph
        plt.figure(figsize=figsize)
        # pos = nx.spring_layout(G)
        # linewitdth
        lw_list = []
        for i,j in G.edges():
            if res_mid[-1][i*N+j] > tol:
                lw_list.append(lw*res_mid[-1][i*N+j])
            else:
                lw_list.append(0)
        nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color=nodec, alpha=0.6)
        nx.draw_networkx_labels(G, pos, labels=dict_i2n, font_size=font_size, font_family='STIXGeneral')   
        nx.draw_networkx_edges(G, pos, edge_color='k', width=lw_list, alpha=0.8)
        
        plt.grid(False)
        plt.box(False)
        plt.savefig("{}_res-G-{}.png".format(gtype,mp_lab), dpi=300, bbox_inches='tight')
        plt.show()   
