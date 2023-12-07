#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 23:18:32 2023

@author: Yu

Construct irregular hexagon tiling
"""
#%%
import math
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

# solving odes
import scipy.integrate as spi

import random
import time
import tracemalloc

path = ''
# path = '/n/home13/yuutian/PO-GraphCurvature/results/others'
#%%
# def hexagon_generator(edge_length, offset):
#   """Generator for coordinates in a hexagon."""
#   x, y = offset
#   for angle in range(0, 360, 60):
#     x += math.cos(math.radians(angle)) * edge_length
#     y += math.sin(math.radians(angle)) * edge_length
#     yield x, y

# #%%
# edge_length = 0.1
# col_width = edge_length * 3
# row_height = math.sin(math.pi / 3) * edge_length

# for row in range(7):
#     for col in range(2):
#         x = (col + 0.5 * (row % 2)) * col_width
#         y = row * row_height
#         for angle in range(0, 360, 60):
#           x += math.cos(math.radians(angle)) * edge_length
#           y += math.sin(math.radians(angle)) * edge_length
          
#%%
row = 7
col = 7
G = nx.hexagonal_lattice_graph(row, col)
N = G.number_of_nodes()
print("#nodes:", N, "#edges:", G.number_of_edges())

# relabel
mapping = {n:i for i,n in enumerate(sorted(G.nodes()))}
G = nx.relabel_nodes(G, mapping)

pos = nx.get_node_attributes(G, 'pos')
nx.draw_networkx(G, pos=pos, node_size=100, with_labels=True)

#%%
# choose 1/3 nodes to violate the position
nodes_mod = [0, 17, 4, 21, 8, 25, 12, 29, 46, 61,
             43, 57, 39, 53, 35, 49, 31, 64, 81, 67,
             85, 71, 89, 75, 93, 110, 125, 107, 121, 103,
             117, 99, 113, 95]

mu, sigma = 0, 0.2 # mean and standard deviation
pos_new = pos.copy()

np.random.seed(0)
for i in nodes_mod:
    posi = (pos[i][0]-np.random.normal(mu, sigma), pos[i][1]+np.random.normal(mu, sigma))
    pos_new[i] = posi


nx.draw_networkx(G, pos=pos_new, node_size=50, with_labels=False)

#%%
# modifications
pos_new[39] = (pos_new[39][0], 
               pos_new[39][1]+(pos_new[40][1]-pos_new[39][1])*0.2)

nx.draw_networkx(G, pos=pos_new, node_size=50, with_labels=False)
#%%
# save the pos
nx.set_node_attributes(G, pos_new, "pos")
# save the weights
# Update the edge weights
dist_fac = 10
dist_dig = 1

edges_dist = {}
edges_weis = {}
for u,v in G.edges():
    edges_dist[(u,v)] = round(np.linalg.norm(np.array(pos_new[u]) - np.array(pos_new[v]))*dist_fac, dist_dig)
    edges_weis[(u,v)] = round(1/edges_dist[(u,v)], dist_dig+1)

nx.set_edge_attributes(G, edges_dist, "distance")
nx.set_edge_attributes(G, edges_weis, "weight")

# nx.write_gpickle(G, "hexagon-{}-{}.gpickle".format(row, col))
nx.write_gml(G, "hexagon-{}-{}.gpickle".format(row, col))

#%%
# read in graph
row = 7
col = 7
gtype = "hexagon-{}-{}".format(row, col)
# G = nx.read_gpickle("hexagon-{}-{}.gpickle".format(row, col))
G = nx.read_gml("hexagon-{}-{}.gpickle".format(row, col))
pos = nx.get_node_attributes(G, 'pos')
nx.draw_networkx(G, pos=pos, node_size=100)

N = G.number_of_nodes()
edges_weis = nx.get_edge_attributes(G, "weight")

# relabel
mapping = {n:int(n) for n in sorted(G.nodes())} # note the labels are str
G = nx.relabel_nodes(G, mapping)
pos = nx.get_node_attributes(G, 'pos')
# relabel nodes
dict_i2n = {n:n+1 for n in G.nodes()}

#%%
# Choose oscilators
# No = 3
# random.seed(0)
# oscii = sorted(random.sample(list(G.nodes()), k=No))
# print(oscii)
# oscii = [56, 6] # no = 2, for N = 100+ hexagons
# oscii = [56, 6, 117] # no = 3, for N = 100
oscii = [56, 6, 117, 46, 48] # no = 5, for N = 100


# oscii = [10, 50, 11, 56, 85, 67, 88, 70, 54, 36]
# oscii = [1, 14, 41, 63, 64, 75, 82, 93, 95, 97, 101, 103, 142, 143, 145, 150, 
#          195, 209, 223, 227, 244, 249, 256, 265, 266, 288, 310, 317, 323, 327, 
#          333, 338, 341, 362, 366, 394, 408, 414, 430, 444, 453, 458, 483, 488, 
#          497, 505, 516, 520, 523, 533, 545, 556, 561, 565, 573, 581, 597, 616, 
#          625, 626, 633, 640, 655, 684, 700, 720, 722, 727, 736, 747, 773, 776, 
#          802, 803, 818, 822, 824, 829, 844, 847, 849, 860, 864, 870, 886, 888, 
#          891, 896, 909, 911, 913, 920, 923, 929, 931, 934, 940, 960, 984, 988] # no=100
No = len(oscii)

print("Oscillators:", [dict_i2n[i] for i in oscii])
dict_o2i = {n:i for i,n in enumerate(oscii)}

#%%
# Plot the network
# where source and sink has different colours
node_size = 200
font_size = 13
# lw = .5
lw_fac = 5

# nodes
chigh = ['b', 'y', 'r', 'cyan', 'skyblue']
nodec = []
for i in range(N):
    # if No == 2:
    #     if i in oscii:
    #         nodec.append(chigh[2])
    #     else:
    #         nodec.append(chigh[1])
    # elif No == 3:
    #     if i in oscii2:
    #         nodec.append(chigh[2])
    #     elif i in oscii:
    #         nodec.append(chigh[0])
    #     else:
    #         nodec.append(chigh[1])
    # elif No == 5:
    #     if i in oscii2:
    #         nodec.append(chigh[2])
    #     elif i in oscii3:
    #         nodec.append(chigh[0])
    #     elif i in oscii:
    #         nodec.append(chigh[3])
    #     else:
    #         nodec.append(chigh[1])
    # elif No == 10:
    #     if i in oscii2:
    #         nodec.append(chigh[2])
    #     elif i in oscii3:
    #         nodec.append(chigh[0])
    #     elif i in oscii5:
    #         nodec.append(chigh[3])
    #     elif i in oscii:
    #         nodec.append(chigh[4])
    #     else:
    #         nodec.append(chigh[1])
    if i == oscii[0]:
        nodec.append(chigh[2])
    elif i in oscii[1:]:
        nodec.append(chigh[0])
    else:
        nodec.append(chigh[1])

plt.figure(figsize=(9, 6))
# pos = nx.spring_layout(G)
nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color=nodec, alpha=0.6)
nx.draw_networkx_labels(G, pos, labels=dict_i2n, font_size=font_size, font_family='STIXGeneral')   
# nx.draw_networkx_edges(G, pos, edge_color='grey', width=lw, alpha=0.8)
# edge weights inversely correlated with the distance
weights = np.array(list(edges_weis.values()))*lw_fac
nx.draw_networkx_edges(G, pos, edge_color='grey', width=weights, alpha=0.8)

plt.grid(False)
plt.box(False)
plt.savefig(path+"{}_Graph-No{}.pdf".format(gtype, No), dpi=300, bbox_inches='tight')
plt.show()

#%%
W = nx.to_numpy_array(G, weight='weight')
A = W.copy()
A[A>0] = 1

# distance matrix
L = nx.to_numpy_array(G, weight='distance')

# Conductivity
D = A.copy()*1.0

plt.imshow(L, cmap='Blues')
plt.colorbar()
plt.title("Distance matrix")
plt.show()

plt.imshow(D, cmap='Blues')
plt.colorbar()
plt.title("Conductivity matrix")
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
# import math

# amplitude
# a = 2.
theta = 0.001

# phases
# phs = np.arange(-1, 1.05, 0.05)
phs = [-1., -0.95, -0.9, -0.85, -0.8, -0.75, -0.7, -0.65, -0.6, -0.55, -0.5,
       -0.45, -0.4, -0.35, -0.3, -0.25, -0.2, -0.15, -0.1, -0.05, 0., 
       0.05,  0.1,  0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 
       0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.]
print("phases/pi considered:", phs)
# phases = random.choices(phs, k=No)
# phases[0] = 0.
# print(phases)
# phases = [0., 1.] # No = 2
# phases = [0., 1., 1.] # No = 3
phases = [0., 1., 1., 1., 1.] # No = 5
# phases = [0., 1., 1., 0., 0.] # No = 5
# phases = [0., 1., 1., 1., 0.] # No = 5
# phases = [0., 1., 1., .5, 0.] # No = 5
# phases = [0., .55, -0.35, 0.1, -0.75] # No = 5 random
# phases = [0., 1., 0., 1., 1., 0., 1., 0., 1., 1.] # No = 10
# phases = [0.0, 0.8, -0.6, -0.75, -0.75, -0.85, 0.6, 1.0, 0.05, 0.4] # random
# phases = [0.0, 0.7, 0.95, 0.2, 0.0, 0.3, -0.1, -0.15, 0.75, -0.65, -0.25, -0.45, 
#           -0.75, -0.45, -0.35, -0.25, -0.85, 0.55, -0.95, -0.45, -0.7, 0.15, 
#           -0.3, 1.0, -0.75, -0.8, 0.45, 0.9, -0.4, 0.75, 0.45, -0.4, 0.7, -0.4, 
#           -0.8, 0.1, -0.05, -0.35, 0.6, 0.65, -0.75, -0.8, 0.0, -0.95, -0.35, 
#           0.4, -0.7, -0.7, 0.15, 0.65, 1.0, -0.85, 0.65, -0.6, -0.55, -0.25, 
#           -0.8, 0.1, -0.45, 0.85, 0.6, 0.45, -0.6, -0.85, -0.55, 1.0, 0.15, 
#           -0.7, -0.65, -0.8, 0.65, 0.7, -0.95, 0.05, 0.5, -0.6, 0.0, -0.5, 0.3, 
#           1.0, 0.6, 0.25, -0.8, -0.1, -0.3, 0.65, -0.8, 0.2, 0.0, 0.15, 0.3, 
#           -0.15, 0.05, 0.35, 0.8, 0.7, 1.0, 0.3, 0.75, 0.1] # No = 100
phases = np.array(phases)
t_lags = phases/(2*theta)

# Amplitude
magrs = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
         1., 
         1.11, 1.25, 1.43, 1.67, 2., 2.5, 3., 3.33, 5.]
print("Amplitude ratio considered:", phs)
# amprs = random.choices(magrs, k=No)
# amprs[0] = 1.
# amprs = [5., 5.] # No = 2
# amprs = [5., 5., 5.] # No = 3
amprs = [5., 5., 5., 5., 5.] # No = 5
# amprs = [2., 2., 2., 2., 2., 2., 2., 2., 2., 2.] # No = 10
# amprs = [1.0, 0.2, 1.67, 2.0, 0.7, 0.7, 1.11, 3.0, 1.25, 0.4, 0.8, 0.7, 1.67, 
#          1.0, 2.5, 1.11, 3.33, 0.3, 0.4, 0.7, 2.0, 1.0, 3.0, 2.5, 3.0, 1.0, 0.8, 
#          0.8, 1.11, 0.2, 0.3, 2.5, 1.11, 1.25, 0.3, 0.9, 2.0, 0.7, 3.33, 0.9, 
#          5.0, 3.0, 5.0, 1.25, 0.6, 3.0, 0.2, 3.0, 2.0, 0.7, 1.25, 0.7, 1.0, 0.6, 
#          2.5, 2.0, 0.8, 0.8, 0.4, 3.33, 0.6, 2.0, 0.8, 3.33, 1.0, 0.4, 3.33, 
#          1.25, 0.3, 1.67, 2.5, 0.8, 0.2, 1.0, 1.0, 0.3, 2.5, 0.2, 1.25, 0.4, 
#          0.7, 1.67, 1.43, 0.5, 2.5, 2.0, 0.3, 0.3, 1.67, 1.43, 3.33, 1.43, 5.0, 
#          1.67, 0.7, 3.0, 5.0, 0.8, 0.7, 2.0] # No = 100

# Oscillation 
def oscillation(t, rate):
  '''input to the source nodes'''
  # return rate*(np.sin(2*math.pi*f*t + math.pi/2) + 1)
  return rate*(np.sin(2*math.pi*theta*t) + 1)

N = G.number_of_nodes()
x0 = 1.0*np.ones(N);
D0 = 1.0*A.copy()
ND=MaxTime=5000.0;
TS=0.01

INPUT=np.hstack((x0,D0.flatten()))

#%%
# starting the monitoring
tracemalloc.start()
t_s = time.time()
# Differential equations - for scipy.integrate.odeint
# def diff_eqs(INP,t):  
def diff_eqs(t, INP):
    '''The main set of equations'''
    Y = np.zeros((N + N*N))
    V = INP   
    for i in range(N):
        if i in oscii:
            Y[i] = oscillation(t+t_lags[dict_o2i[i]], at*amprs[dict_o2i[i]]) - bt*V[i]
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

print("time for solving ODEs:", time.time() - t_s)
# displaying the memory
print(tracemalloc.get_traced_memory())
 
# stopping the library
tracemalloc.stop()

#%%
# snapshot at stationary
# Record the results
figsize = (9, 6)
T_res = int(100/TS)
tol = 1e-8

res_min = []
res_max = []
res_mid = []
for i in range(N):
    # for j in range(i+1, N):
    for j in range(N):
        # res_min.append(min(RES[-T_res:,j+N*(1+i)]))
        # res_max.append(max(RES[-T_res:,j+N*(1+i)]))
        res_min.append(min(RES.y[j+N*(1+i), -T_res:]))
        res_max.append(max(RES.y[j+N*(1+i), -T_res:]))
        res_mid.append((res_min[-1] + res_max[-1])/2)
        
#%%
# Diagnosis
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
          'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan', 'lime']
colors_io = ['r', 'b', 'y', 'tab:cyan','tab:purple','tab:orange', 'tab:green',
             'tab:brown', 'tab:pink', 'tab:gray']
figsize=(9., 3.)
fz = 16
plt.rcParams.update({'font.size': fz})

# Visualisation - the state value
plt.figure(figsize=figsize)
for i in range(No):
  plt.plot(t_range, oscillation(t_range+t_lags[i], at) - bt*RES.y[oscii[i], :], color=colors_io[i], label='node {}'.format(dict_i2n[oscii[i]]), alpha=0.8)
plt.xlabel('Time')
plt.ylabel('Input / Output')
# plt.legend(framealpha=0.3, loc='upper right', ncols=1, fontsize=12) # No=2,3
# plt.legend(framealpha=0.3, loc='upper right', ncols=3, fontsize=12) # No=5
plt.legend(framealpha=0.3, loc='upper right', ncols=4, fontsize=12) # No=10
plt.savefig(gtype+"-No{}_ost-phase_io.png".format(No), dpi=300, bbox_inches='tight')
plt.show()


plt.figure(figsize=figsize)
for i in range(No):
  plt.plot(t_range, RES.y[oscii[i],:], color=colors_io[i], label='node {}'.format(dict_i2n[oscii[i]]), alpha=0.8)

# for i in range(N):
#     if i in oscii:
#         plt.plot(t_range, RES.y[i, :], color=colors_io[i], label='node {}'.format(dict_i2n[i]), alpha=0.8)Adjacency
#     else:
#         plt.plot(t_range, RES.y[i :], color=colors[i], linestyle='dashed', label='node {}'.format(dict_i2n[i]), alpha=0.8)
plt.xlabel('Time')
plt.ylabel('# Particles')
# plt.legend(framealpha=0.3, loc='upper right', ncols=1, fontsize=12) # No=2,3
# plt.legend(framealpha=0.3, loc='upper right', ncols=3, fontsize=12) # No=5
plt.legend(framealpha=0.3, loc='upper right', ncols=4, fontsize=12) # No=10
plt.savefig(gtype+"-No{}_ost-phase_Nt.png".format(No), dpi=300, bbox_inches='tight')
plt.show()


# Visualisation - the conductivity of the edges in the shortest paths
# path_sesti = [(6,21), (21,22), (22,38), (38,39), (39,55), (55,56)] # No=2, N=100
path_sesti = [(56,72), (71,72), (71,87), (86,87), (86,102), (101,102), (101,117)] # No=3, N=100
# path_sesti = [(55,54), (54, 53), (53, 52), (52, 51), (51,50), (55,56), (56,57), (57,58), (58,59)] # No=3, N=100
# colors = ['tab:blue', 'tab:red']

plt.figure(figsize=figsize)
ic = 0
for i in range(N):
    for j in range(i+1, N):
        if (i,j) in path_sesti:
            plt.plot(t_range, RES.y[j+N*(1+i),:], color=colors[ic], label='({},{})'.format(dict_i2n[i],dict_i2n[j]), alpha=0.6)
            ic += 1
      #plot steady states
      # plt.hlines(y=D_steady[node, i], xmin=0, xmax=t_range[-1], color=colors[i], linestyles='--', label='({},{})'.format(node,i))
plt.ylabel('Conductivity')
plt.xlabel('Time')
# plt.legend(framealpha=0.3, loc='upper left', ncols=1, fontsize=12) # No=2
# plt.legend(framealpha=0.3, loc='upper left', ncols=3, fontsize=12) # No=3
plt.legend(framealpha=0.3, loc='upper left', ncols=4, fontsize=12) # No=5
plt.savefig(gtype+"-No{}_ost-phase_D-sp.png".format(No), dpi=300, bbox_inches='tight')
plt.show()
       

#%%  
# Draw the graph
# node_size = 300
# font_size = 16
tol = 0.1
lw = .5
lw_ratio = .5
lw_zero = .1
figsize=(9, 6)
chigh = ['b', 'y', 'r', 'cyan', 'skyblue']

# pos = nx.spring_layout(G)
# linewitdth
lw_list = []
for i,j in G.edges():
    if res_mid[i*N+j] > tol:
        lw_list.append(lw*res_mid[i*N+j])
    else:
        lw_list.append(lw_zero)
        
plt.figure(figsize=figsize)

# other nodes
nodes_others = [i for i in G.nodes() if i not in oscii]

nx.draw_networkx_nodes(G, pos, nodelist=nodes_others, node_size=node_size, node_color=chigh[1], alpha=0.6)
# nx.draw_networkx_nodes(G, pos, nodelist=oscii, node_size=node_size, node_color=phases, cmap='RdYlBu', alpha=0.6)
nx.draw_networkx_nodes(G, pos, nodelist=oscii, node_size=node_size, node_color=[-i for i in phases], cmap='coolwarm', alpha=0.6)
nx.draw_networkx_labels(G, pos, labels=dict_i2n, font_size=font_size, font_family='STIXGeneral')   
nx.draw_networkx_edges(G, pos, edge_color='k', width=np.array(lw_list)*lw_ratio, alpha=1.)

plt.grid(False)
plt.box(False)
plt.savefig(path+"{}_res-G-No{}.png".format(gtype, No), dpi=300, bbox_inches='tight')
plt.show()    

#%%
# Video
import matplotlib.animation as animation

num = 1000 # number of plots in a video
t_lag = int(MaxTime/TS/num)
print("# plots in a video:", num, "per", t_lag)

# the max value
D_res = np.zeros((N,N))
for i,j in G.edges():
    D_res[i,j] = RES.y[(i+1)*N+j, -1]
    D_res[j,i] = RES.y[(j+1)*N+i, -1]
edge_max = round(np.max(D_res)+0.1, 1)
print("max weight value:", edge_max)
node_max = round(np.max(RES.y[:N, :])+0.1, 1)
print("max node state:", node_max)

# node_size = 300
# font_size = 16
lw = 0.5
edge_color='grey'

edges = list(G.edges())

fig, ax = plt.subplots(figsize=figsize)

# Function - plot the network at a time point
def update(idx):
    ax.clear()
    time = idx*t_lag
    # print(idx, time)
    node_color = RES.y[:N, time]
    # edge_width = [RES.y[(u+1)*N+v, time]*lw/edge_max for u,v in edges]
    edge_width = [RES.y[(u+1)*N+v, time]*lw for u,v in edges]
    # nx.draw_networkx(G, pos=pos, with_labels=True, node_size=node_size, 
    #                 node_color=node_color, vmax=node_max, cmap='YlOrRd',
    #                 edge_color=edge_color, width=edge_width, font_size=font_size,
    #                 font_family='STIXGeneral', ax=ax)
    nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color=node_color, vmax=node_max, cmap='YlOrRd', ax=ax)
    # nx.draw_networkx_labels(G, pos, labels=dict_i2n, font_size=font_size, font_family='STIXGeneral', ax=ax)   
    nx.draw_networkx_edges(G, pos, edge_color=edge_color, width=edge_width, ax=ax)
    # ax.set_title("Continuous: "+r"$l_{0,4}, l_{3,4}$" + " = {}, ".format(int(dist_s))+ r"$l_{0,1}, l_{1,2}, l_{2,3}$"+" = {}".format(int(dist)))
    # plt.grid(False)
    plt.box(False)
    plt.tight_layout()
    # imshow the weight matrix
    # frames.append([plt.imshow(Ds[t, :, :], cmap='Blues', vmax=vmax, animated=True)])

# Make the video
ani = animation.FuncAnimation(fig, update, frames=num, interval=10, repeat=False)
# interval: Delay between frames in milliseconds
# animation.ArtistAnimation(fig, frames, interval=10, blit=True,
#                                 repeat_delay=1000)
# ani.save(path+gtype+'_net-No{}-nolab.mp4'.format(No))
fps = 10
FFwriter = animation.FFMpegWriter(fps=fps)
# fps: Movie frame rate (per second).
ani.save(path+gtype+'_net-No{}-nolabx{}.mp4'.format(No, int(fps*MaxTime/num)), writer = FFwriter)
plt.show()   

#%%
### Plot the graph at different time instances ###
# Choose the time to plot
rate = 0.0001
end = int(1500/TS)
for i in range(No):
    plt.plot(t_range[-end:], RES.y[oscii[i],-end:], color=colors_io[i], label='node {}'.format(dict_i2n[oscii[i]]), alpha=0.8)

    maxval = max(RES.y[oscii[i],-end:])
    minval = min(RES.y[oscii[i],-end:])
    print((np.where(RES.y[oscii[i],-end:] > maxval*(1-rate))[0] + (MaxTime/TS - end))*TS)
    print((np.where(RES.y[oscii[i],-end:] < minval*(1+rate))[0] + (MaxTime/TS - end))*TS)

#%%
# set the indices
indices = [3720, 4220, 4720] # No=3

#%%
# Plot
for idx in range(len(indices)):
    time = int(indices[idx]/TS)
    # print(idx, time)
    node_color = RES.y[:N, time]
    # edge_width = [RES.y[(u+1)*N+v, time]*lw/edge_max for u,v in edges]
    edge_width = [RES.y[(u+1)*N+v, time]*lw for u,v in edges]
    # nx.draw_networkx(G, pos=pos, with_labels=True, node_size=node_size, 
    #                 node_color=node_color, vmax=node_max, cmap='YlOrRd',
    #                 edge_color=edge_color, width=edge_width, font_size=font_size,
    #                 font_family='STIXGeneral', ax=ax)
    plt.figure(figsize=figsize)
    nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color=node_color, vmax=node_max, cmap='YlOrRd')
    # nx.draw_networkx_labels(G, pos, labels=dict_i2n, font_size=font_size, font_family='STIXGeneral', ax=ax)   
    nx.draw_networkx_edges(G, pos, edge_color=edge_color, width=edge_width)
    # ax.set_title("Continuous: "+r"$l_{0,4}, l_{3,4}$" + " = {}, ".format(int(dist_s))+ r"$l_{0,1}, l_{1,2}, l_{2,3}$"+" = {}".format(int(dist)))
    # plt.grid(False)
    plt.grid(False)
    plt.box(False)
    plt.savefig(path+"{}_res-G-No{}-t{}.png".format(gtype, No, idx), dpi=300, bbox_inches='tight')
    plt.show() 


#%%
# plot the graph
node_size = 200
font_size = 13
# lw = .5
lw_fac = 5

# nodes
chigh = ['b', 'y', 'r', 'cyan', 'skyblue']

plt.figure(figsize=(9, 6))
# pos = nx.spring_layout(G)
nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color=chigh[1], alpha=0.6)
nx.draw_networkx_labels(G, pos, labels=dict_i2n, font_size=font_size, font_family='STIXGeneral')   
# nx.draw_networkx_edges(G, pos, edge_color='grey', width=lw, alpha=0.8)
# edge weights inversely correlated with the distance
weights = np.array(list(edges_weis.values()))*lw_fac
nx.draw_networkx_edges(G, pos, edge_color='grey', width=weights, alpha=0.8)

plt.grid(False)
plt.box(False)
plt.savefig(path+"{}_Graph.pdf".format(gtype, No), dpi=300, bbox_inches='tight')
plt.show()











