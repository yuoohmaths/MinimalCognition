#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 15:59:19 2023

@author: lingy145
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modelling slime mould

@author: Yu Tian

This is to model the dynamics on cycle graphs, where we have one source and one
sink, and the input rate and the output rate is fixed (or not oscilated).
"""
#%%
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

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

# visualisation
pos = nx.spring_layout(G)
# pos = {0: np.array([-0.6 , -0.8]),
#  1: np.array([ 0.6, -0.8  ]),
#  2: np.array([0.97, 0.3]),
#  3: np.array([0., 1.        ]),
#  4: np.array([-0.97,  0.3])}
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
dist_para = 1.
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
###############################################################################
#                          Steady states
###############################################################################

D_steady = np.zeros((N,N))

#Check if graph is C_n

# Check which path is the shortest
all_paths = []
paths = nx.all_simple_paths(G, source=souri[0], target=sinki[0])
all_paths.extend(paths)

print("Paths between source and sink:", all_paths) #print all paths

path_length_forward = 0
path_length_backwards = 0

# compute the length of the path
for i in range(0, len(all_paths[0])):
        path_length_forward += L[all_paths[0][i]][all_paths[0][(i+1)%len(all_paths[0])]]

for i in range(0, len(all_paths[1])):
        path_length_backwards += L[all_paths[1][i]][all_paths[1][(i+1)%len(all_paths[1])]]

# update the steady states depending on which one is shortest        
if path_length_forward > path_length_backwards:
#    for i in range(0, len(all_paths[1])):
#        D_steady[all_paths[1][i]][all_paths[1][(i+1)%len(all_paths[1])]] = 0
#       D_steady[all_paths[1][(i+1)%len(all_paths[1])]][all_paths[1][i]] = 0
    for j in range(0, len(all_paths[1])):
        D_steady[all_paths[1][j]][all_paths[1][(j+1)%len(all_paths[1])]] = at*q/lam
        D_steady[all_paths[1][(j+1)%len(all_paths[1])]][all_paths[1][j]] = at*q/lam  

elif path_length_forward < path_length_backwards:
#    for j in range(0, len(all_paths[0])):
#       D_steady[all_paths[0][j]][all_paths[0][(j+1)%len(all_paths[0])]] = 0
#       D_steady[all_paths[0][(j+1)%len(all_paths[0])]][all_paths[0][j]] = 0
    for i in range(0, len(all_paths[0])):
        D_steady[all_paths[0][i]][all_paths[0][(i+1)%len(all_paths[0])]] = at*q/lam
        D_steady[all_paths[0][(i+1)%len(all_paths[0])]][all_paths[0][i]] = at*q/lam

# if they have the same length
else:    
    for j in range(0, len(all_paths[0])):
        D_steady[all_paths[0][j]][all_paths[0][(j+1)%len(all_paths[0])]] =  at*q/(lam*2)
        D_steady[all_paths[0][(j+1)%len(all_paths[0])]][all_paths[0][j]] =  at*q/(lam*2)
    for i in range(0, len(all_paths[1])):
        D_steady[all_paths[1][i]][all_paths[1][(i+1)%len(all_paths[1])]] = at*q/(lam*2)
        D_steady[all_paths[1][(i+1)%len(all_paths[1])]][all_paths[1][i]] = at*q/(lam*2)

#%%
### Continuous ###
# Initialisation
# Input rate
def source_input(t, rate):
  '''input to the source nodes'''
  return rate

# Output rate
def sink_output(t, rate):
  '''ouput from the sink nodes'''
  return rate

N = G.number_of_nodes()
#x0 = 3.0*np.ones(N);
x0 = np.random.uniform(0,10,N) # random initial conditions
# random initial conditions
init_D = np.random.uniform(0,4,(len(A),len(A)));
D0 = A.copy()*init_D;
ND=MaxTime=100000.0;
TS=0.01

INPUT=np.hstack((x0,D0.flatten()))

# Produce bifurcation plots

first_dists = np.arange(18, 22.02, 0.05) 


res_end = []
res_end_part = [] #number of particles

T_res = int(1/TS) # record the min/mas of the last T_res steps

num_iter = 10; #number of interations

for k in range(num_iter): 
   
    for l_12 in first_dists:
        x0 = np.random.uniform(0.1,2,N) # random initial conditions
        #init_D = np.random.uniform(0.5,1,(len(A),len(A)));
        init_D = np.random.uniform(8,12,(len(A),len(A)));
        init_D = np.maximum( init_D, init_D.transpose() )
        D0 = A.copy()*init_D;
        INPUT=np.hstack((x0,D0.flatten()))
        L[0,1]=L[1,0]=l_12
        print(k,L[0,1])
        
        res_end.append([])
        res_end_part.append([])
        #%%
        # Differential equations - for scipy.integrate.odeint
        def diff_eqs(t,INP):  
            '''The main set of equations'''
            Y = np.zeros((N + N*N))
            V = INP   
            for i in range(N):
                if i in souri:
                    Y[i] = source_input(t, at)
                elif i in sinki:
                    Y[i] = -sink_output(t, bt)*V[i]
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
        RES = spi.solve_ivp(diff_eqs,[t_start,t_end], INPUT, 
                            t_eval=np.linspace(t_end-5,t_end,100),
                            method ="Radau")
        assert RES.success,RES.message
        res_end[-1].append(RES.y)
        # for i in range(N):
        #     res_end_part[-1].append(RES.y[-T_res:,i])
        #     for j in range(i+1, N):
        #         res_end[-1].append(RES.y[-T_res:,j+N*(1+i)])
    

#%%
# plot
res_end = np.array(res_end)
res_end = res_end[:,0,:,:]     #remove dimension 1     
res_end_part = np.array(res_end_part)



colors = ['tab:blue', 'tab:red', 'tab:green', 'tab:orange', 'tab:purple', 'tab:orange']

# plot for the shortest path
#plt.figure(figsize=(6, 3.8))
#plt.figure(figsize=(4, 2.5), dpi=100)
figsize=(12., 4.)
fz = 16
plt.rcParams.update({'font.size': fz})

for k in range(num_iter):
    for i in range(N):
        plt.plot(first_dists, res_end[k*len(first_dists):(k+1)*len(first_dists),N+N*i+(i+1)%N], marker='.', linestyle='dashed',
                 color=colors[i], label='({},{})'.format(dict_i2n[i],dict_i2n[(i+1)%N]))
plt.xlabel("Length: $l_{12}$")
plt.ylabel("Conductivity: $D_{(i,j)}$")
plt.ylim([-0.01, 12.])
line_red = Line2D([0], [0], label='(1,2), (2,3)', marker='.', linestyle='dashed', color='tab:red')
line_purple = Line2D([0], [0], label='(3,4), (4,5), (5,1)', marker='.', linestyle='dashed', color='tab:purple')
plt.legend(handles=[line_red, line_purple])

#plt.legend(handles=[purple_patch])
#plt.savefig(gtype+"_ost-pis-sp.pdf", dpi=300, bbox_inches='tight')
plt.show()    

# for k in range(num_iter):
#     idx = 0
#     for i in range(N):
#         for j in range(i+1, N):
#     #        if (i,j) in path_sesti:
#             if (L[i,j]>0): 
#                 print(i,j)
#                 plt.plot(first_dists, res_end[k*len(first_dists):(k+1)*len(first_dists),idx], marker='.', linestyle='dashed',
#                          color=colors[i], label='({},{})'.format(dict_i2n[i],dict_i2n[j]))
#             idx += 1
# plt.xlabel("length of l12")
# plt.ylabel("Conductivity")
# plt.ylim([-0.01, 12.])
# plt.grid()
# #plt.legend()
# #plt.savefig(gtype+"_ost-pis-sp.pdf", dpi=300, bbox_inches='tight')
# plt.show()    


# # plot for other edges
# plt.figure(figsize=(6, 3.8), dpi=100)

# for k in range(num_iter):
#     idx = 0
#     for i in range(N):
#         for j in range(i+1, N):
#             print(i,j)
#             if (L[i,j]>0) and ((i,j) not in path_sesti):
#                 plt.plot(first_dists, res_end[k*len(first_dists):(k+1)*len(first_dists),idx], marker='.', linestyle='dashed',
#                          color=colors[i+2], label='({},{})'.format(dict_i2n[i],dict_i2n[j]), alpha=0.8)
#             idx += 1
# plt.xlabel("length of l12")
# plt.ylabel("Conductivity")
# plt.ylim([-0.01, 11.5])
# plt.grid()
# plt.legend()
# #plt.savefig(gtype+"_ost-pis-sp.pdf", dpi=300, bbox_inches='tight')
# plt.show()    



plt.figure(figsize=(6, 3.8))
#idx = 0
for k in range(num_iter):
    for i in range(N):
        plt.plot(first_dists, res_end[k*len(first_dists):(k+1)*len(first_dists),i], marker='.', #linestyle='dashed',
                        color=colors[i], #label='node {}'.format(dict_i2n[i]), 
                        markersize=1, alpha=0.8)
    #    idx += 1
plt.xlabel("Length: $l_{12}$")
plt.ylabel("# Particles")
plt.ylim([1.0, 7.5])
plt.grid()
line_blue = Line2D([0], [0], label='Node 1', marker='.', linestyle='dashed', color='tab:blue')
line_red = Line2D([0], [0], label='Node 2', marker='.', linestyle='dashed', color='tab:red')
line_green = Line2D([0], [0], label='Node 3', marker='.', linestyle='dashed', color='tab:green')
line_orange= Line2D([0], [0], label='Node 4', marker='.', linestyle='dashed', color='tab:orange')
line_purple = Line2D([0], [0], label='(Node 5', marker='.', linestyle='dashed', color='tab:purple')
plt.legend(handles=[line_blue, line_red, line_green, line_orange, line_purple])

#plt.legend(['node 1', 'node 2', 'node 3', 'node 4', 'node 5'],framealpha=0.6, loc='upper right')
#plt.legend()
#plt.savefig(gtype+"_ost-pis-sp.pdf", dpi=300, bbox_inches='tight')
plt.show()    


# print(RES)
# #%%
# # Visualisation - shortest path vs not
# pl.figure(figsize=(15, 12))
# colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
#           'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']

# # Visualisation - the state value
# pl.subplot(311)
# for i in range(N):
#   if i in souri:
#       pl.plot(t_range, RES[:,i], color='b', label='source {}'.format(dict_i2n[i]), alpha=0.8)
#   elif i in sinki:
#       pl.plot(t_range, RES[:,i], color='r', label='sink {}'.format(dict_i2n[i]), alpha=0.8)
#   else:
# # 	  pl.plot(t_range, RES[:,i], color=colors[i], linestyle='dashed', label='node {}'.format(i), alpha=0.8)
#       pl.plot(t_range, RES[:,i], linestyle='dashed', label='node {}'.format(dict_i2n[i]), alpha=0.8)
# pl.xlabel('Time')
# pl.ylabel('# Particles')
# pl.legend()

# # Visualisation - the conductivity of the edges in the shortest path
# pl.subplot(312)
# colors = ['tab:blue', 'tab:red']
# for i in range(N):
#     for j in range(i+1, N):
#         if (i,j) in path_sesti:
#             pl.plot(t_range, RES[:,j+N*(1+i)], color=colors[i], label='({},{})'.format(dict_i2n[i],dict_i2n[j]), alpha=0.6)
#             #plot steady states
#             # pl.hlines(y=D_steady[i, j], xmin=0, xmax=t_range[-1], color=colors[i], linestyles='--', label='({},{})'.format(dict_i2n[i],dict_i2n[j]))
#             pl.hlines(y=D_steady[i, j], xmin=0, xmax=t_range[-1], color=colors[i], linestyles='--')
# pl.ylabel('Conductivity')
# pl.xlabel('Time')
# pl.legend(loc='lower right')

# # Visualisation - the conductivity of the edges to the sink
# pl.subplot(313)
# colors = ['tab:purple', 'tab:red', 'tab:green', 'tab:orange']
# for i in range(N):
#     for j in range(i+1, N):
#         if (L[i,j]>0) and ((i,j) not in path_sesti):
#             pl.plot(t_range, RES[:,j+N*(1+i)], color=colors[i], label='({},{})'.format(dict_i2n[i],dict_i2n[j]), alpha=0.6)
#             #plot steady states
#             # pl.hlines(y=D_steady[i, j], xmin=0, xmax=t_range[-1], color=colors[i], linestyles='--', label='({},{})'.format(dict_i2n[i],dict_i2n[j]))
#             pl.hlines(y=D_steady[i, j], xmin=0, xmax=t_range[-1], color=colors[i], linestyles='--')
# pl.ylabel('Conductivity')
# pl.xlabel('Time')
# pl.legend()
# pl.savefig(gtype+"_continuous-1st-sp.pdf", dpi=200, bbox_inches='tight')
# pl.show()

# #%%
# # Visualisation - shortest path vs not - separated
# colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
#           'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
# figsize=(12., 4.)
# fz = 16
# plt.rcParams.update({'font.size': fz})

# # Visualisation - the state value
# plt.figure(figsize=figsize)
# for i in range(N):
#   if i in souri:
#       plt.plot(t_range, RES[:,i], color='b', label='node {}'.format(dict_i2n[i]), alpha=0.8)
#   elif i in sinki:
#       plt.plot(t_range, RES[:,i], color='r', label='node {}'.format(dict_i2n[i]), alpha=0.8)
#   else:
#  	  plt.plot(t_range, RES[:,i], color=colors[i], linestyle='dashed', label='node {}'.format(dict_i2n[i]), alpha=0.8)
#       # plt.plot(t_range, RES[:,i], linestyle='dashed', label='node {}'.format(dict_i2n[i]), alpha=0.8)
# plt.xlabel('Time')
# plt.ylabel('# Particles')
# plt.legend(framealpha=0.6, loc='upper right')
# plt.savefig(gtype+"_const-phase_Nt.png", dpi=300, bbox_inches='tight')
# plt.show()  

# #%%
# # Visualisation - the conductivity of the edges in the shortest path
# figsize=(9, 3)
# plt.figure(figsize=figsize)
# colors = ['tab:blue', 'tab:red']
# for i in range(N):
#     for j in range(i+1, N):
#         if (i,j) in path_sesti:
#             plt.plot(t_range, RES[:,j+N*(1+i)], color=colors[i], label='({},{})'.format(dict_i2n[i],dict_i2n[j]), alpha=0.6)
#             #plot steady states
#             # pl.hlines(y=D_steady[i, j], xmin=0, xmax=t_range[-1], color=colors[i], linestyles='--', label='({},{})'.format(dict_i2n[i],dict_i2n[j]))
#             plt.hlines(y=D_steady[i, j], xmin=0, xmax=t_range[-1], color=colors[i], linestyles='--')
# pl.ylabel('Conductivity')
# pl.xlabel('Time')
# pl.legend(framealpha=0.6, loc='lower right')
# plt.savefig(gtype+"_const-phase_D-sp.png", dpi=300, bbox_inches='tight')
# plt.show()

# #%%
# # Visualisation - the conductivity of the edges to the sink
# plt.figure(figsize=figsize)
# colors = ['tab:purple', 'tab:red', 'tab:green', 'tab:orange', 'tab:pink', 'tab:olive', 'tab:cyan']
# for i in range(N):
#     for j in range(i+1, N):
#         if (L[i,j]>0) and ((i,j) not in path_sesti):
#             plt.plot(t_range, RES[:,j+N*(1+i)], color=colors[i], label='({},{})'.format(dict_i2n[i],dict_i2n[j]), alpha=0.6)
#             #plot steady states
#             # pl.hlines(y=D_steady[i, j], xmin=0, xmax=t_range[-1], color=colors[i], linestyles='--', label='({},{})'.format(dict_i2n[i],dict_i2n[j]))
#             plt.hlines(y=D_steady[i, j], xmin=0, xmax=t_range[-1], color=colors[i], linestyles='--')
# plt.ylabel('Conductivity')
# plt.xlabel('Time')
# plt.legend(framealpha=0.6)
# plt.savefig(gtype+"_const-phase_D-nsp.png", dpi=300, bbox_inches='tight')
# plt.show()

# #%%
# # Video - conductivity matrix
# import matplotlib.animation as animation

# num = 1000 # the number of plots in the video
# t_lag = int(MaxTime/TS/num)
# print("# plots in a video:", num, "per", t_lag)

# # the max value
# D_res = np.zeros((N,N))
# for i,j in G.edges():
#     D_res[i,j] = RES[-1, (i+1)*N+j]
#     D_res[j,i] = RES[-1, (j+1)*N+i]
# vmax = round(np.max(D_res) + 0.1, 1)
# print("max conductivity:", vmax)

# frames = [] # for storing the generated images
# fig = plt.figure()
# for i in range(num):
#     # construct the conductancy matrix at time t
#     t = i*t_lag
#     D_res = np.zeros((N,N))
#     for i,j in G.edges():
#         D_res[i,j] = RES[t, (i+1)*N+j]
#         D_res[j,i] = RES[t, (j+1)*N+i]
#     frames.append([plt.imshow(D_res, cmap='Blues', vmax=vmax, animated=True)])

# ani = animation.ArtistAnimation(fig, frames, interval=10, blit=True,
#                                 repeat_delay=1000)
# ani.save(gtype+'_continuous-1.mp4',)
# plt.show()

# #%%
# # Video
# import matplotlib.animation as animation

# num = 1000 # number of plots in a video
# t_lag = int(MaxTime/TS/num)
# print("# plots in a video:", num, "per", t_lag)

# # the max value
# D_res = np.zeros((N,N))
# for i,j in G.edges():
#     D_res[i,j] = RES[-1, (i+1)*N+j]
#     D_res[j,i] = RES[-1, (j+1)*N+i]
# edge_max = round(np.max(D_res)+0.1, 1)
# print("max weight value:", edge_max)
# node_max = round(np.max(RES[:,:N])+0.1, 1)
# print("max node state:", node_max)

# node_size = 500
# font_size = 16
# lw = 3
# edge_color='grey'

# edges = list(G.edges())

# fig, ax = plt.subplots(figsize=(6,4))

# # Function - plot the network at a time point
# def update(idx):
#     ax.clear()
#     time = idx*t_lag
#     # print(idx, time)
#     node_color = RES[time, :N]
#     edge_width = [RES[time, (u+1)*N+v]*lw/edge_max for u,v in edges]
#     nx.draw_networkx(G, pos=pos, with_labels=True, labels=dict_i2n, node_size=node_size, 
#                     node_color=node_color, vmax=node_max, cmap='YlOrRd',
#                     edge_color=edge_color, width=edge_width, font_size=font_size,
#                     ax=ax)
#     ax.set_title(r"$l_{sp}$" + " = {}, ".format(int(dist_s))+ r"$l_{nsp}$"+" = {}".format(int(dist)))
#     # plt.grid(False)
#     # plt.box(False)
#     # imshow the weight matrix
#     # frames.append([plt.imshow(Ds[t, :, :], cmap='Blues', vmax=vmax, animated=True)])

# # Make the video
# ani = animation.FuncAnimation(fig, update, frames=num, interval=10, repeat=False)
# # animation.ArtistAnimation(fig, frames, interval=10, blit=True,
# #                                 repeat_delay=1000)
# ani.save(gtype+'_continuous-1_net.mp4')
# plt.show()