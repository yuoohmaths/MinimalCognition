#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modelling slime mould - oscilation 

author: Yu

This is to model the dynamics on cycle graphs, where we have two nodes oscilates between source and sink.
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
N = G.number_of_nodes()
print("{} nodes with {} edges".format(N, G.number_of_edges()))
gtype = 'cycle-n{}'.format(N)

# visualisation
pos = nx.spring_layout(G)
nx.draw_networkx(G, pos=pos, with_labels=True)
print("Nodes:", G.nodes())

#%%
# Choose source and sink
sources = [0]
sinks = [3]

print("Sources:", sources)
print("Sinks:", sinks)

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
    if i in sources:
        nodec.append(chigh[0])
    elif i in sinks:
        nodec.append(chigh[2])
    else:
        nodec.append(chigh[1])

plt.figure(figsize=(9, 5))
# pos = nx.spring_layout(G)
nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color=nodec, alpha=0.6)
nx.draw_networkx_labels(G, pos, font_size=font_size, font_family='STIXGeneral')   
nx.draw_networkx_edges(G, pos, edge_color='grey', width=lw, alpha=0.8)

plt.grid(False)
plt.box(False)
plt.savefig("{}_Graph.pdf".format(gtype), dpi=300, bbox_inches='tight')
plt.show()

#%%
# shortest paths and others
path_sest = [(0,4), (3,4)]

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
edges = [(0,4), (3,4)]
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
theta = 0.01

# difference in phase = 2*pi*f*t_lag
t_lag = [0, 1/(8*theta), 1/(4*theta), 3/(8*theta), 1/(2*theta)][4]

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
# Differential equations - for scipy.integrate.odeint
def diff_eqs(INP,t):  
    '''The main set of equations'''
    Y = np.zeros((N + N*N))
    V = INP   
    for i in range(N):
        if i in sources:
            Y[i] = oscillation(t, at) - bt*V[i]
        elif i in sinks:
            Y[i] = oscillation(t+t_lag, at) - bt*V[i]
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

# print(RES)
#%%
# Visualisation 1 - edges from source and to sink
pl.figure(figsize=(20, 16))
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']

# Visualisation - the state value
pl.subplot(411)
for i in range(N):
  if i in sources:
      pl.plot(t_range, oscillation(t_range, at) - bt*RES[:,i], color='b', label='initial source {}'.format(i), alpha=0.8)
  elif i in sinks:
      pl.plot(t_range, oscillation(t_range+t_lag, at) - bt*RES[:,i], color='r', label='initial sink {}'.format(i), alpha=0.8)
pl.xlabel('Time')
pl.ylabel('Input vs Output')
pl.legend()

pl.subplot(412)
for i in range(N):
  if i in sources:
      pl.plot(t_range, RES[:,i], color='b', label='node {}'.format(i), alpha=0.8)
  elif i in sinks:
      pl.plot(t_range, RES[:,i], color='r', label='node {}'.format(i), alpha=0.8)
  else:
	    pl.plot(t_range, RES[:,i], color=colors[i], linestyle='dashed', label='node {}'.format(i), alpha=0.8)
pl.xlabel('Time')
pl.ylabel('# Particles')
pl.legend()

# Visualisation - the conductivity of the edges from the source
node = sources[0]
pl.subplot(413)
for i in range(N):
  if L[node,i]>0:
      pl.plot(t_range, RES[:,i+N*(1+node)], color=colors[i], label='({},{})'.format(node,i), alpha=0.6)
      #plot steady states
      # pl.hlines(y=D_steady[node, i], xmin=0, xmax=t_range[-1], color=colors[i], linestyles='--', label='({},{})'.format(node,i))
pl.ylabel('Conductancy')
pl.xlabel('Time')
pl.legend()

# Visualisation - the conductivity of the edges to the sink
node = sinks[0]
pl.subplot(414)
for i in range(N):
  if L[node,i]>0:
      pl.plot(t_range, RES[:,i+N*(1+node)], color=colors[i], label='({},{})'.format(node,i), alpha=0.6)
      #plot steady states
      # pl.hlines(y=D_steady[node, i], xmin=0, xmax=t_range[-1], color=colors[i], linestyles='--', label='({},{})'.format(node,i))
pl.ylabel('Conductancy')
pl.xlabel('Time')
pl.legend()
pl.savefig(gtype+"_continuous-1ost-diff{:1f}pi.pdf".format(round(t_lag*2*theta,1)), dpi=200, bbox_inches='tight')
pl.show()

#%%
# visualisation 2 - edges in the shortes paths and not
pl.figure(figsize=(20, 16))
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']

# Visualisation - the state value
pl.subplot(411)
for i in range(N):
  if i in sources:
      pl.plot(t_range, oscillation(t_range, at) - bt*RES[:,i], color='b', label='initial source {}'.format(i), alpha=0.8)
  elif i in sinks:
      pl.plot(t_range, oscillation(t_range+t_lag, at) - bt*RES[:,i], color='r', label='initial sink {}'.format(i), alpha=0.8)
pl.xlabel('Time')
pl.ylabel('Input vs Output')
pl.legend()

pl.subplot(412)
for i in range(N):
  if i in sources:
      pl.plot(t_range, RES[:,i], color='b', label='node {}'.format(i), alpha=0.8)
  elif i in sinks:
      pl.plot(t_range, RES[:,i], color='r', label='node {}'.format(i), alpha=0.8)
  else:
	    pl.plot(t_range, RES[:,i], color=colors[i], linestyle='dashed', label='node {}'.format(i), alpha=0.8)
pl.xlabel('Time')
pl.ylabel('# Particles')
pl.legend()

# Visualisation - the conductivity of the edges in the shortest paths
pl.subplot(413)
for i in range(N):
    for j in range(i+1, N):
        if (i,j) in path_sest:
            pl.plot(t_range, RES[:,j+N*(1+i)], color=colors[i], label='({},{})'.format(i,j), alpha=0.6)
      #plot steady states
      # pl.hlines(y=D_steady[node, i], xmin=0, xmax=t_range[-1], color=colors[i], linestyles='--', label='({},{})'.format(node,i))
pl.ylabel('Conductancy')
pl.xlabel('Time')
pl.legend()

# Visualisation - the conductivity of the edges that are not in the shortest path
pl.subplot(414)
for i in range(N):
    for j in range(i+1, N):
        if (L[i,j]>0) and ((i,j) not in path_sest):
            pl.plot(t_range, RES[:,j+N*(1+i)], color=colors[i], label='({},{})'.format(i,j), alpha=0.6)
      #plot steady states
      # pl.hlines(y=D_steady[node, i], xmin=0, xmax=t_range[-1], color=colors[i], linestyles='--', label='({},{})'.format(node,i))
pl.ylabel('Conductancy')
pl.xlabel('Time')
pl.legend()
pl.savefig(gtype+"_continuous-1ost-p2-diff{:1f}pi.pdf".format(round(t_lag*2*theta,1)), dpi=200, bbox_inches='tight')
pl.show()
#%%
# Video - conductivity matrix
import matplotlib.animation as animation

num = 1000 # the number of plots in the video
t_lag = int(MaxTime/TS/num)
print("# plots in a video:", num, "per", t_lag)

# the max value
D_res = np.zeros((N,N))
for i,j in G.edges():
    D_res[i,j] = RES[-1, (i+1)*N+j]
    D_res[j,i] = RES[-1, (j+1)*N+i]
vmax = round(np.max(D_res) + 0.1, 1)
print("max conductancy:", vmax)

frames = [] # for storing the generated images
fig = plt.figure()
for i in range(num):
    # construct the conductancy matrix at time t
    t = i*t_lag
    D_res = np.zeros((N,N))
    for i,j in G.edges():
        D_res[i,j] = RES[t, (i+1)*N+j]
        D_res[j,i] = RES[t, (j+1)*N+i]
    frames.append([plt.imshow(D_res, cmap='Blues', vmax=vmax, animated=True)])

ani = animation.ArtistAnimation(fig, frames, interval=10, blit=True,
                                repeat_delay=1000)
ani.save(gtype+'_continuous-1ost.mp4',)
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
    D_res[i,j] = RES[-1, (i+1)*N+j]
    D_res[j,i] = RES[-1, (j+1)*N+i]
edge_max = round(np.max(D_res)+0.1, 1)
print("max weight value:", edge_max)
node_max = round(np.max(RES[:,:N])+0.1, 1)
print("max node state:", node_max)

node_size = 500
font_size = 16
lw = 3
edge_color='grey'

edges = list(G.edges())

fig, ax = plt.subplots(figsize=(6,4))

# Function - plot the network at a time point
def update(idx):
    ax.clear()
    time = idx*t_lag
    # print(idx, time)
    node_color = RES[time, :N]
    edge_width = [RES[time, (u+1)*N+v]*lw/edge_max for u,v in edges]
    nx.draw_networkx(G, pos=pos, with_labels=True, node_size=node_size, 
                    node_color=node_color, vmax=node_max, cmap='YlOrRd',
                    edge_color=edge_color, width=edge_width, font_size=font_size,
                    ax=ax)
    ax.set_title("Continuous: "+r"$l_{0,4}, l_{3,4}$" + " = {}, ".format(int(dist_s))+ r"$l_{0,1}, l_{1,2}, l_{2,3}$"+" = {}".format(int(dist)))
    # plt.grid(False)
    # plt.box(False)
    # imshow the weight matrix
    # frames.append([plt.imshow(Ds[t, :, :], cmap='Blues', vmax=vmax, animated=True)])

# Make the video
ani = animation.FuncAnimation(fig, update, frames=num, interval=10, repeat=False)
# animation.ArtistAnimation(fig, frames, interval=10, blit=True,
#                                 repeat_delay=1000)
ani.save(gtype+'_continuous-1ost_net.mp4')
plt.show()
