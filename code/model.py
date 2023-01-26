# -*- coding: utf-8 -*-
"""
Modelling slime mould 

author: Yu

This is to model the dynamics on cycle graphs, where we have one source and one
sink, and the input rate and the output rate is fixed (or not oscillated).
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
nx.draw(G, pos=pos, with_labels=True)
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
### Discrete dynamics ###
# dt
step = 0.01
# Number of time steps
T = 100000
# Tolerance for convergence
tol = 1e-4

# Initial condition
x_init = np.ones(N)*1.

# Record the results of the number of particles
res = np.zeros((T+1, N))
res[0, :] = x_init.copy()
# Record the results of the conductivity
Ds = np.zeros((T+1, N, N))
Ds[0, :, :] = D.copy()

# Iteration starts
for t in range(T):
    
    # Matrix - distribution of particles if there are as many as possible
    M = np.zeros((N,N))
    for j,i in G.edges():
        M[i,j] = step*(res[t, i]-res[t, j])*Ds[t,i,j]/L[i,j]
        M[j,i] = - M[i,j]
    
    # Matrix - update if distribute more than one has
    for i in range(N):
        if sum(M[:,i]) < -res[t, i]:
            # redistribute
            con_sum = sum([Ds[t,j,i]/L[j,i] for j in list(G[i])])
            for j in list(G[i]):
                M[j,i] = -res[t, i]*Ds[t,j,i]/L[j,i]/con_sum
                M[i,j] = -M[j,i]
    
    # Update the state value for each node
    for i in range(N):
        
        res_out = sum(M[:,i])
        # numerical error
        if abs(res_out) < tol:
            res_out = 0.
            
        res[t+1, i] = res[t, i] + res_out
        
        if i in sources:
            # x_new[i] += step*at
            res[t+1, i] += step*at
        elif i in sinks:
            # x_new[i] -= step*bt*x_old[i]
            res[t+1, i] -= step*bt*res[t, i]
    
    # Check convergence
    # if np.linalg.norm(res[t+1,:] - res[t,:]) < tol:
    #     print("Converge in {} steps".format(t+1))
    #     break
    
    # Update the conductivity for each edge
    for j,i in G.edges():
        Ds[t+1,j,i] = (1 + step*q*abs(res[t,j] - res[t,i])/L[j,i] - step*lam)*Ds[t,j,i]
        # if the conductancy is negative
        if Ds[t+1,j,i] < 0:
            Ds[t+1,j,i] = 0
    Ds[t+1,:,:] = Ds[t+1,:,:] + Ds[t+1,:,:].T

#%%
# Visualisation
plt.figure(figsize=(15, 10))

# Visualisation - the state value
plt.subplot(311)
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']

t_range = np.arange(T+1)*step

for i in range(N):
  if i in sources:
      plt.plot(t_range, res[:,i], color='b', label='source {}'.format(i))
  elif i in sinks:
      plt.plot(t_range, res[:,i], color='r', label='sink {}'.format(i))
  else:
	  plt.plot(t_range, res[:,i], color=colors[i], linestyle='dashed', label='node {}'.format(i))
plt.xlabel('Time (step {})'.format(step))
plt.ylabel('# Particles')
plt.legend()

# Visualisation - conductivity of the edges from the source
node = sources[0]
plt.subplot(312)
for i in range(N):
  if L[node,i]>0:
	  plt.plot(t_range, Ds[:, node, i], color=colors[i], label='({},{})'.format(node,i))
plt.ylabel('Conductancy')
plt.xlabel('Time')
plt.legend()

# Visualisation - conductivity of the edges to the sink
node = sinks[0]
plt.subplot(313)
for i in range(N):
  if L[node,i]>0:
	  plt.plot(t_range, Ds[:,node, i], color=colors[i], label='({},{})'.format(node,i))
plt.ylabel('Conductancy')
plt.xlabel('Time')
plt.legend()
# plt.savefig(gtype+"_discrete-1.pdf", dpi=200, bbox_inches='tight')
plt.show()

#%%
# Video - conductivity matrix
import matplotlib.animation as animation

num = 1000 # number of plots in a video
t_lag = int(T/num)
print("# plots in a video:", num, "per", t_lag)

frames = [] # for storing the generated images
vmax = round(np.max(Ds[-1,:,:])+0.1, 1)
print("max weight value:", vmax)

fig = plt.figure()
for i in range(num):
    t = i*t_lag
    frames.append([plt.imshow(Ds[t, :, :], cmap='Blues', vmax=vmax, animated=True)])

ani = animation.ArtistAnimation(fig, frames, interval=10, blit=True,
                                repeat_delay=1000)
ani.save(gtype+'_discret-1.mp4')
plt.show()

#%%
# Video - node (state values) and edges (conductivity)
num = 1000 # number of plots in a video
t_lag = int(T/num)
print("# plots in a video:", num, "per", t_lag)

frames = [] # for storing the generated images
edge_max = round(np.max(Ds[-1,:,:])+0.1, 1)
print("max weight value:", edge_max)
node_max = round(np.max(res)+0.1, 1)
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
    node_color = res[time, :]
    edge_width = [Ds[time, u, v]*lw/edge_max for u,v in edges]
    nx.draw_networkx(G, pos=pos, with_labels=True, node_size=node_size, 
                    node_color=node_color, vmax=node_max, cmap='YlOrRd',
                    edge_color=edge_color, width=edge_width, font_size=font_size,
                    ax=ax)
    ax.set_title("Discrete (step={}): ".format(step)+r"$l_{0,4}, l_{3,4}$" + " = {}, ".format(int(dist_s))+ r"$l_{0,1}, l_{1,2}, l_{2,3}$"+" = {}".format(int(dist)))
    # plt.grid(False)
    # plt.box(False)
    # imshow the weight matrix
    # frames.append([plt.imshow(Ds[t, :, :], cmap='Blues', vmax=vmax, animated=True)])

# Make the video
ani = animation.FuncAnimation(fig, update, frames=num, interval=10, repeat=False)
# animation.ArtistAnimation(fig, frames, interval=10, blit=True,
#                                 repeat_delay=1000)
ani.save(gtype+'_discret-1_net.mp4')
plt.show()

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
x0 = 1.0*np.ones(N);
D0 = 1.0*A.copy()
ND=MaxTime=1000.0;
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
            Y[i] = source_input(t, at)
        elif i in sinks:
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
RES = spi.odeint(diff_eqs,INPUT,t_range)

# print(RES)
#%%
# Visualisation
pl.figure(figsize=(15, 10))
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']

# Visualisation - the state value
pl.subplot(311)
for i in range(N):
  if i in sources:
      pl.plot(t_range, RES[:,i], color='b', label='source {}'.format(i))
  elif i in sinks:
      pl.plot(t_range, RES[:,i], color='r', label='sink {}'.format(i))
  else:
	    pl.plot(t_range, RES[:,i], color=colors[i], linestyle='dashed', label='node {}'.format(i))
pl.xlabel('Time')
pl.ylabel('# Particles')
pl.legend()

# Visualisation - the conductivity of the edges from the source
node = sources[0]
pl.subplot(312)
for i in range(N):
  if L[node,i]>0:
	  pl.plot(t_range, RES[:,i+N*(1+node)], color=colors[i], label='({},{})'.format(node,i))
pl.ylabel('Conductancy')
pl.xlabel('Time')
pl.legend()

# Visualisation - the conductivity of the edges to the sink
node = sinks[0]
pl.subplot(313)
for i in range(N):
  if L[node,i]>0:
	  pl.plot(t_range, RES[:,i+N*(1+node)], color=colors[i], label='({},{})'.format(node,i))
pl.ylabel('Conductancy')
pl.xlabel('Time')
pl.legend()
pl.savefig(gtype+"_continuous-1.pdf", dpi=200, bbox_inches='tight')
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
ani.save(gtype+'_continuous-1.mp4',)
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
ani.save(gtype+'_continuous-1_net.mp4')
plt.show()
