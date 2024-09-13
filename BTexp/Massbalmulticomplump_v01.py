# %%
# Importing

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import pickle
import pandas as pd

# %%
# Setting Values
# Basic conditions
L = 1
D = 0.2
N = 101
A_cros = D**2/4*np.pi
epsi = 0.4      # m^3/m^3
rho_s = 1000    # kg/m^3
dp = 0.02       # m (particle diameter)
#mu = [1.81, 1.81] # Pa sec (visocisty of gas)
mu_av = 1.81E-5
n_comp = 3

# %%
# Pressure boundary conditions
P_out = 1       # bar
#### HOW TO DETERMINE THIS? ####
T_feed = 300    # K
P_feed = 2      # bar
#################################
u_feed = 0.1            # m/s
y_feed = [0.2, 0.3, 0.5]

# %%
## Ergun equation
Q = u_feed*A_cros*epsi  # volumetric flowrate

Cv = 5E-3       # m^3/s/bar
P_L = P_out + Q/Cv
Rgas = 8.314    # J/mol/K
P_av = (P_out+P_feed)/2
DPDz_term1 = -1.75*P_av*1E5/Rgas/T_feed*(1-epsi)/epsi*u_feed**2
DPDz_term2 = -150*mu_av/dp**2*(1-epsi)**2/epsi**2*u_feed
DPDz = DPDz_term1 + DPDz_term2
print(P_av)
print(DPDz)
P_0 = P_L + DPDz*1E-5
P_in = P_0 + Q/Cv
print('P_L = ', P_L)
print('P_0 = ', P_0)
print('P_in = ', P_in)

# %%
## z domain and P for position
z = np.linspace(0,L, N)
P = DPDz*1E-5*z + P_0   # bar
C = P*1E5/Rgas/T_feed   # mol/m^3
print(P)


# %%
# Import Data
file = open('./savedresults.pickle', 'rb')
y_res = pickle.load(file)
file.close()

# %%
# Spliting data
#dsize = y_res.shape[1]
#print(dsize/N)
#n_comp = int( (dsize/N +1)/2)
#print(n_comp)

yi_end = y_res[-1, :N*(n_comp-1)]
qi_end = y_res[-1, N*(n_comp-1):]

yi_av_end_list = []
qi_av_end_list = [] 

y_sum_end = 0
for ii in range(n_comp):
    if ii < n_comp-1:
        yi_av_tmp = np.mean(yi_end[ii*N:(ii+1)*N])
        yi_av_end_list.append(yi_av_tmp)
        y_sum_end = y_sum_end + yi_av_tmp
    qi_av_tmp = np.mean(qi_end[ii*N:(ii+1)*N])
    qi_av_end_list.append(qi_av_tmp)
yi_rest_end = 1-y_sum_end

print(yi_av_end_list)
print(yi_rest_end)
print('Sum of yi = ', np.sum(yi_av_end_list)+yi_rest_end)
print(qi_av_end_list)    

yi_av_end = np.array(yi_av_end_list + [yi_rest_end])
qi_av_end = np.array(qi_av_end_list)

# %%
# Initial value for lumped model
V = np.pi*D**2/4*L      # m^3
P_av = (P_0 + P_L)/2    # bar
T_av = T_feed           # K
ngi = epsi*P_av*V/Rgas/T_av*yi_av_end*1E5 # mol
y0 = np.concatenate([ngi, qi_av_end])


# %% Isotherm function as T ad P
def isomix(P_in,T):
    pp = []
    for ii in range(len(P_in)):
        p_tmp = P_in[ii]
        #ind_tmp = P_in[ii] < 1E-4
        #p_tmp[ind_tmp] = 0
        pp.append(p_tmp)
    q1 = 1*pp[0]*0.05/(1 + 0.3*pp[0] + 0.05*pp[1] + 0.4*pp[2])
    q2 = 3*pp[1]*0.3/(1 + 0.3*pp[0] + 0.05*pp[1] + 0.4*pp[2])
    q3 = 4*pp[2]*0.4/(1 + 0.3*pp[0] + 0.05*pp[1] + 0.4*pp[2])
    return [q1, q2, q3]
# %%
# ODE function
Cv = 1E-6  # m^3/Pa/sec
P_outer = 5.2
# Info for mass transfer
k = [1.5, 1.5, 1.5]
D_AB = [1E-6, 1E-6, 1E-6]

def mabal(y,t):
    n = y[:n_comp]
    q = y[n_comp:]
    
    n_sum = np.sum(n)
    y_frac = n/n_sum
    P = n_sum*Rgas*T_feed/epsi/V/1E5 # bar
    P_part =y_frac*P                # bar
    Q_flow = Cv*(P_outer- P)*1E5   # m^3/sec
    if Q_flow > 0:
        F_flow = Q_flow*(P_outer/Rgas/T_feed)*1E5
        y_flow = np.array(y_feed)
    else:
        F_flow = Q_flow*(P/Rgas/T_feed)*1E5
        y_flow = y_frac
    q_star = isomix(P_part, T_feed)
    dqdt = np.array(k)*(q_star - q)
    dndt = y_flow*F_flow - (1-epsi)*V*rho_s*dqdt
    dydt = np.concatenate([dndt, dqdt])
    return dydt

# TESTING the 
y_res_bl = mabal(y0,0)
print(y_res_bl)
# %%
# Solve ODE
t_dom = np.linspace(0,50,5001)
y_res_bl = odeint(mabal, y0,t_dom,)

# %%
# Draw graph (testing)
plt.figure()
plt.plot(t_dom, y_res_bl[:,0], label = 'Gas comp1')
plt.plot(t_dom, y_res_bl[:,1], label = 'Gas comp2')
plt.plot(t_dom, y_res_bl[:,2], label = 'Gas comp3')
plt.xlabel('time (sec)', fontsize = 13)
plt.ylabel('mass in gas (mol)', fontsize = 13)
plt.legend(fontsize = 14)

plt.figure()
plt.plot(t_dom,y_res_bl[:,3], label = 'Solid comp 1')
plt.plot(t_dom,y_res_bl[:,4], label = 'Solid comp 2')
plt.plot(t_dom,y_res_bl[:,5], label = 'Solid comp 3')
plt.xlabel('time (sec)', fontsize = 13)
plt.ylabel('mass in solid (mol/kg)', fontsize = 13)
plt.legend(fontsize = 14)
#ni_av = P_0
#y0 = 


# %% 
'''
yi = y_res[:, :N*(n_comp-1)]
qi = y_res[:,N*(n_comp-1):]


# Initial data  
y_av_list = []
q_av_list = []
#print(y_rest)
#print(y_rest.shape)
#print(yi[:10])
y_sum = np.zeros_like(yi[:,0])
for ii in range(n_comp):
    if ii < n_comp-1:
        y_av_tmp = np.mean(yi[:,ii*N:(ii+1)*N],1)
        y_av_list.append(y_av_tmp)
        y_sum = y_sum + y_av_tmp
    q_av_tmp = np.mean(qi[:,ii*N:(ii+1)*N],1)
    q_av_list.append(q_av_tmp)
    
y_rest = 1- y_sum
print(y_rest.shape)

# %%
# Draw graph
#plt.figure()
#t_dom = np.linspace(0,100,10001)
#plt.plot(t_dom, y_av_list[0],)
#plt.plot(t_dom, y_av_list[1],)
#plt.plot(t_dom, y_rest,)
#plt.plot(t_dom, y_av_list[0] + y_av_list[1] + y_rest)
'''
# %%
