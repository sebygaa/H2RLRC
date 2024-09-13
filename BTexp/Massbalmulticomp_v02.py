
#%%
# Importing Libraries
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
# %%
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
y_feed = [0.45, 0.3, 0.25]

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
print('P_av = ', P_av) # in bar
print('dPdz =', DPDz) # in bar
print('P_0', P_L)
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
# backward FDM
#h = L/(N-1)
h = []
for ii in range(N-1):
    h.append(z[ii+1]-z[ii])
h.append(h[-1])
#h_arr = h*np.ones(N)
h_arr = np.array(h)
    
d_00 = np.diag(1/h_arr)
d_lo = np.diag(-1/h_arr[:-1],-1)
d = d_00+ d_lo
d[0,0] = 0
print(d)

dd_00 = np.diag(-2/h_arr**2)
dd_hi = np.diag(1/h_arr[1:]**2, 1)
dd_lo = np.diag(1/h_arr[:-1]**2, -1)
dd = dd_00 + dd_hi + dd_lo
dd[0,0:2] = 0
dd[-1,-3:] = 0
dd[-1,-2] = 1/h_arr[-1]**2
dd[-1,-1] = -1/h_arr[-1]**2
print(dd)
# %% Isotherm function as T ad P
def isomix(P_in,T):
    pp = []
    for ii in range(len(P_in)):
        p_tmp = P_in[ii][:]
        #ind_tmp = P_in[ii] < 1E-4
        #p_tmp[ind_tmp] = 0
        pp.append(p_tmp)
    q1 = 1*pp[0]*0.1/(1 + 0.1*pp[0] + 0.3*pp[1] + 0.4*pp[2])
    q2 = 3*pp[1]*0.3/(1 + 0.1*pp[0] + 0.3*pp[1] + 0.4*pp[2])
    q3 = 4*pp[2]*0.4/(1 + 0.1*pp[0] + 0.3*pp[1] + 0.4*pp[2])
    return [q1, q2, q3]

# %%
# Info for mass transfer
k = [2.5, 2.5, 20.5]
D_AB = [1E-7, 1E-7, 1E-7]
# %%
# ODE for mass balance
def massbal(y,t):
    y_comp = []
    P_part = []
    dy_comp = []
    ddy_comp = []
    for ii in range(n_comp-1):
        y_tmp = np.array(y[ii*N:(ii+1)*N])
        y_tmp[y_tmp<1E-6] = 0
        dy_tmp = d@y_tmp
        ddy_tmp = dd@y_tmp
        P_part_tmp = y_tmp*P
        #y_tmp[y_tmp<1E-6] = 0
        y_comp.append(y_tmp)
        P_part.append(P_part_tmp)
        dy_comp.append(dy_tmp)
        ddy_comp.append(ddy_tmp)

    y_rest = 1-np.sum(y_comp, 0)
    P_part.append(P*y_rest)
    #print('Shape of P_part[0]',P_part[0].shape)
    q = []
    for ii in range(n_comp-1, 2*n_comp-1):
        q.append(y[ii*N:(ii+1)*N])
    # Solid uptake component
    dqdt = []
    qstar = isomix(P_part,300)
    for ii in range(n_comp):
        dqdt_tmp = k[ii]*(qstar[ii] - q[ii])
        dqdt.append(dqdt_tmp)
    # Solid uptake total
    Sig_dqdt = np.sum(dqdt, 0)
    #print(Sig_dqdt)
    dy_compdt = []
    for ii in range(n_comp-1):
        term1 = -u_feed*dy_comp[ii]
        term2 = D_AB[ii]*0 # How to calculate d(yC)dz
        term3 = -rho_s*(1-epsi)/epsi*dqdt[ii]
        term4 = 0
        term5 = +y_comp[ii]*rho_s*Sig_dqdt*(1-epsi)/epsi
        
        #term1[0] = 0
        term1[0] = u_feed*d[1,1]*(y_feed[ii]-y_comp[ii][0])
        #term5[0] = 0
        #term1[-1] = -u_feed*d[1,1]*C[-1]*(y_comp[ii][-2]-y_comp[ii][-1])
        #term3[-1] = 0
        #term5[-1] = 0

        #dydt_tmp = term1+(term2 + term3+ term4 + term5)/C
        dydt_tmp = term1 + term3/C + term5/C
        #dydt_tmp[0] = 0
        #dydt_tmp[N-1] = 0 
        #ind_both = ((y_comp[ii]<1E-6) + (dydt_tmp < 0))>1.5
        #dydt_tmp[ind_both] = 0 
        dy_compdt.append(dydt_tmp)
    
    dydt = []
    for yy in dy_compdt:
        dydt.append(yy)
        #print(yy.shape)
    for qq in dqdt:
        dydt.append(qq)
        #print(qq.shape)
    dydt_arr = np.concatenate(dydt)
    
    return dydt_arr

# %% 
# Initial value
y0 = np.zeros(N*5)
y0[:N] = 0.75
y0[N:2*N] = 0.25
P0_1 = P_0*y0[0:N]
P0_2 = P_0*y0[N:2*N]
P0_3 = P_0*(1-y0[0:N]-y0[N:2*N])

print('Here')
q0_1,q0_2, q0_3 = isomix([P0_1,P0_2, P0_3], 300)

y0[2*N:3*N] = q0_1
y0[3*N:4*N] = q0_2
y0[4*N:5*N] = q0_3

#y0[0]  = 0
massbal(y0, 1)
# %%
# Run
t_dom = np.linspace(0,100,10001)
y_res = odeint(massbal, y0,t_dom,)

# %%
print(y_res)
print(y_res.shape)
# %%
# Finding the last mole fraction in gas phase
y_fra_res_list = []
y_fra_sum = np.zeros_like(y_res[:,:N])
y_fra_rest = np.ones_like(y_res[:,:N])
for ii in range(n_comp-1):
    y_fra_tmp = y_res[:,ii*N:(ii+1)*N]
    y_fra_res_list.append(y_fra_tmp)
    y_fra_sum = y_fra_sum + y_fra_tmp
y_fra_rest = y_fra_rest - y_fra_sum
y_fra_res_list.append(y_fra_rest)

# %%
# Convert into concentration
C_tmp = []
q_tmp = []
mat_y_rest = np.zeros([len(t_dom), N])
for ii in range(n_comp-1):
    mat_y_tmp = y_res[:, ii*N:(ii+1)*N]
    mat_q_tmp = y_res[:, (n_comp-1+ii)*N : (n_comp+ii)*N]
    mat_C_tmp = mat_y_tmp@np.diag(C)
    C_tmp.append(mat_C_tmp)
    q_tmp.append(mat_q_tmp)
    mat_y_rest = mat_y_rest + mat_y_tmp
C_tmp.append(mat_y_rest)
q_tmp.append(y_res[:, (2*n_comp-2)*N: (2*n_comp-1)*N])

print(len(q_tmp))
plt.figure(dpi = 80)
plt.plot(z, C_tmp[0][-2000,:])
plt.plot(z, C_tmp[1][-2000,:])
plt.plot(z, C_tmp[2][-2000,:])

plt.figure(dpi = 80)
plt.plot(z, q_tmp[0][-2000,:])
plt.plot(z, q_tmp[1][-2000,:])
plt.plot(z, q_tmp[2][-2000,:])

# %%
# Rearranging solid phase concentrations
q_res_list = []
for ii in range(n_comp):
    q_tmp = y_res[:,(n_comp-1+ii)*N:(n_comp+ii)*N]
    q_res_list.append(q_tmp)

# %%
# Draw Graph for Gas phase mole fractions
ls = ['-','--','-.',':']
n_ls = -1
plt.figure(dpi = 150, figsize = [6,5])
cc = 0

for ii in range(len(t_dom))[:12000:800]:
    n_ls = n_ls+1

    plt.plot(z, y_fra_res_list[0][ii,:], 
             color = 'b', linestyle = ls[cc%len(ls)],
             alpha = 0.5)
    
    plt.plot(z, y_fra_res_list[1][ii,:], 
             color = 'r', linestyle = ls[cc%len(ls)],
             alpha = 0.5)
    
    plt.plot(z, y_fra_res_list[2][ii,:], 
             color = 'k', linestyle = ls[cc%len(ls)], 
             alpha = 0.5, label = 't = {}'.format(t_dom[ii]))
    cc=cc+1

plt.legend(fontsize = 11, loc = [1.04, 0.02])
plt.xlabel('z-axis (m) of packed bed', fontsize = 12)
plt.ylabel('gas phase mole fraction (mol/mol)', fontsize = 12)
plt.grid(linestyle = ':')
plt.savefig('Savedfig_y_frac.png', dpi = 100, 
            bbox_inches = 'tight')

# %%
# Draw solid phase mole fraction
ls = ['-','--','-.',':']
n_ls = -1
fig,ax = plt.subplots(dpi = 150, figsize = [6,5])
cc = 0

for ii in range(len(t_dom))[:12000:800]:
    n_ls = n_ls+1

    plt.plot(z, q_res_list[0][ii,:], 
             color = 'b', linestyle = ls[cc%len(ls)],
             alpha = 0.5)
    
    plt.plot(z, q_res_list[1][ii,:], 
             color = 'r', linestyle = ls[cc%len(ls)],
             alpha = 0.5)
    
    plt.plot(z, q_res_list[2][ii,:], 
             color = 'k', linestyle = ls[cc%len(ls)],
             alpha = 0.5, label = 't = {}'.format(t_dom[ii]))
    cc=cc+1

plt.legend(fontsize = 11, loc = [1.04, 0.02])
plt.xlabel('z-axis (m) of packed bed', fontsize = 12)
plt.ylabel('solid phase mole fraction (mol/kg)', fontsize = 12)
plt.grid(linestyle = ':')
plt.savefig('Savedfig_q_res.png', dpi = 100,
            bbox_inches = 'tight')

# %%
# [TO THE END] Draw Graph for Gas phase mole fractions
ls = ['-','--','-.',':']
n_ls = -1
plt.figure(dpi = 150, figsize = [6,5])
cc = 0

for ii in range(len(t_dom))[::1000]:
    n_ls = n_ls+1

    plt.plot(z, y_fra_res_list[0][ii,:], 
             color = 'b', linestyle = ls[cc%len(ls)],
             alpha = 0.5)
    
    plt.plot(z, y_fra_res_list[1][ii,:], 
             color = 'r', linestyle = ls[cc%len(ls)],
             alpha = 0.5)
    
    plt.plot(z, y_fra_res_list[2][ii,:], 
             color = 'k', linestyle = ls[cc%len(ls)],
             alpha = 0.5, label = 't = {}'.format(t_dom[ii]))
    cc=cc+1

plt.legend(fontsize = 11, loc = [1.04, 0.02])
plt.xlabel('z-axis (m) of packed bed', fontsize = 12)
plt.ylabel('gas phase mole fraction (mol/mol)', fontsize = 12)
plt.grid(linestyle = ':')
plt.savefig('Savedfig_y_frac_end.png', dpi = 100,
            bbox_inches = 'tight')


# %%
# [TO THE END] Draw solid phase mole fraction
ls = ['-','--','-.',':']
n_ls = -1
plt.figure(dpi = 150, figsize = [6,5])
cc = 0

for ii in range(len(t_dom))[::1000]:
    n_ls = n_ls+1

    plt.plot(z, q_res_list[0][ii,:], 
             color = 'b', linestyle = ls[cc%len(ls)],
             alpha = 0.5)
    
    plt.plot(z, q_res_list[1][ii,:], 
             color = 'r', linestyle = ls[cc%len(ls)],
             alpha = 0.5)
    
    plt.plot(z, q_res_list[2][ii,:], 
             color = 'k', linestyle = ls[cc%len(ls)],
             alpha = 0.5, label = 't = {}'.format(t_dom[ii]))
    cc=cc+1

plt.legend(fontsize = 11, loc = [1.04, 0.02])

plt.xlabel('z-axis (m) of packed bed', fontsize = 12)
plt.ylabel('solid phase mole fraction (mol/kg)', fontsize = 12)
plt.grid(linestyle = ':')
plt.savefig('Savedfig_q_res_end.png', dpi = 100,
            bbox_inches = 'tight')

# %%
n_ls = -1
plt.figure()
for ii in range(len(t_dom))[:5000:400]:
    n_ls = n_ls+1

    plt.plot(z, y_res[ii,1*N:2*N],)
plt.savefig('Savedfig02.png', dpi = 100)

# %%
n_ls = -1
plt.figure()
for ii in range(len(t_dom))[:5000:400]:
    n_ls = n_ls+1

    plt.plot(z, 1 - y_res[ii,1*N:2*N] - y_res[ii,0*N:1*N] ,)
plt.savefig('Savedfig03.png', dpi = 100)

# %%
# HERE is an Important tip!
# Starting with high concentration makes easy to calculate
# Make the initial y1 high !

# %%

import pickle
file = open('savedresults.pickle','wb')

pickle.dump(y_res, file)
file.close()
# %%
# Flip the initial conditions
y0_rev_pre = y_res[-1]
print(y0_rev_pre.shape)
A_flip = np.zeros([N,N])
for ii in range(N):
    A_flip[ii, -1-ii] = 1

y0_tmp = []
for ii in range(2*n_comp-1):
    mat_y0_pre_tmp = y0_rev_pre[ii*N:(ii+1)*N]
    mat_y0_rev_tmp = A_flip@mat_y0_pre_tmp
    y0_tmp.append(mat_y0_rev_tmp)
y0_rev = np.concatenate(y0_tmp)
print(y0_rev.shape)

# %% 
# Run the simulation
t_dom = np.linspace(0,50,5001)
y_rev_result_pre = odeint(massbal, y0_rev, t_dom)

# %%
# Flip the result again

y_rev_list = [] 
for ii in range(2*n_comp-1):
    y_rev_result_tmp = y_rev_result_pre[:, ii*N: (ii+1)*N]
    y_rev_list.append(y_rev_result_tmp@A_flip)
y_rev_result = np.concatenate(y_rev_list, axis = 1)
print(y_rev_result.shape)

# %%
# Result rearrangement again

y_rest_sum = np.zeros([len(t_dom),N])
y_fra_rev_res_list = []
q_rev_res_list = []
for ii in range(n_comp-1):
    y_fra_tmp = y_rev_result[:, ii*N:(ii+1)*N]
    q_tmp = y_rev_result[:, (n_comp-1+ii)*N:(n_comp+ii)*N]
    y_rest_sum = y_rest_sum + y_fra_tmp
    y_fra_rev_res_list.append(y_fra_tmp)
    q_rev_res_list.append(q_tmp)

y_fra_rev_res_list.append(1-y_rest_sum)    
q_rev_res_list.append(y_rev_result[:, (2*n_comp-2)*N: (2*n_comp-1)*N])
    

# %%
# [Reverse initial cond.] Draw Graph for Gas phase mole fractions
ls = ['-','--','-.',':']
n_ls = -1
plt.figure(dpi = 150, figsize = [6,5])
cc = 0

for ii in range(len(t_dom))[::500]:
    n_ls = n_ls+1

    plt.plot(z, y_fra_rev_res_list[0][ii,:], 
             color = 'b', linestyle = ls[cc%len(ls)],
             alpha = 0.5)
    
    plt.plot(z, y_fra_rev_res_list[1][ii,:], 
             color = 'r', linestyle = ls[cc%len(ls)],
             alpha = 0.5)
    
    plt.plot(z, y_fra_rev_res_list[2][ii,:], 
             color = 'k', linestyle = ls[cc%len(ls)],
             alpha = 0.5, label = 't = {}'.format(t_dom[ii]))
    cc=cc+1

plt.legend(fontsize = 11, loc = [1.04, 0.02])
plt.xlabel('z-axis (m) of packed bed', fontsize = 12)
plt.ylabel('gas phase mole fraction (mol/mol)', fontsize = 12)
plt.grid(linestyle = ':')
plt.savefig('Savedfig_y_frac_end.png', dpi = 100,
            bbox_inches = 'tight')

# %%
