# %% 
# Importing packages

# %%
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import time

# %%
# Parameters with time-related units

# %%
L = 1               # (m)
v = 0.05             # (m/sec)
v = 0.02             # (m/sec)
N = 101              # -
k_mass = [0.05, 0.05] # (1/sec) mass transfer coefficient
D_dif = 1E-6        # (m^2/sec) axial dispersion coeffic.

# %%
# Parma. w/o time term

# %%
R_gas = 8.3145      # (J/mol/K) gas const.
T_gas = 300         # (K)
rho_s = 1000        # (kg/m^3)
epsi = 0.3          # (m^3/m^3) void frac

# %%
# Boundary condition

# %%
C1_sta = 1*3E5/R_gas/T_gas  # (mol/m^3)
C2_sta = 0*3E5/R_gas/T_gas  # (mol/m^3)


# %%
# Isotherm model

# %%
'''
def ex_Lang(P, iso_params):
    denom = 1
    numer = []
    for iso,pp in zip(iso_params, P):
        denom = denom + iso[1]*pp
    for iso, pp in zip(iso_params, P):
        numer.append(iso[0]*iso[1]*pp)
    return (1/denom)*np.array(numer)
'''

qm1 = 3
b1 = 1
qm2 = 1
b2 = 0.5
#qm2 = 0
#b2 = 0.0

def ex_Lang(P1,P2, qm1_par):
    denom = 1 + b1*P1 + b2*P2
    numer1 = qm1_par*b1*P1
    numer2 = qm2*b2*P2
    q1 = numer1/denom
    q2 = numer2/denom
    return q1,q2

# %%
# FDM matrix generating

# %%
h = L/(N-1)
h_arr = h*np.ones(N)
d0 = np.diag(-1/h_arr[1:], k=-1)
d1 = np.diag(1/h_arr, k = 0)
d = d0+d1
d[0,0] = 0
d[-1,-1] = 0
d[-1, -2] = 0
#print('d = ')
#print(d)

dd0 = np.diag(1/h_arr[1:]**2, k=-1)
dd1 = np.diag(-2/h_arr**2)
dd2 = np.diag(1/h_arr[1:]**2,k=1)
dd =dd0 + dd1 + dd2
dd[0,:] = 0
dd[-1,-2] = 1/h**2
dd[-1,-1] = 1/h**2
#print('dd=')
#print(dd)

# %%
# ODE equation in function form 

# %%
def massbal(y,t, qm1_var, k_var):
    C1 = y[0:N]
    C2 = y[1*N:2*N]
    q1 = y[2*N:3*N]
    q2 = y[3*N:4*N]
    
    dC1 = d@C1
    dC2 = d@C2
    
    ddC1 = dd@C1
    ddC2 = dd@C2
    P1 = C1*R_gas*T_gas
    P2 = C2*R_gas*T_gas

    qsta = ex_Lang(P1/1E5, P2/1E5, qm1_var)
    qsta1= np.array(qsta[0]) 
    qsta2 = np.array(qsta[1])
    dq1dt = k_var*(qsta1 - q1)
    dq2dt = k_mass[1]*(qsta2 - q2)

    dC1dt= -v*dC1 + D_dif*ddC1 - (1-epsi)/epsi*rho_s*dq1dt
    dC2dt= -v*dC2 + D_dif*ddC2 - (1-epsi)/epsi*rho_s*dq2dt
    dC1dt[0] = v*(C1_sta-C1[0])/h -(1-epsi)/epsi*rho_s*dq1dt[0]
    dC2dt[0] = v*(C2_sta-C2[0])/h-(1-epsi)/epsi*rho_s*dq2dt[0]
    dC1dt[-1] = v*(C1[-2] - C1[-1])/h - (1-epsi)/epsi*rho_s*dq1dt[-1]
    dC2dt[-1] = v*(C2[-2] - C2[-1])/h - (1-epsi)/epsi*rho_s*dq2dt[-1]

    dydt = np.concatenate((dC1dt,dC2dt,dq1dt,dq2dt))
    return dydt

# %% 
# Inital conditions

# %%
C1_init = 0*3E5/R_gas/T_gas*np.ones(N)  # initial mol frac = 0
C2_init = 1*3E5/R_gas/T_gas*np.ones(N)  # initial mol frac = 0
P1_init = C1_init*R_gas*T_gas
P2_init = C2_init*R_gas*T_gas
q1_init, q2_init = ex_Lang(P1_init, P2_init, 3)

# %%
# Solve PDE
k_test = 0.05
y0 = np.concatenate((C1_init,C2_init, q1_init, q2_init))
#t_test =np.linspace(0,1500,30001)
t_test =np.linspace(0,3000,60001)
y_res = odeint(massbal, y0, t_test, args=(3, k_test))

# %% 
# Sorting Results
C1_res = y_res[:,0:N]
C2_res = y_res[:,1*N:2*N]
q1_res = y_res[:,2*N:3*N]
q2_res = y_res[:,3*N:4*N]

def yres2bt(y,N_node, ):
    C1_res = y[:,0*N:1*N]
    C2_res = y[:,1*N:2*N]
    #q1_res = y[:,2*N:3*N]
    #q2_res = y[:,2*N:3*N]
    C1_end = C1_res[:,-1]
    C2_end = C2_res[:,-1]
    Ctot_end = C1_end + C2_end + 1e-6
    yfrac1 = C1_end/Ctot_end
    return yfrac1

# %% Breakthrough data    
y1_frac_res_k1 = yres2bt(y_res, N)
    
# %%
# TEST
# %%
plt.figure(dpi = 150)
plt.plot(t_test, y1_frac_res_k1,
         linewidth = 1.9, color = 'k', linestyle = '-',
         label = 'Mass transfer (1/s) = 5x10$^{-2}$')
plt.legend(fontsize = 10)
plt.ylim([-0.01, 1.1])

# %%
# k = 0.01
# %%
k_test = 0.01
y0 = np.concatenate((C1_init,C2_init, q1_init, q2_init))
#t_test =np.linspace(0,250,2501)
#t_test =np.linspace(0,250,5001)
#t_test =np.linspace(0,1000,20001)
y_res = odeint(massbal, y0, t_test, args=(3.0, k_test))
y1_frac_res_k2 = yres2bt(y_res, N)

# %%
# k = 0.005
# %%
k_test = 0.005
y0 = np.concatenate((C1_init,C2_init, q1_init, q2_init))
#t_test =np.linspace(0,1500,30001)
#t_test =np.linspace(0,250,5001)
#t_test =np.linspace(0,1000,20001)
y_res = odeint(massbal, y0, t_test, args=(3.0, k_test))
y1_frac_res_k3 = yres2bt(y_res, N)

# %%
# Graph for diff. MTC

plt.figure(dpi = 150)

plt.plot(t_test, y1_frac_res_k1,
         linewidth = 1.9, color = 'k', linestyle = '-',
         label = 'Mass trans. coef (1/s) = 5x10$^{-2}$')

plt.plot(t_test, y1_frac_res_k2,
         linewidth = 1.9, color = 'k', linestyle = '--',
         label = 'Mass trans. coef (1/s) = 1x10$^{-2}$')

plt.plot(t_test, y1_frac_res_k3,
         linewidth = 1.9, color = 'k', linestyle = '-.',
         label = 'Mass trans. coef (1/s) = 5x10$^{-3}$')
plt.legend(fontsize = 10)
plt.ylim([-0.01, 1.1])

# %%
# Graph for C1

# %%
plt.figure(dpi = 90)
lstyle = ['-','--','-.',(0,(3,3,1,3,1,3)),':',]
cline = 0
C_res = C1_res
z = L*np.linspace(0,1,N)
for i in range(0,len(t_test),5000):
    plt.plot(z,C_res[i,:],
    color = 'k', linestyle = lstyle[cline%len(lstyle)],
    label = 't = {0:4.2f}'.format(t_test[i]))
    cline = cline + 1
plt.legend(loc = [1.03, 0.02])
plt.ylabel('Concentration 1 (mol/m$^{3}$)')
plt.xlabel('Axial distance (m)')
plt.grid(linestyle = ':', linewidth = 0.7)
#plt.savefig('C1_Profile.png', dpi = 150)


# %%
# Graph for q1

# %%
plt.figure(dpi = 90)
lstyle = ['-','--','-.',(0,(3,3,1,3,1,3)),':',]
cline = 0
q_res = q1_res
z = L*np.linspace(0,1,N)
for i in range(0,len(t_test),5000):
    plt.plot(z,q_res[i,:],
    color = 'k', linestyle = lstyle[cline%len(lstyle)],
    label = 't = {0:4.2f}'.format(t_test[i]))
    cline = cline + 1
plt.legend(loc = [1.03, 0.02])
plt.ylabel('Concentration 1 (mol/m$^{3}$)')
plt.xlabel('Axial distance (m)')
plt.grid(linestyle = ':', linewidth = 0.7)
#plt.savefig('q1_Profile.png', dpi = 150)




# %%

q_test = 3
q1_init, q2_init = ex_Lang(P1_init, P2_init, q_test)
y0 = np.concatenate((C1_init,C2_init, q1_init, q2_init))

y_res = odeint(massbal, y0, t_test, args=(q_test, 0.05))
y1_frac_res_q1 = yres2bt(y_res, N)

# %%

q_test = 3.2
q1_init, q2_init = ex_Lang(P1_init, P2_init, q_test)
y0 = np.concatenate((C1_init,C2_init, q1_init, q2_init))

y_res = odeint(massbal, y0, t_test, args=(q_test, 0.05))
y1_frac_res_q2 = yres2bt(y_res, N)

# %%
# k = 0.005
# %%
q_test = 3.5
q1_init, q2_init = ex_Lang(P1_init, P2_init, q_test)
y0 = np.concatenate((C1_init,C2_init, q1_init, q2_init))

y_res = odeint(massbal, y0, t_test, args=(q_test, 0.05))
y1_frac_res_q3 = yres2bt(y_res, N)

# %%
# Graph for diff. MTC

plt.figure(dpi = 150)

plt.plot(t_test, y1_frac_res_q1,
         linewidth = 1.9, color = 'k', linestyle = '-',
         label = 'Iso. Para. (mol/kg) = 3')

plt.plot(t_test, y1_frac_res_q2,
         linewidth = 1.9, color = 'k', linestyle = '--',
         label = 'Iso. Para. (mol/kg) = 3.2')

plt.plot(t_test, y1_frac_res_q3,
         linewidth = 1.9, color = 'k', linestyle = '-.',
         label = 'Iso. Para. (mol/kg) = 3.5')
plt.legend(fontsize = 10)
plt.ylim([-0.01, 1.1])
# %%
