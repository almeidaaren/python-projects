#!/usr/bin/env python
# coding: utf-8

# In[1]:


import math
import numpy as np
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy import optimize
import warnings
warnings.filterwarnings("ignore")
################################################################################
# This is Problem 1 #
################################################################################
r_val = float(11000) # The resistor value in Ohms
ide_value = float(1.7) # Ideality factor of the Diode
temp = float(350) # P2_temperature in Kelvin of the Diode
I_s = float(1E-9) # Saturation Current in Amps
src_v = np.arange(0.1, 2.6, 0.1) # Source Voltage array values
q = float(1.6021766208e-19)
k = float(1.380648e-23)

V_diode = [] # Create the Diode Voltage List
diode_Current = [] # Create the Diode Current List

len_Volt_sweep = len(src_v) # Get the length of Volt_sweep array

# Create ERR formula as a function
def ERR(V_d, V_initial = 0):
    Err = ((V_d - V_initial)/r_val) + I_s*((math.exp(( q * V_d)/(ide_value*k*temp))) - 1)
    return Err

# Create the Diode current formula as a function
def I_Diode(V_dio):
    for i in range(len_Volt_sweep):
        result = I_s*((math.exp((q * V_dio[i])/(ide_value*k*temp))) - 1)

        # Store the result value in the diode_Current list
        diode_Current.append(result)

    # Return the diode_Current list created from the for loop
    return diode_Current

# Sweep Voltage to find roots using fsolve
for i in range(len_Volt_sweep):
    roots = fsolve(ERR, 0.6, src_v[i])
    V_diode.append(roots[0]) # Store the 0 value in the return from fsolve

# We need to call the function
I_Diode(V_diode)

plt.plot(src_v, diode_Current, color = 'blue')
plt.plot(V_diode, diode_Current, color = 'red')
plt.title('Plot for Problem 1')
plt.ylabel('Log(Diode Current)')
plt.yscale('log')
plt.xlabel('Voltages (0.0 - 2.5)')
diode_source = mpatches.Patch(color='blue', label='log(Diode Current) vs SourceVoltage')
diode_diode = mpatches.Patch(color='red', label='log(Diode Current) vs DiodeVoltage')
plt.legend(handles=[diode_source,diode_diode])
plt.show()

################################################################################
# This is Problem 2 #
################################################################################


import numpy as np
from scipy import optimize

# Constants
k = 1.380648e-23
q = 1.6021766208e-19

# Given values
area = 1.0E-8
P2_temp = 375.0 # New temperature of P2
P2_VDD_STEP = 0.6
tol = 1e-6 # Tolerance
r_val = 10000
phi_val = 0.8
ide_val = 1.5

# Load diode data from file
DiodeData = np.loadtxt('DiodeIV.txt', dtype='float64')

# Extract voltage and current data
src_v_col = 0 # column 0
meas_I_col = 1 # column 1
src_v = DiodeData[:, src_v_col] # initializing voltage array
meas_i = DiodeData[:, meas_I_col] # initializing current array
P2_V_diode = np.zeros_like(src_v)

# Define functions to compute diode current and solve for diode voltage

def compute_diode_current(Vd, n, T, Is):
    Vt = n*k*T/q
    DiodeCurrent = Is*(np.exp(Vd/Vt)-1)
    return DiodeCurrent

def solve_diode_v(Vd, V, R, n, T, Is):
    return (Vd/R - V/R + compute_diode_current(Vd, n, T, Is))

# Define functions to optimize resistance, ideality factor, and phi

def opt_r(r_value, ide_value, phi_value, area, P2_temp, src_v, meas_i):
    est_v = np.zeros_like(src_v)
    diode_i = np.zeros_like(src_v)
    prev_v = P2_VDD_STEP
    is_value = area * P2_temp * P2_temp * np.exp(-phi_value * q / ( k * P2_temp ) )
    for index in range(len(src_v)):
        prev_v = optimize.fsolve(solve_diode_v, prev_v,
                                  (src_v[index], r_value, ide_value, P2_temp, is_value),
                                  xtol=1e-12)[0]
        est_v[index] = prev_v # store for error analysis
        # Calculate the diode current
        diode_i = compute_diode_current(est_v, ide_value, P2_temp, is_value)
    return (meas_i - diode_i)/(meas_i + diode_i + 1e-15)

def opt_n(ide_value, r_value, phi_value, area, P2_temp, src_v, meas_i):
    est_v = np.zeros_like(src_v)
    diode_i = np.zeros_like(src_v)
    prev_v = P2_VDD_STEP
    is_value = area * P2_temp * P2_temp * np.exp(-phi_value * q / ( k * P2_temp ) )
    for index in range(len(src_v)):
        prev_v = optimize.fsolve(solve_diode_v, prev_v,
                                  (src_v[index], r_value, ide_value, P2_temp, is_value),
                                  xtol=1e-12)[0]
        est_v[index] = prev_v # store for error analysis
        # Calculate the diode current
        diode_i = compute_diode_current(est_v, ide_value, P2_temp, is_value)
    return (meas_i - diode_i)/(meas_i + diode_i + 1e-15)

def opt_phi(phi_value,r_value,ide_value,area,P2_temp,source_v,meas_i):
    est_v = np.zeros_like(source_v) # an array to hold the diode
    voltages
    diode_i = np.zeros_like(source_v) # an array to hold the diode
    currents
    prev_v = P2_VDD_STEP # an initial guess for the voltage
 # need to compute the reverse bias saturation current for this phi!
    is_value = area * P2_temp * P2_temp * np.exp(-phi_value * q / ( k *P2_temp ) )
    for index in range(len(source_v)):
        prev_v = optimize.fsolve(solve_diode_v,prev_v,(source_v[index],r_value,ide_value,P2_temp,is_value),xtol=1e-12)[0]
        est_v[index] = prev_v # store for error analysis
 # Calculate the Diode Current
    diode_i = compute_diode_current(est_v,ide_value,P2_temp,is_value)
    return (meas_i - diode_i)/(meas_i + diode_i +1e-15)
################################################################################
# This is how leastsq calls opt_r #
################################################################################
# We need to run a for loop to keep optipizing values
epoch = 80 # Number of iterations
for i in range(epoch):
# Find the R value to optimize
    r_val_opt = optimize.leastsq(opt_r,r_val, args=(ide_val,phi_val, area,P2_temp, src_v, meas_i))
    r_val = r_val_opt[0][0]
# Find the n value to optimize
    ide_val_opt = optimize.leastsq(opt_r,ide_val, args=(ide_val,phi_val,area,P2_temp,src_v,meas_i))
    ide_val = ide_val_opt[0][0]
# Find the phi value to optimize
    phi_val_opt = optimize.leastsq(opt_r,phi_val, args=(ide_val,phi_val,area,P2_temp,src_v,meas_i))
    phi_val = phi_val_opt[0][0]
    ERROR = opt_r(r_val, ide_val, phi_val, area, temp, src_v, meas_i)
    print(len(ERROR))
    avg_ERR = np.sum(np.abs(ERROR))/len(ERROR)
    print('\nIteration Number: ', i)
    print('r_val: %.4f' %r_val)
    print('ide_val: %.4f' %ide_val)
    print('phi_val: %.4f' %phi_val)
    print('Average Error: %.4f' %avg_ERR)
    if avg_ERR < tol:
        break
# solve_diode_v(Vd, V, R, n, T, Is)
Is = area * P2_temp**2 * np.exp(-phi_val * q / (k * P2_temp ) )
for i in range(len(src_v)):
    P2_V_diode[i] = optimize.fsolve(solve_diode_v, 0.6, (src_v[i],r_val,ide_val, P2_temp, Is), xtol=1e-12)[0]
DiodeCurr = compute_diode_current(P2_V_diode, ide_val, P2_temp, Is)
plt.plot(src_v, np.log(meas_i), 'b', label="measured current")
plt.plot(src_v, np.log(DiodeCurr), 'r', label="estimated current")
plt.xlabel('Source voltage')
plt.legend(loc="center right")
plt.show()


# In[ ]:





# In[ ]:




