# encoding: utf-8
'''
@author: yang shu
@contact: yangshuinhust@gmail.com
@software: vscode
@file: main_simu.py
@time: 2021-04-03
@desc: the main file to drive this simulation
'''

import numpy as np
import matplotlib.pyplot as plt

from third_order_sys import SS_sys
from DMC_def import DMC

num_step=500   #simulate for 1000 time steps
# system definition====================================
# paras

x0=np.zeros((3))
noise_std=0.0
dt=0.5
# real physical system 
a = np.array([[-1.0,0,0],[1,-1,0],[0,1,-1]])
b = np.array([[1.0],[0],[0]])
c = np.array([0,0,1.0])
d = 0.0
    
sys=SS_sys(x0,a,b,c,d,noise_std,dt)

# controller defination =====================================
# paras
p=5    # prediction horizon
m=1    # control horizon
Wy=1.0   # weight for output tracking
Wu=0.0  # weight for control move

# state space model in the DMC 
a = np.array([[-1.0,0,0],[1,-1,0],[0,1,-1]])
b = np.array([[1.0],[0],[0]])
c = np.array([0,0,1.0])
d = 0.0

controller=DMC(x0=x0,A=a,B=b,C=c,D=d,p=p,m=m,Wy=Wy,Wu=Wu,dt=dt)
# system simulation====================================
t=np.arange(num_step)
y=np.zeros((num_step))
u=np.zeros((num_step))

set_point=np.concatenate((0.0*np.ones(50),1.0*np.ones(100),3.0*np.ones(100),-1.0*np.ones(150),0.0*np.ones(100)))
# set_point=np.concatenate((np.zeros(1950),1.0*np.ones(50)))
for i in range(num_step-1):
    
    u[i+1]=controller.get_ctrl_move(set_point[i])
    y[i+1]=sys.get_ctrl(u=u[i+1])
    controller.update_DMC(y[i+1])

fig, (ax1, ax2) = plt.subplots(2)
fig.suptitle('DMC control on a simple first order delay system')
ax1.step(t, y,label='y')
ax1.step(t, set_point,label='set point')
ax1.legend()

ax2.step(t, u,label='u')
ax2.legend()

plt.show()