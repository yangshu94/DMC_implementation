# encoding: utf-8
'''
@author: yang shu
@contact: yangshuinhust@gmail.com
@software: vscode
@file: third_order_sys.py
@time: 2021-04-03
@desc: a simple example of a third order time delay system
'''

import numpy as np
from random import gauss
from scipy import signal
import matplotlib.pyplot as plt


class SS_sys(object):
    def __init__(self,x0,A,B,C,D,noise_std,dt):
        """Define a simple first order time delay system given the time constant tau

        Args:
            x0 (double): the initial system state
            tau (double): time constant of the delay
            noise_std (double): std of the gaussian measurement noise
            dt(double): discretization time step
        """
        self.x=x0
        
        sys_ss=signal.StateSpace(A, B, C, D)
        sys_disc=sys_ss.to_discrete(dt)
        
        self.A=np.squeeze(sys_disc.A)
        self.B=np.squeeze(sys_disc.B)
        self.C=np.squeeze(sys_disc.C)
        self.D=np.squeeze(sys_disc.D)
        
        self.noise_std=noise_std  
        
    def get_ctrl(self,u):
        self.x=self.A @ self.x + self.B * u
        y=self.C @ self.x+gauss(0, self.noise_std)
        
        return y
        

if __name__ == '__main__':  
    num_step=100   #simulate for 1000 time steps
    # system definition====================================
    # paras
    x0=np.zeros((3))
    noise_std=0.0
    dt=0.5
    
    a = np.array([[-1.0,0,0],[1,-1,0],[0,1,-1]])
    b = np.array([[1.0],[0],[0]])
    c = np.array([0,0,1.0])
    d = 0
    
    sys_temp=SS_sys(x0,a,b,c,d,noise_std,dt)
    
    # system simulation====================================
    
    t=np.arange(num_step)
    u=2*np.ones((num_step))
    y=np.zeros((num_step))
    for i in t:
        y[i]=sys_temp.get_ctrl(u=u[i])
    plt.step(t, y)
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.title('Step response of the modelled system')
    plt.grid()
    plt.show()