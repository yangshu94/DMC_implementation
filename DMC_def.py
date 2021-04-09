# encoding: utf-8
'''
@author: yang shu
@contact: yangshuinhust@gmail.com
@software: vscode
@file: DMC_def.py
@time: 2021-04-03
@desc: defination of DMC
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import copy


class DMC(object):
    def __init__(self,x0,A,B,C,D,p,m,Wy,Wu,dt):
        """Define the internal model and DMC parameters

        Args:
            x0 (numpy) : system initial state
            A (numpy): state space matrix A in continuous time
            B (numpy): state space matrix B in continuous time
            C (numpy): state space matrix C in continuous time
            D (numpy): state space matrix D in continuous time
            p (int): prediction horizon
            m (int): control horizon
            Wy (numpy): weight for state
            Wu (numpy): weight for control move
            dt (double): discretization time step
        """
        
        self.x=x0
        
        sys_ss=signal.StateSpace(A, B, C, D)
        sys_disc=sys_ss.to_discrete(dt)
        
        self.A=np.squeeze(sys_disc.A)
        self.B=np.squeeze(sys_disc.B)
        self.C=np.squeeze(sys_disc.C)
        self.D=np.squeeze(sys_disc.D)
        
        # calculate the step response
        _, step_resp = signal.dstep(sys_disc,n=p+1)
        self.step_resp=np.squeeze(step_resp)[1:]
        
        # calculate the forced dynamics
        sf=np.zeros((p,m))
        for i in range(m):
            sf[i:,i]=self.step_resp[0:p-i]
        self.sf=sf    
        
        self.p=p
        self.m=m
        self.dt=dt
        self.Wy=Wy
        self.Wu=Wu
        
        self.err=0.0  # additive disturbance
        
        self.u_hist=0.0
        
    def update_DMC(self,y):
        self.x=self.A @ self.x + self.B * self.u_hist
        y_pred=self.C @ self.x
        self.err=y-y_pred
        

    def predict_free_response(self):
        """calcualte free response given current internal state

        Returns:
            [numpy (p,dim_y)]: the predicted free response without control moves
        """
        
        y_pred=np.zeros((self.p))
        x_temp=copy.deepcopy(self.x)
        for i in range(self.p):
            x_temp=self.A @ x_temp+self.B * self.u_hist
            y_pred[i]=self.C @ x_temp
        
        y_pred=y_pred+self.err
        
        return y_pred
    
    def get_ctrl_move(self,r):
        """calculate the next control move

        Args:
            r (double): the set point

        Returns:
            double: the manipulated input for next time step
        """
        y_pred=self.predict_free_response()
        free_err=r-y_pred
        forced_coef=np.transpose(self.sf) * self.Wy @ self.sf +self.Wu
        free_coef=np.transpose(self.sf) * self.Wy @ free_err
        # delta_u=np.linalg.pinv(forced_coef)@free_coef
        delta_u= np.linalg.lstsq(forced_coef,free_coef)
        
        u_next=self.u_hist+delta_u[0][0]
        self.u_hist=u_next
        
        return u_next
