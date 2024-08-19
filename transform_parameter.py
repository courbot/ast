#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Jun 30 2023
Last update Aug 19 2024

This file describes the class "Param" which gathers all the parameters useful
to perform one of the studied time-frequency transform.

@author: courbot
"""
import numpy as np

class Param:
  
    # We first describe the default values.
    # Some can be changed from the constructor below.
    f0 = 0.1
    vpo = 100 # number of voices per octave
    no = 6    # number of octaves
    
    n_sc = vpo*no
    lam = 150
    

    # a0 =2**(1/vpo)
    # Scales over which the TF is computed.
    scale = 2**(np.linspace(3,-3.3,  n_sc))

    # Number of time point.
    N = 1000
    
    # Time axis
    time_t = np.linspace(0,1,N)
    
    # Time origin and max value
    t0= 0.5
    tmax = 1
    
    # Time spacing
    dt = time_t[1]-time_t[0]
    
    # Sampling frequency
    sfreq = 1/dt
    
    fs = N
    
    alpha = 601 # Alpha is related to the gaussian analytic function writing
    beta = (alpha-1)/2 # Beta is related to the Cauchy wavelet formulation
    
    # Normalized scales   
    scale_fac = 5*(alpha-1)/(np.pi*fs);
    scale_corr = scale_fac *scale 
    
    # Frequencies, in Hz
    freq = f0/(scale*2*dt) 
    
    def __init__(self,alpha,N,tmax=1,scale=None,t0=None): 
        self.alpha = alpha
        self.N = N

        if scale is not None:
            self.scale = scale# self.scale = 2**(np.linspace(6, -3.3, self.n_sc))
        
        if t0 is None:
            self.t0 = tmax/2
        else:
            self.t0 = t0

        self.time_t = np.linspace(0,tmax,self.N)
        self.dt = self.time_t[1]-self.time_t[0]
        self.sfreq = 1/self.dt
        self.fs = self.N / tmax
        self.beta = (self.alpha-1.0)/2
        self.scale_fac = 5*(self.alpha-1)/(np.pi*self.fs)
        self.scale_corr = self.scale_fac *self.scale
        self.freq = self.f0/(self.scale*2*self.dt)    