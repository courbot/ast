#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Jan 24 2024
Last update Aug 19 2024

This file implements the analytical Stockwell transform that can be applied to
a complex signal.

A version of the Cauchy wavelet transform is also available.

@author: courbot
"""

import numpy as np
from scipy.fftpack import  fft, ifft

def the_ST_transform(x,param):

    # Specrtum of x
    Fx = fft(x)

    # Range of frequencies
    nu = np.linspace(param.fs/param.N,
                     param.fs,
                     param.N) 

    xi_Hz = param.f0/(param.scale*2*param.dt)
    AST = np.zeros(shape=(param.n_sc,param.N)) * (0+0j)
    
    # Centered time
    tw = param.time_t - param.t0 
    
    for i in range(param.n_sc):
        
        # Center of the "wavelet" so that it points to xi
        f1 = 2*np.pi*xi_Hz[i] / param.beta 

        W = c_window(nu,param.fs,f1,param.beta) # later on, this could be part
                                                # of Param.

        res_before_shift = np.sqrt(xi_Hz[i])*ifft(Fx * W) # direct space
        
        shift_freq = -xi_Hz[i] # Frequency shift

        Cxxi = np.exp(1j * 2*np.pi * tw* shift_freq) # direct space
      
        AST[i] = Cxxi * res_before_shift
        
    
    return AST, np.array(xi_Hz),np.array(param.scale)


def c_window(nu,fs,xi,lam):        
    
    # AST window computation.
    # Computation in log is sometimes necessary to avoid overflow.
    log_phi = lam * np.log(nu/xi) - 2*np.pi*(nu/xi)

    phi = np.exp(log_phi)
    phi = 2*phi/phi.max()

    return phi

def the_AWT_transform(x,param):
    
    Fx = fft(x)

    # Range of frequencies
    nu = np.linspace(param.fs/param.N,
                     param.fs,
                     param.N) 

    xi_Hz = param.f0/(param.scale*2*param.dt)
    AWT = np.zeros(shape=(param.n_sc,param.N)) * (0+0j)

    for i in range(param.n_sc):

        f1 = 2*np.pi*xi_Hz[i] / param.beta

        W = c_window(nu,param.fs,f1,param.beta)

        AWT[i] = ifft(Fx * W)
    
    return AWT, np.array(xi_Hz),np.array(param.scale)
