#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Jun 28 2023
Last update Aug 19 2024

This files describe the estimation of the pair correlation function between
zeros of a spectrogram modulus, 
Note that distances are computed in hyperbolic geometry, in the complex upper-
half plane.

@author: courbot
"""

import numpy as np
from sklearn.neighbors import KernelDensity




def est_g_from_Sww(modulus,par,range_r0,r0_step):
    """
    Estimate the pair correlation function
    from a spectrogram modulus
    in hyperbolic geometry.
    """
    # Find the zeros set, and scale it
    zeros = get_scaled_zero_set(modulus,par)

    lim = range_r0.max()

    # Find all distances, accounting for inner points.
    all_dist, _ = get_all_dist(zeros,lim)
    g_est = est_g0_from_dists(all_dist, range_r0,r0_step,par.alpha)
    
    return g_est


def get_all_dist(zeros_set,lim,verbose=False):
    """ 
    Compute the distances between the inner points of a set and each points
    of the same set.
    In hyperbolic geometry.
    """
       
    mask_inner = get_inner_mask(zeros_set, lim)
 
    if verbose:print('Nb inner = %.0f'%mask_inner.sum())
    zeros_set_in =  zeros_set[mask_inner]

    if verbose:print('Computing distances...')
    interior_indices, = np.where(mask_inner)
    num_interior_particles = len(interior_indices)

    d_all = []
        
    for p in range(num_interior_particles):
            index = interior_indices[p]
            d = calc_dph(zeros_set[index],zeros_set)
            
            # remove distance to self
            d_unique = np.delete(d,index)

            if p == 0:
                d_all = np.array(d_unique)
            else:
                d_all = np.append(d_all,np.array(d_unique))

    all_dist =  np.array(d_all)

    return all_dist, zeros_set_in



def get_inner_mask(zeros_set, lim,verbose=False):
    """ 
    Get the mask indicating the inner points of a point cloud, defined as
    'distant by lim from the border', knowing we are in hyperbolic geometry.
    
    Note that this is done rather empirically : we build border vectors, then compute
    the minimum distance between any point and the said borders.
    Then, points inner enough are located in the mask.
    
    
    """
    if verbose:print('Searching for the inner points...')

    Nf = 200
    Nt = 1000

    Fmax = zeros_set.imag.max()
    Tmax = zeros_set.real.max()
    Fmin = zeros_set.imag.min()
    Tmin = zeros_set.real.min()

    left = Tmin + 1j * np.linspace(Fmin,Fmax,Nf)
    min_dist_left = np.min(calc_dph(zeros_set,left.reshape(-1,1)),axis=0)

    right = Tmax  + 1j * np.linspace(Fmin,Fmax,Nf)
    min_dist_right = np.min(calc_dph(zeros_set,right.reshape(-1,1)),axis=0)

    bot = Tmin+ np.linspace(0,Tmax,Nt) + Fmin * 1j
    min_dist_bot = np.min(calc_dph(zeros_set,bot.reshape(-1,1)),axis=0)

    top = Tmin+np.linspace(0,Tmax,Nt) +  1j * Fmax
    min_dist_top = np.min(calc_dph(zeros_set,top.reshape(-1,1)),axis=0)

    mask_inner = (min_dist_right > lim) *  (min_dist_left > lim) * (min_dist_top > lim) *  (min_dist_bot > lim)

    return mask_inner


 
def calc_dph(a,b):
    """ 
    Distance in the pseudo-hyperbolic geometry.
    """
    return np.abs( (a-b)/(a-np.conj(b)))

def get_scaled_zero_set(Sww,par):
    """Find the zero set, and scale it
    
    Source of the values : 
        https://github.com/gkoliander/WaveletPPP
        "Scale correction: due to dimensionless construction of the filterbank, 
        this is the factor that transforms the y_scaling values to correct y values"
    
    """

    xgrid = par.time_t
    ygrid = par.scale_corr

    y0,x0 = extr2minth_table(Sww,th = 1e-12)

    locminx = xgrid[x0]
    locminy = ygrid[y0]
    zeros_set = locminx+1j*locminy
    
    return zeros_set


def est_g0_from_dists(all_dist, range_r0,r0_step,alpha,norma= False,verbose=False):
    """ 
    Estimation of the pair correlation function from a set of computed
    inter-point distances.
    This is basically a weighted KDE estimation based on the "all_dist" sample. 
    """

    h = r0_step

    weight = ( (1-all_dist.flatten()**2)**2)/all_dist.flatten()

    kde = KernelDensity(kernel="epanechnikov", bandwidth=h)
    kde.fit(all_dist.flatten().reshape(-1,1),sample_weight=weight)
    kdefit = np.exp(kde.score_samples(range_r0.reshape(-1,1)))

    # Explanation : there are twice too many points because distances are
    # computed both way
    # so the histo normalization misses a factor sqrt(2)
    
    g_est = kdefit/np.sqrt(2)
    
    return g_est



def g_hp2_scalaire(r_in,alpha,h):
    """ Compute the theretical g0 over the hyperbolic plane,
    provided the value of alpha defining the plane,
    and h the integration step.
    
    Source of the equations : 
        Abreu et al. 2020
        FILTERING WITH WAVELET ZEROS AND GAUSSIAN
        ANALYTIC FUNCTIONS
        
    """
    r = r_in
    first = (1-r**2)**2 / (4 * h * r)
    
    numer2 = (alpha+1) * (1 - (r-h)**2)**alpha * (r-h)**4 - (1-(1-(r-h)**2)**(alpha+1))**2
    denom2 = (1- (r-h)**2) * (1-(1-(r-h)**2)**alpha)**2
        
    second =  numer2/denom2
    
    numer3 = (alpha+1) * (1 - (r+h)**2)**alpha * (r+h)**4 - (1-(1-(r+h)**2)**(alpha+1))**2
    denom3 = (1- (r+h)**2) * (1-(1-(r+h)**2)**alpha)**2
    third = numer3/denom3
    
    out = first*(second-third)
 
    return out

# =============================================================================
# Search for zeroes
# =============================================================================


def extr2minth(M,th):
    """
    Locates the zeros within STFT (eventually squared) modulus,
    looking if ich pixel is a local minima among its 8 neighbors.
    
    As used in several papers, such as :
        Bardenet, R., Flamant, J., & Chainais, P. (2020). 
        On the zeros of the spectrogram of white noise. 
        Applied and Computational Harmonic Analysis, 48(2), 682-705.

    Parameters
    ----------
    M : 2D array
        Modulus of the STFT.
    th : float
        minimal value to be considered a minima.

    Returns
    -------
    x : 1D array
        x-location of the zeros (time axis).
    y : 1D array
        y-location of the zeros (frequency axis).

    """

    C,R = M.shape

    Mid_Mid = np.zeros((C,R), dtype=bool)

    for c in range(1, C-1):
        for r in range(1, R-1):
            T = M[c-1:c+2,r-1:r+2]
            Mid_Mid[c, r] = (np.min(T) == T[1, 1]) * (np.min(T) > th)
            
    x, y = np.where(Mid_Mid)
    return x, y

def extr2minth_table(M,th):
    """
    (same as extr2minth, but faster as lookup is vectorized)
    Locates the zeros within STFT (eventually squared) modulus,
    looking if ich pixel is a local minima among its 8 neighbors.
    
    As used in several papers, such as :
        Bardenet, R., Flamant, J., & Chainais, P. (2020). 
        On the zeros of the spectrogram of white noise. 
        Applied and Computational Harmonic Analysis, 48(2), 682-705.

    Parameters
    ----------
    M : 2D array
        Modulus of the STFT.
    th : float
        minimal value to be considered a minima.

    Returns
    -------
    x : 1D array
        x-location of the zeros (time axis).
    y : 1D array
        y-location of the zeros (frequency axis).

    """
    C,R = M.shape
    Mroll = np.zeros(shape=(9,M.shape[0],M.shape[1]))

    Mid_Mid = np.zeros((C,R), dtype=bool)
    c = 0
    for i in (-1,0,1):
        for j in (-1,0,1):
            Mroll[c,:,:] = np.roll(M,i,j)            
            c+=1
            
    mini = np.amin(Mroll,axis=0)
    Mid_Mid = (mini==Mroll[4])*(mini > th)
    
    # ignore the borders as in the original case
    Mid_Mid[0:2,:] = 0 ; Mid_Mid[-2:,:] = 0
    Mid_Mid[:,0:2] = 0 ; Mid_Mid[:,-2:] = 0
    
    x, y = np.where(Mid_Mid)
    return x, y

