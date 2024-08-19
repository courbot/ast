#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Jun 14 2024
Last update Aug 19 2024

This files describes the functions used to display the analytical Stockwell
transform in the Poincare disk D.

@author: courbot
"""
import scipy.interpolate as interp
import matplotlib.pyplot as plt
import numpy as np

import hyper_pcf as pcf



  
def plot_poincare(modulus,param,cmap,with_zeros=True,phase=False,fac_x=1,fac_y=1):
    """
    Main function. 
    Compute and plot the TF and its zeros set in the Poincare disk.
    """

    if phase:
        # Note : interpolating phase is a non-trivial task. Thus we stick to 
        # a simple 'nearest-neighbor' approach. 
        mod_poin,xgrid,ygrid= interp_to_Poincare(modulus,param,fac_x,fac_y,method='nearest',step=10)
    else:
        mod_poin,xgrid,ygrid= interp_to_Poincare(modulus,param,fac_x,fac_y,step=50)


    # Display the grid    
    Poincare_plot_grid(xgrid,ygrid,fac_x,fac_y)          

    extent = [-1,1,-1,1]

    # Display the image
    if phase:
        plt.imshow(mod_poin,extent=extent,cmap=cmap)
    else: 
        plt.imshow(np.log(mod_poin),extent=extent,cmap=cmap)
            
    # Contour of the Poincare disk
    dtheta = np.linspace(-np.pi,np.pi,100)
    dr = np.ones_like(dtheta)
    plt.plot( dr * np.sin(dtheta),dr*np.cos(dtheta),'--',c='gray')
    
    # Display the zeros
    if with_zeros:
        zeros = pcf.get_scaled_zero_set(modulus,param)
        x_proj,y_proj = Poincare_proj(zeros.real,zeros.imag,xgrid.max(),fac_x,fac_y)
        Poincare_plot_points(mod_poin,x_proj,y_proj,'r') 
        
        
    plt.xlim(-1.05,1.05)
    plt.ylim(-1.05,1.05)
    plt.axis('off')   ;


def Poincare_plot_grid(xgrid,ygrid,fac_x=1,fac_y=1,c='white'):
    """ 
    Compute and display the grid for the Poincare disk.
    """
    
    fontsize = 12
    ally = np.exp(np.linspace(np.log(ygrid.min()),np.log(ygrid.max()),10))
    ally = ally[ally < 10]
    ally = ally[2:]
    ally = np.array([0.5,1,2,4,8])
    
    # lines for fixed scales and varying time    
    for y_sca in ally:
        # what line is it in the plane :
        x_line = np.linspace(xgrid.min(),xgrid.max(),100)
        y_line = np.ones_like(x_line) * y_sca
        
        # corresponding line in the Poincare disk.
        x_circle,y_circle = Poincare_proj(x_line,y_line,xgrid.max(),fac_x,fac_y)

        plt.plot(x_circle,y_circle,c,lw=1,alpha=0.5)
        
        # label
        ylab = str(np.round( 10*(y_sca))/10)
        if ylab!='0.5':
            plt.text(x_circle[-1] + 0.03,y_circle[-1],
                     ylab,font={'size':fontsize})

    # lines for fixed time and varying scale
    x_re = np.linspace(xgrid.min(),xgrid.max(),11)
    
    for x_sca in x_re:
        
        # what line is it in the plane :
        y_line = np.linspace(ygrid.min(),ygrid.max(),1000)
        x_line = np.ones(shape=1000) * x_sca - xgrid.max()/2
        
        # corresponding line in the Poincare disk.
        x_circle,y_circle = Poincare_projline(x_line,y_line,fac_x,fac_y)
        
        plt.plot(x_circle,y_circle,c,lw=1,alpha=0.5) 
        
        # used for adjustments
        d_h = (x_circle[0])
        d_v = y_circle[0]
        
        # labels
        xlab = str(np.round( 10*(x_sca-x_re.max()/2))/10)
        
        plt.text(x_circle[0]+0.1*d_h,y_circle[0]+0.05*d_v, 
                 xlab, font={'size':fontsize},
                 verticalalignment='center',horizontalalignment='center')


def Poincare_plot_points(AST_poin,x_proj,y_proj,c='r'):
    """ Plot a set of points in the Poincare disk, scaling when necessary.
    
    Note : 2000 is a hard-wired value for the resolution of the image (see the 
    interp_to_poincare function), this is sufficient for display but could be 
    improved.
    """
    
    
    mask = np.isnan(AST_poin)
    mask_pt = np.zeros_like(x_proj)
    for i in range(x_proj.size):
        x_proj_im,y_proj_im = int((x_proj[i] + 1)/2 * 2000),  int((y_proj[i] + 1)/2 * 2000)
        mask_pt[i] = mask[y_proj_im,x_proj_im]==True
    mask_pt  = mask_pt==False
    
    
    plt.scatter(x_proj[mask_pt],y_proj[mask_pt],marker='.',s=10,color=c);
  

# =============================================================================
# Various projection functions
# =============================================================================

def Poincare_proj(x,y,xmax,fac_x = 1,fac_y = 1):
    """Project a point from C to D.
    If some "zoom" is needed, this is handled by fac_x (time) and fac_y (scale).
    """
    
    x0 = (x-xmax/2) * fac_x
    y0 = y*fac_y
    
    cp = x0 +1j*y0
    
    out = (cp - 1j) / (cp + 1j)
    
    x_proj = out.real
    y_proj = out.imag
    
    return x_proj,y_proj

def Poincare_projline(x,y,fac_x = 1,fac_y = 1):
    "Same as above but without centering the time axis."
    x0 = (x) * fac_x
    y0 = y*fac_y
    
    cp = x0 +1j*y0
    
    out = (cp - 1j) / (cp + 1j)
    
    x_proj = out.real
    y_proj = out.imag
    
    return x_proj,y_proj


def get_scaled_grid(param):

    xgrid = param.time_t
    ygrid = param.scale_corr
    return xgrid,ygrid

def interp_to_Poincare(modulus,param,fac_x = 1,fac_y = 1,method='linear',step=10):
    """
    Interpolates the results of the AST transform (modulus, or phase) from a 2D
    grid representing the upper-half complex plane, to the Poincare disk.
    
    Some interpolation parameters are passed as arguments, as the interpolation 
    can be tricky. Notably :
        - 'method' describes the interp. method
        - 'step' is the under-resolution performed before interpolation. 
        So step=1 means no under-resolution but time-consuming computations.
    
    """

    xgrid,ygrid = get_scaled_grid(param)
    dap,dbp = Poincare_proj(xgrid,ygrid.reshape(-1,1),xgrid.max(),fac_x,fac_y)
    dapf,dbpf = dap.flatten(),dbp.flatten()
    
    values = modulus.flatten()
    source = np.array([dapf,dbpf])
    
    # Diameter of the Poincare Disk in pixel. Sufficient in practice but could
    # be a parameter.

    reso = 2000
       
    dest_x,dest_y = np.mgrid[0:reso,0:reso]
    
    destination = np.array([2*dest_y.flatten()/reso-1,2*dest_x.flatten()/reso-1])
    out= interp.griddata(source[:,::step].T,values[::step],destination[:,:].T,method=method)
    
    
    # Interpolation should not be made beyond what is possible in the Poincare 
    # disk (interp.griddata doesn't know the borders) so we make a mask to 
    # to define the "forbidden zone".
    
    ima = (out.reshape(reso,reso))
    
    rectangle = np.ones_like(modulus)
    
    
    rectangle[:,0] =np.nan
    rectangle[:,-1] =np.nan
    rectangle[-1,:] =np.nan
    rectangle[0,:] =np.nan
    
    values_rect = rectangle.flatten()
    out= interp.griddata(source[:,::step].T,values_rect[::step],destination[:,:].T)
    mask = np.log(out.reshape(reso,reso))
    mask = (1.0*np.isnan(mask) + 1.0*np.isnan(mask[::-1,:]))>0 # symetry considerations    
    
    ima[mask] +=np.nan
    
    return ima,xgrid,ygrid