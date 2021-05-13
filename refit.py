import numpy as np
import sys
from line_fit_config import *

def one_gaussian(naxis1,naxis2,result_list,results_dir,lam_r,params,gmodel):

    results = np.reshape(result_list, (naxis2,naxis1))

    a = np.zeros([naxis2,naxis1])
    b = np.zeros([naxis2,naxis1])
    flux = np.zeros([naxis2,naxis1])
    vel = np.zeros([naxis2,naxis1])
    sig = np.zeros([naxis2,naxis1])

    redchi = np.zeros([naxis2,naxis1])

    for i in np.arange(naxis1):
        for j in np.arange(naxis2):
            a[j,i] = results[j,i].params['a'].value
            b[j,i] = results[j,i].params['b'].value
            flux[j,i] = results[j,i].params['flux'].value
            vel[j,i] = results[j,i].params['vel'].value
            sig[j,i] = results[j,i].params['sig'].value

            redchi[j,i] = results[j,i].redchi

    redchi_mask = redchi*0

    #rc_limit = np.nanmedian(redchi)+(5*np.nanstd(redchi))
    rc_limit = redchi_limit

    redchi_mask[redchi>rc_limit] = 1
    
    y_mask, x_mask = np.where(redchi_mask == 1)

    radius = refit_radius

    a[np.where(redchi_mask == 1)] = np.nan
    b[np.where(redchi_mask == 1)] = np.nan
    flux[np.where(redchi_mask == 1)] = np.nan
    vel[np.where(redchi_mask == 1)] = np.nan
    sig[np.where(redchi_mask == 1)] = np.nan
    
    refit_pixs = open(results_dir+'/refit_pix.txt','w')

    for k in np.arange(len(y_mask)):
        print("Progress {:2.1%}".format(float(k)/float(len(y_mask))), end="\r")
        
        refit_pixs.write('x = '+str(x_mask[k])+' y = '+str(y_mask[k])+'\n')
    
        p_a = np.nanmedian(a[y_mask[k]-radius:y_mask[k]+radius,x_mask[k]-radius:x_mask[k]+radius])
        p_b = np.nanmedian(b[y_mask[k]-radius:y_mask[k]+radius,x_mask[k]-radius:x_mask[k]+radius])
        p_flux = np.nanmedian(flux[y_mask[k]-radius:y_mask[k]+radius,x_mask[k]-radius:x_mask[k]+radius])
        p_vel = np.nanmedian(vel[y_mask[k]-radius:y_mask[k]+radius,x_mask[k]-radius:x_mask[k]+radius])
        p_sig = np.nanmedian(sig[y_mask[k]-radius:y_mask[k]+radius,x_mask[k]-radius:x_mask[k]+radius])
        
        if ~np.isfinite(p_a):
            p_a = results[y_mask[k],x_mask[k]].params['a'].value
        if ~np.isfinite(p_b):
            p_b = results[y_mask[k],x_mask[k]].params['b'].value
        if ~np.isfinite(p_flux):
            p_flux = results[y_mask[k],x_mask[k]].params['flux'].value
        if ~np.isfinite(p_vel):
            p_vel = results[y_mask[k],x_mask[k]].params['vel'].value
        if ~np.isfinite(p_sig):
            p_sig = results[y_mask[k],x_mask[k]].params['sig'].value

        nparams = gmodel.make_params()
            
        nparams['a'].value = p_a 
        nparams['b'].value = p_b
        nparams['flux'].value = init_params['flux']
        nparams['flux'].min = init_params['flux_min']
        nparams['vel'].value = round(p_vel)
        nparams['sig'].value = p_sig
        
        nparams['vel'].min = p_vel+refit_vel_min
        nparams['vel'].max = p_vel+refit_vel_max
        
        nparams['sig'].min = init_params['sig_min']
        nparams['sig'].max = init_params['sig_max']
        
        nparams['lam0'].value = 4861.325
        nparams['lam0'].vary = False

        results[y_mask[k],x_mask[k]] = gmodel.fit(results[y_mask[k],x_mask[k]].data, nparams, x=lam_r)
        
    refit_pixs.close()
    
    result_list = np.ravel(results)
    
    return result_list

def two_gaussians(naxis1,naxis2,result_list,results_dir,lam_r,params,gmodel):

    results = np.reshape(result_list, (naxis2,naxis1))

    a = np.zeros([naxis2,naxis1])
    b = np.zeros([naxis2,naxis1])
    flux1 = np.zeros([naxis2,naxis1])
    flux2 = np.zeros([naxis2,naxis1])
    vel1 = np.zeros([naxis2,naxis1])
    vel2 = np.zeros([naxis2,naxis1])
    sig1 = np.zeros([naxis2,naxis1])
    sig2 = np.zeros([naxis2,naxis1])

    redchi = np.zeros([naxis2,naxis1])

    for i in np.arange(naxis1):
        for j in np.arange(naxis2):
            a[j,i] = results[j,i].params['a'].value
            b[j,i] = results[j,i].params['b'].value
            flux1[j,i] = results[j,i].params['flux1'].value
            flux2[j,i] = results[j,i].params['flux2'].value
            vel1[j,i] = results[j,i].params['vel1'].value
            vel2[j,i] = results[j,i].params['vel2'].value
            sig1[j,i] = results[j,i].params['sig1'].value
            sig2[j,i] = results[j,i].params['sig2'].value

            redchi[j,i] = results[j,i].redchi

    redchi_mask = redchi*0

    #rc_limit = np.nanmedian(redchi)+(5*np.nanstd(redchi)) 
    rc_limit = redchi_limit

    redchi_mask[redchi>rc_limit] = 1
    
    y_mask, x_mask = np.where(redchi_mask == 1)

    radius = refit_radius

    a[np.where(redchi_mask == 1)] = np.nan
    b[np.where(redchi_mask == 1)] = np.nan
    flux1[np.where(redchi_mask == 1)] = np.nan
    flux2[np.where(redchi_mask == 1)] = np.nan
    vel1[np.where(redchi_mask == 1)] = np.nan
    vel2[np.where(redchi_mask == 1)] = np.nan
    sig1[np.where(redchi_mask == 1)] = np.nan
    sig2[np.where(redchi_mask == 1)] = np.nan
    
    refit_pixs = open(results_dir+'/refit_pix.txt','w')

    for k in np.arange(len(y_mask)):
        print("Progress {:2.1%}".format(float(k)/float(len(y_mask))), end="\r")
        
        refit_pixs.write('x = '+str(x_mask[k])+' y = '+str(y_mask[k])+'\n')
    
        p_a = np.nanmedian(a[y_mask[k]-radius:y_mask[k]+radius,x_mask[k]-radius:x_mask[k]+radius])
        p_b = np.nanmedian(b[y_mask[k]-radius:y_mask[k]+radius,x_mask[k]-radius:x_mask[k]+radius])
        p_flux1 = np.nanmedian(flux1[y_mask[k]-radius:y_mask[k]+radius,x_mask[k]-radius:x_mask[k]+radius])
        p_flux2 = np.nanmedian(flux2[y_mask[k]-radius:y_mask[k]+radius,x_mask[k]-radius:x_mask[k]+radius])
        p_vel1 = np.nanmedian(vel1[y_mask[k]-radius:y_mask[k]+radius,x_mask[k]-radius:x_mask[k]+radius])
        p_vel2 = np.nanmedian(vel2[y_mask[k]-radius:y_mask[k]+radius,x_mask[k]-radius:x_mask[k]+radius])
        p_sig1 = np.nanmedian(sig1[y_mask[k]-radius:y_mask[k]+radius,x_mask[k]-radius:x_mask[k]+radius])
        p_sig2 = np.nanmedian(sig2[y_mask[k]-radius:y_mask[k]+radius,x_mask[k]-radius:x_mask[k]+radius])
        
        if ~np.isfinite(p_a):
            p_a = results[y_mask[k],x_mask[k]].params['a'].value
        if ~np.isfinite(p_b):
            p_b = results[y_mask[k],x_mask[k]].params['b'].value
        if ~np.isfinite(p_flux1):
            p_flux1 = results[y_mask[k],x_mask[k]].params['flux1'].value
        if ~np.isfinite(p_flux2):
            p_flux2 = results[y_mask[k],x_mask[k]].params['flux2'].value
        if ~np.isfinite(p_vel1):
            p_vel1 = results[y_mask[k],x_mask[k]].params['vel1'].value
        if ~np.isfinite(p_vel2):
            p_vel2 = results[y_mask[k],x_mask[k]].params['vel2'].value
        if ~np.isfinite(p_sig1):
            p_sig1 = results[y_mask[k],x_mask[k]].params['sig1'].value
        if ~np.isfinite(p_sig2):
            p_sig2 = results[y_mask[k],x_mask[k]].params['sig2'].value
            
        nparams = gmodel.make_params()
            
        nparams['a'].value = p_a 
        nparams['b'].value = p_b
        nparams['flux1'].value = init_params['flux']
        nparams['flux2'].value = init_params['flux']
        nparams['flux1'].min = init_params['flux_min']
        nparams['flux2'].min = init_params['flux_min']
        nparams['vel1'].value = round(p_vel1)
        nparams['vel2'].value = round(p_vel2)
        nparams['sig1'].value = p_sig1
        nparams['sig2'].value = p_sig2
        
        nparams['vel1'].min = p_vel1+refit_vel_min
        nparams['vel1'].max = p_vel1+refit_vel_max
        
        nparams['vel2'].min = p_vel2+refit_vel_min
        nparams['vel2'].max = p_vel2+refit_vel_max
        
        nparams['sig1'].min = init_params['sig_min']
        nparams['sig1'].max = init_params['sig_max']
        
        nparams['sig2'].min = init_params['sig_min']
        nparams['sig2'].max = init_params['sig_max']
        
        nparams['lam01'].value = params['lam01'].value
        nparams['lam01'].vary = False
        nparams['lam02'].value = params['lam02'].value
        nparams['lam02'].vary = False

        results[y_mask[k],x_mask[k]] = gmodel.fit(results[y_mask[k],x_mask[k]].data, nparams, x=lam_r)
    
    refit_pixs.close()
   
    result_list = np.ravel(results)
    
    return result_list

def two_gaussians_cons(naxis1,naxis2,result_list,results_dir,lam_r,params,gmodel):

    results = np.reshape(result_list, (naxis2,naxis1))

    a = np.zeros([naxis2,naxis1])
    b = np.zeros([naxis2,naxis1])
    flux = np.zeros([naxis2,naxis1])
    vel = np.zeros([naxis2,naxis1])
    sig = np.zeros([naxis2,naxis1])

    redchi = np.zeros([naxis2,naxis1])

    for i in np.arange(naxis1):
        for j in np.arange(naxis2):
            a[j,i] = results[j,i].params['a'].value
            b[j,i] = results[j,i].params['b'].value
            flux[j,i] = results[j,i].params['flux'].value
            vel[j,i] = results[j,i].params['vel'].value
            sig[j,i] = results[j,i].params['sig'].value

            redchi[j,i] = results[j,i].redchi

    redchi_mask = redchi*0

    #rc_limit = np.nanmedian(redchi)+(5*np.nanstd(redchi)) 
    rc_limit = redchi_limit

    redchi_mask[redchi>rc_limit] = 1
    
    y_mask, x_mask = np.where(redchi_mask == 1)

    radius = refit_radius

    a[np.where(redchi_mask == 1)] = np.nan
    b[np.where(redchi_mask == 1)] = np.nan
    flux[np.where(redchi_mask == 1)] = np.nan
    vel[np.where(redchi_mask == 1)] = np.nan
    sig[np.where(redchi_mask == 1)] = np.nan
    
    refit_pixs = open(results_dir+'/refit_pix.txt','w')

    for k in np.arange(len(y_mask)):
        print("Progress {:2.1%}".format(float(k)/float(len(y_mask))), end="\r")
 
        refit_pixs.write('x = '+str(x_mask[k])+' y = '+str(y_mask[k])+'\n')
 
        p_a = np.nanmedian(a[y_mask[k]-radius:y_mask[k]+radius,x_mask[k]-radius:x_mask[k]+radius])
        p_b = np.nanmedian(b[y_mask[k]-radius:y_mask[k]+radius,x_mask[k]-radius:x_mask[k]+radius])
        p_flux = np.nanmedian(flux[y_mask[k]-radius:y_mask[k]+radius,x_mask[k]-radius:x_mask[k]+radius])
        p_vel = np.nanmedian(vel[y_mask[k]-radius:y_mask[k]+radius,x_mask[k]-radius:x_mask[k]+radius])
        p_sig = np.nanmedian(sig[y_mask[k]-radius:y_mask[k]+radius,x_mask[k]-radius:x_mask[k]+radius])
        
        if ~np.isfinite(p_a):
            p_a = results[y_mask[k],x_mask[k]].params['a'].value
        if ~np.isfinite(p_b):
            p_b = results[y_mask[k],x_mask[k]].params['b'].value
        if ~np.isfinite(p_flux):
            p_flux = results[y_mask[k],x_mask[k]].params['flux'].value
        if ~np.isfinite(p_vel):
            p_vel = results[y_mask[k],x_mask[k]].params['vel'].value
        if ~np.isfinite(p_sig):
            p_sig = results[y_mask[k],x_mask[k]].params['sig'].value
            
        nparams = gmodel.make_params()
            
        nparams['a'].value = p_a 
        nparams['b'].value = p_b
        nparams['flux'].value = init_params['flux']
        nparams['flux'].min = init_params['flux_min']
        nparams['vel'].value = round(p_vel)
        nparams['sig'].value = p_sig

        nparams['vel'].min = p_vel+refit_vel_min
        nparams['vel'].max = p_vel+refit_vel_max
        
        nparams['sig'].min = init_params['sig_min']
        nparams['sig'].max = init_params['sig_max']
        
        nparams['lam01'].value = params['lam01'].value
        nparams['lam01'].vary = False
        nparams['lam02'].value = params['lam02'].value
        nparams['lam02'].vary = False

        results[y_mask[k],x_mask[k]] = gmodel.fit(results[y_mask[k],x_mask[k]].data, nparams, x=lam_r)

    refit_pixs.close()
   
    result_list = np.ravel(results)
    
    return result_list

def two_gaussians_cons_s2(naxis1,naxis2,result_list,results_dir,lam_r,params,gmodel):

    results = np.reshape(result_list, (naxis2,naxis1))

    a = np.zeros([naxis2,naxis1])
    b = np.zeros([naxis2,naxis1])
    flux = np.zeros([naxis2,naxis1])
    ratio = np.zeros([naxis2,naxis1])
    vel = np.zeros([naxis2,naxis1])
    sig = np.zeros([naxis2,naxis1])

    redchi = np.zeros([naxis2,naxis1])

    for i in np.arange(naxis1):
        for j in np.arange(naxis2):
            a[j,i] = results[j,i].params['a'].value
            b[j,i] = results[j,i].params['b'].value
            flux[j,i] = results[j,i].params['flux'].value
            ratio[j,i] = results[j,i].params['ratio'].value
            vel[j,i] = results[j,i].params['vel'].value
            sig[j,i] = results[j,i].params['sig'].value

            redchi[j,i] = results[j,i].redchi

    redchi_mask = redchi*0

    #rc_limit = np.nanmedian(redchi)+(5*np.nanstd(redchi)) 
    rc_limit = redchi_limit

    redchi_mask[redchi>rc_limit] = 1
    
    y_mask, x_mask = np.where(redchi_mask == 1)

    radius = refit_radius

    a[np.where(redchi_mask == 1)] = np.nan
    b[np.where(redchi_mask == 1)] = np.nan
    flux[np.where(redchi_mask == 1)] = np.nan
    ratio[np.where(redchi_mask == 1)] = np.nan
    vel[np.where(redchi_mask == 1)] = np.nan
    sig[np.where(redchi_mask == 1)] = np.nan
    
    refit_pixs = open(results_dir+'/refit_pix.txt','w')

    for k in np.arange(len(y_mask)):
        print("Progress {:2.1%}".format(float(k)/float(len(y_mask))), end="\r")

        refit_pixs.write('x = '+str(x_mask[k])+' y = '+str(y_mask[k])+'\n')
    
        p_a = np.nanmedian(a[y_mask[k]-radius:y_mask[k]+radius,x_mask[k]-radius:x_mask[k]+radius])
        p_b = np.nanmedian(b[y_mask[k]-radius:y_mask[k]+radius,x_mask[k]-radius:x_mask[k]+radius])
        p_flux = np.nanmedian(flux[y_mask[k]-radius:y_mask[k]+radius,x_mask[k]-radius:x_mask[k]+radius])
        p_ratio = np.nanmedian(ratio[y_mask[k]-radius:y_mask[k]+radius,x_mask[k]-radius:x_mask[k]+radius])
        p_vel = np.nanmedian(vel[y_mask[k]-radius:y_mask[k]+radius,x_mask[k]-radius:x_mask[k]+radius])
        p_sig = np.nanmedian(sig[y_mask[k]-radius:y_mask[k]+radius,x_mask[k]-radius:x_mask[k]+radius])
        
        if ~np.isfinite(p_a):
            p_a = results[y_mask[k],x_mask[k]].params['a'].value
        if ~np.isfinite(p_b):
            p_b = results[y_mask[k],x_mask[k]].params['b'].value
        if ~np.isfinite(p_flux):
            p_flux = results[y_mask[k],x_mask[k]].params['flux'].value
        if ~np.isfinite(p_ratio):
            p_ratio = results[y_mask[k],x_mask[k]].params['ratio'].value
        if ~np.isfinite(p_vel):
            p_vel = results[y_mask[k],x_mask[k]].params['vel'].value
        if ~np.isfinite(p_sig):
            p_sig = results[y_mask[k],x_mask[k]].params['sig'].value
            
        nparams = gmodel.make_params()
            
        nparams['a'].value = p_a 
        nparams['b'].value = p_b
        nparams['flux'].value = init_params['flux']
        nparams['flux'].min = init_params['flux_min']
        nparams['ratio'].value = p_ratio
        nparams['ratio'].min = init_params['s2_ratio_min']
        nparams['ratio'].max = init_params['s2_ratio_max']
        nparams['vel'].value = round(p_vel)
        nparams['sig'].value = p_sig

        nparams['vel'].min = p_vel+refit_vel_min
        nparams['vel'].max = p_vel+refit_vel_max
        
        nparams['sig'].min = init_params['sig_min']
        nparams['sig'].max = init_params['sig_max']
        
        nparams['lam01'].value = 6716.44
        nparams['lam01'].vary = False
        nparams['lam02'].value = 6730.81
        nparams['lam02'].vary = False

        results[y_mask[k],x_mask[k]] = gmodel.fit(results[y_mask[k],x_mask[k]].data, nparams, x=lam_r)
        
    refit_pixs.close()
   
    result_list = np.ravel(results)
    
    return result_list


def three_gaussians(naxis1,naxis2,result_list,results_dir,lam_r,params,gmodel):

    results = np.reshape(result_list, (naxis2,naxis1))

    a = np.zeros([naxis2,naxis1])
    b = np.zeros([naxis2,naxis1])
    flux1 = np.zeros([naxis2,naxis1])
    flux2 = np.zeros([naxis2,naxis1])
    flux3 = np.zeros([naxis2,naxis1])
    vel1 = np.zeros([naxis2,naxis1])
    vel2 = np.zeros([naxis2,naxis1])
    vel3 = np.zeros([naxis2,naxis1])
    sig1 = np.zeros([naxis2,naxis1])
    sig2 = np.zeros([naxis2,naxis1])
    sig3 = np.zeros([naxis2,naxis1])

    redchi = np.zeros([naxis2,naxis1])

    for i in np.arange(naxis1):
        for j in np.arange(naxis2):
            a[j,i] = results[j,i].params['a'].value
            b[j,i] = results[j,i].params['b'].value
            flux1[j,i] = results[j,i].params['flux1'].value
            flux2[j,i] = results[j,i].params['flux2'].value
            flux3[j,i] = results[j,i].params['flux3'].value
            vel1[j,i] = results[j,i].params['vel1'].value
            vel2[j,i] = results[j,i].params['vel2'].value
            vel3[j,i] = results[j,i].params['vel3'].value
            sig1[j,i] = results[j,i].params['sig1'].value
            sig2[j,i] = results[j,i].params['sig2'].value
            sig3[j,i] = results[j,i].params['sig3'].value

            redchi[j,i] = results[j,i].redchi

    redchi_mask = redchi*0

    #rc_limit = np.nanmedian(redchi)+(5*np.nanstd(redchi)) 
    rc_limit = redchi_limit

    redchi_mask[redchi>rc_limit] = 1
    
    y_mask, x_mask = np.where(redchi_mask == 1)

    radius = refit_radius

    a[np.where(redchi_mask == 1)] = np.nan
    b[np.where(redchi_mask == 1)] = np.nan
    flux1[np.where(redchi_mask == 1)] = np.nan
    flux2[np.where(redchi_mask == 1)] = np.nan
    flux3[np.where(redchi_mask == 1)] = np.nan
    vel1[np.where(redchi_mask == 1)] = np.nan
    vel2[np.where(redchi_mask == 1)] = np.nan
    vel3[np.where(redchi_mask == 1)] = np.nan
    sig1[np.where(redchi_mask == 1)] = np.nan
    sig2[np.where(redchi_mask == 1)] = np.nan
    sig3[np.where(redchi_mask == 1)] = np.nan
    
    refit_pixs = open(results_dir+'/refit_pix.txt','w')

    for k in np.arange(len(y_mask)):
        print("Progress {:2.1%}".format(float(k)/float(len(y_mask))), end="\r")

        refit_pixs.write('x = '+str(x_mask[k])+' y = '+str(y_mask[k])+'\n')
    
        p_a = np.nanmedian(a[y_mask[k]-radius:y_mask[k]+radius,x_mask[k]-radius:x_mask[k]+radius])
        p_b = np.nanmedian(b[y_mask[k]-radius:y_mask[k]+radius,x_mask[k]-radius:x_mask[k]+radius])
        p_flux1 = np.nanmedian(flux1[y_mask[k]-radius:y_mask[k]+radius,x_mask[k]-radius:x_mask[k]+radius])
        p_flux2 = np.nanmedian(flux2[y_mask[k]-radius:y_mask[k]+radius,x_mask[k]-radius:x_mask[k]+radius])
        p_flux3 = np.nanmedian(flux3[y_mask[k]-radius:y_mask[k]+radius,x_mask[k]-radius:x_mask[k]+radius])
        p_vel1 = np.nanmedian(vel1[y_mask[k]-radius:y_mask[k]+radius,x_mask[k]-radius:x_mask[k]+radius])
        p_vel2 = np.nanmedian(vel2[y_mask[k]-radius:y_mask[k]+radius,x_mask[k]-radius:x_mask[k]+radius])
        p_vel3 = np.nanmedian(vel3[y_mask[k]-radius:y_mask[k]+radius,x_mask[k]-radius:x_mask[k]+radius])
        p_sig1 = np.nanmedian(sig1[y_mask[k]-radius:y_mask[k]+radius,x_mask[k]-radius:x_mask[k]+radius])
        p_sig2 = np.nanmedian(sig2[y_mask[k]-radius:y_mask[k]+radius,x_mask[k]-radius:x_mask[k]+radius])
        p_sig3 = np.nanmedian(sig3[y_mask[k]-radius:y_mask[k]+radius,x_mask[k]-radius:x_mask[k]+radius])
        
        if ~np.isfinite(p_a):
            p_a = results[y_mask[k],x_mask[k]].params['a'].value
        if ~np.isfinite(p_b):
            p_b = results[y_mask[k],x_mask[k]].params['b'].value
        if ~np.isfinite(p_flux1):
            p_flux1 = results[y_mask[k],x_mask[k]].params['flux1'].value
        if ~np.isfinite(p_flux2):
            p_flux2 = results[y_mask[k],x_mask[k]].params['flux2'].value
        if ~np.isfinite(p_flux3):
            p_flux3 = results[y_mask[k],x_mask[k]].params['flux3'].value
        if ~np.isfinite(p_vel1):
            p_vel1 = results[y_mask[k],x_mask[k]].params['vel1'].value
        if ~np.isfinite(p_vel2):
            p_vel2 = results[y_mask[k],x_mask[k]].params['vel2'].value
        if ~np.isfinite(p_vel3):
            p_vel3 = results[y_mask[k],x_mask[k]].params['vel3'].value
        if ~np.isfinite(p_sig1):
            p_sig1 = results[y_mask[k],x_mask[k]].params['sig1'].value
        if ~np.isfinite(p_sig2):
            p_sig2 = results[y_mask[k],x_mask[k]].params['sig2'].value
        if ~np.isfinite(p_sig3):
            p_sig3 = results[y_mask[k],x_mask[k]].params['sig3'].value

        nparams = gmodel.make_params()
            
        nparams['a'].value = p_a 
        nparams['b'].value = p_b
        nparams['flux1'].value = init_params['flux']
        nparams['flux2'].value = init_params['flux']
        nparams['flux3'].value = init_params['flux']
        nparams['flux1'].min = init_params['flux_min']
        nparams['flux2'].min = init_params['flux_min']
        nparams['flux3'].min = init_params['flux_min']
        nparams['vel1'].value = round(p_vel1)
        nparams['vel2'].value = round(p_vel2)
        nparams['vel3'].value = round(p_vel3)
        nparams['sig1'].value = p_sig1
        nparams['sig2'].value = p_sig2
        nparams['sig3'].value = p_sig3

        
        nparams['vel1'].min = p_vel1+refit_vel_min
        nparams['vel1'].max = p_vel1+refit_vel_max
        
        nparams['vel2'].min = p_vel2+refit_vel_min
        nparams['vel2'].max = p_vel2+refit_vel_max
        
        nparams['vel3'].min = p_vel3+refit_vel_min
        nparams['vel3'].max = p_vel3+refit_vel_max
        
        nparams['sig1'].min = init_params['sig_min']
        nparams['sig1'].max = init_params['sig_max']
        
        nparams['sig2'].min = init_params['sig_min']
        nparams['sig2'].max = init_params['sig_max']
        
        nparams['sig3'].min = init_params['sig_min']
        nparams['sig3'].max = init_params['sig_max']
        
        nparams['lam01'].value = 6548.04 
        nparams['lam01'].vary = False
        nparams['lam02'].value = 6562.80
        nparams['lam02'].vary = False
        nparams['lam03'].value = 6583.46
        nparams['lam03'].vary = False

        results[y_mask[k],x_mask[k]] = gmodel.fit(results[y_mask[k],x_mask[k]].data, nparams, x=lam_r)
        
    refit_pixs.close()
   
    result_list = np.ravel(results)
    
    return result_list

def three_gaussians_cons(naxis1,naxis2,result_list,results_dir,lam_r,params,gmodel):

    results = np.reshape(result_list, (naxis2,naxis1))

    a = np.zeros([naxis2,naxis1])
    b = np.zeros([naxis2,naxis1])
    flux = np.zeros([naxis2,naxis1])
    ratio = np.zeros([naxis2,naxis1])
    vel = np.zeros([naxis2,naxis1])
    sig = np.zeros([naxis2,naxis1])

    redchi = np.zeros([naxis2,naxis1])

    for i in np.arange(naxis1):
        for j in np.arange(naxis2):
            a[j,i] = results[j,i].params['a'].value
            b[j,i] = results[j,i].params['b'].value
            flux[j,i] = results[j,i].params['flux'].value
            ratio[j,i] = results[j,i].params['ratio'].value
            vel[j,i] = results[j,i].params['vel'].value
            sig[j,i] = results[j,i].params['sig'].value

            redchi[j,i] = results[j,i].redchi

    redchi_mask = redchi*0

    #rc_limit = np.nanmedian(redchi)+(5*np.nanstd(redchi)) 
    rc_limit = redchi_limit

    redchi_mask[redchi>rc_limit] = 1
    
    y_mask, x_mask = np.where(redchi_mask == 1)

    radius = refit_radius

    a[np.where(redchi_mask == 1)] = np.nan
    b[np.where(redchi_mask == 1)] = np.nan
    flux[np.where(redchi_mask == 1)] = np.nan
    ratio[np.where(redchi_mask == 1)] = np.nan
    vel[np.where(redchi_mask == 1)] = np.nan
    sig[np.where(redchi_mask == 1)] = np.nan
    
    refit_pixs = open(results_dir+'/refit_pix.txt','w')

    for k in np.arange(len(y_mask)):
        print("Progress {:2.1%}".format(float(k)/float(len(y_mask))), end="\r")

        refit_pixs.write('x = '+str(x_mask[k])+' y = '+str(y_mask[k])+'\n')
    
        p_a = np.nanmedian(a[y_mask[k]-radius:y_mask[k]+radius,x_mask[k]-radius:x_mask[k]+radius])
        p_b = np.nanmedian(b[y_mask[k]-radius:y_mask[k]+radius,x_mask[k]-radius:x_mask[k]+radius])
        p_flux = np.nanmedian(flux[y_mask[k]-radius:y_mask[k]+radius,x_mask[k]-radius:x_mask[k]+radius])
        p_ratio = np.nanmedian(ratio[y_mask[k]-radius:y_mask[k]+radius,x_mask[k]-radius:x_mask[k]+radius])
        p_vel = np.nanmedian(vel[y_mask[k]-radius:y_mask[k]+radius,x_mask[k]-radius:x_mask[k]+radius])
        p_sig = np.nanmedian(sig[y_mask[k]-radius:y_mask[k]+radius,x_mask[k]-radius:x_mask[k]+radius])
        
        if ~np.isfinite(p_a):
            p_a = results[y_mask[k],x_mask[k]].params['a'].value
        if ~np.isfinite(p_b):
            p_b = results[y_mask[k],x_mask[k]].params['b'].value
        if ~np.isfinite(p_flux):
            p_flux = results[y_mask[k],x_mask[k]].params['flux'].value
        if ~np.isfinite(p_ratio):
            p_ratio = results[y_mask[k],x_mask[k]].params['ratio'].value
        if ~np.isfinite(p_vel):
            p_vel = results[y_mask[k],x_mask[k]].params['vel'].value
        if ~np.isfinite(p_sig):
            p_sig = results[y_mask[k],x_mask[k]].params['sig'].value
        
        nparams = gmodel.make_params()
        
        nparams['a'].value = p_a 
        nparams['b'].value = p_b
        nparams['flux'].value = init_params['flux']
        nparams['flux'].min = init_params['flux_min']
        nparams['ratio'].value = p_ratio
        nparams['ratio'].min = init_params['n2_ha_ratio_min']
        nparams['ratio'].max = init_params['n2_ha_ratio_max']
        nparams['vel'].value = round(p_vel)
        nparams['sig'].value = p_sig

        nparams['vel'].min = p_vel+refit_vel_min
        nparams['vel'].max = p_vel+refit_vel_max   

        nparams['sig'].min = init_params['sig_min']
        nparams['sig'].max = init_params['sig_max']
        
        nparams['lam01'].value = 6548.04 
        nparams['lam01'].vary = False
        nparams['lam02'].value = 6562.80
        nparams['lam02'].vary = False
        nparams['lam03'].value = 6583.46
        nparams['lam03'].vary = False
        
        results[y_mask[k],x_mask[k]] = gmodel.fit(results[y_mask[k],x_mask[k]].data, nparams, x=lam_r)
        
    refit_pixs.close()
   
    result_list = np.ravel(results)
    
    return result_list

def three_gaussians_2g_cons(naxis1,naxis2,result_list,results_dir,lam_r,params,gmodel):

    results = np.reshape(result_list, (naxis2,naxis1))

    a = np.zeros([naxis2,naxis1])
    b = np.zeros([naxis2,naxis1])
    flux = np.zeros([naxis2,naxis1])
    flux_b = np.zeros([naxis2,naxis1])
    ratio = np.zeros([naxis2,naxis1])
    vel = np.zeros([naxis2,naxis1])
    sig = np.zeros([naxis2,naxis1])
    vel_b = np.zeros([naxis2,naxis1])
    sig_b = np.zeros([naxis2,naxis1])

    redchi = np.zeros([naxis2,naxis1])

    for i in np.arange(naxis1):
        for j in np.arange(naxis2):
            a[j,i] = results[j,i].params['a'].value
            b[j,i] = results[j,i].params['b'].value
            flux[j,i] = results[j,i].params['flux'].value
            flux_b[j,i] = results[j,i].params['flux_b'].value
            ratio[j,i] = results[j,i].params['ratio'].value
            vel[j,i] = results[j,i].params['vel'].value
            sig[j,i] = results[j,i].params['sig'].value
            vel_b[j,i] = results[j,i].params['vel_b'].value
            sig_b[j,i] = results[j,i].params['sig_b'].value

            redchi[j,i] = results[j,i].redchi

    redchi_mask = redchi*0

    #rc_limit = np.nanmedian(redchi)+(5*np.nanstd(redchi)) 
    rc_limit = redchi_limit

    redchi_mask[redchi>rc_limit] = 1
    
    y_mask, x_mask = np.where(redchi_mask == 1)

    radius = refit_radius

    a[np.where(redchi_mask == 1)] = np.nan
    b[np.where(redchi_mask == 1)] = np.nan
    flux[np.where(redchi_mask == 1)] = np.nan
    flux_b[np.where(redchi_mask == 1)] = np.nan
    ratio[np.where(redchi_mask == 1)] = np.nan
    vel[np.where(redchi_mask == 1)] = np.nan
    sig[np.where(redchi_mask == 1)] = np.nan
    vel_b[np.where(redchi_mask == 1)] = np.nan
    sig_b[np.where(redchi_mask == 1)] = np.nan
    
    refit_pixs = open(results_dir+'/refit_pix.txt','w')

    for k in np.arange(len(y_mask)):
        print("Progress {:2.1%}".format(float(k)/float(len(y_mask))), end="\r")

        refit_pixs.write('x = '+str(x_mask[k])+' y = '+str(y_mask[k])+'\n')
    
        p_a = np.nanmedian(a[y_mask[k]-radius:y_mask[k]+radius,x_mask[k]-radius:x_mask[k]+radius])
        p_b = np.nanmedian(b[y_mask[k]-radius:y_mask[k]+radius,x_mask[k]-radius:x_mask[k]+radius])
        p_flux = np.nanmedian(flux[y_mask[k]-radius:y_mask[k]+radius,x_mask[k]-radius:x_mask[k]+radius])
        p_flux_b = np.nanmedian(flux_b[y_mask[k]-radius:y_mask[k]+radius,x_mask[k]-radius:x_mask[k]+radius])
        p_ratio = np.nanmedian(ratio[y_mask[k]-radius:y_mask[k]+radius,x_mask[k]-radius:x_mask[k]+radius])
        p_vel = np.nanmedian(vel[y_mask[k]-radius:y_mask[k]+radius,x_mask[k]-radius:x_mask[k]+radius])
        p_sig = np.nanmedian(sig[y_mask[k]-radius:y_mask[k]+radius,x_mask[k]-radius:x_mask[k]+radius])
        p_vel_b = np.nanmedian(vel_b[y_mask[k]-radius:y_mask[k]+radius,x_mask[k]-radius:x_mask[k]+radius])
        p_sig_b = np.nanmedian(sig_b[y_mask[k]-radius:y_mask[k]+radius,x_mask[k]-radius:x_mask[k]+radius])
        
        if ~np.isfinite(p_a):
            p_a = results[y_mask[k],x_mask[k]].params['a'].value
        if ~np.isfinite(p_b):
            p_b = results[y_mask[k],x_mask[k]].params['b'].value
        if ~np.isfinite(p_flux):
            p_flux = results[y_mask[k],x_mask[k]].params['flux'].value
        if ~np.isfinite(p_flux_b):
            p_flux_b = results[y_mask[k],x_mask[k]].params['flux_b'].value
        if ~np.isfinite(p_ratio):
            p_ratio = results[y_mask[k],x_mask[k]].params['ratio'].value
        if ~np.isfinite(p_vel):
            p_vel = results[y_mask[k],x_mask[k]].params['vel'].value
        if ~np.isfinite(p_sig):
            p_sig = results[y_mask[k],x_mask[k]].params['sig'].value
        if ~np.isfinite(p_vel_b):
            p_vel_b = results[y_mask[k],x_mask[k]].params['vel_b'].value
        if ~np.isfinite(p_sig_b):
            p_sig_b = results[y_mask[k],x_mask[k]].params['sig_b'].value
        
        nparams = gmodel.make_params()
        
        nparams['a'].value = p_a 
        nparams['b'].value = p_b
        nparams['flux'].value = init_params['flux']
        nparams['flux'].min = init_params['flux_min']
        nparams['flux_b'].value = init_params['flux']
        nparams['flux_b'].min = init_params['flux_min']
        nparams['ratio'].value = p_ratio
        nparams['ratio'].min = init_params['n2_ha_ratio_min']
        nparams['ratio'].max = init_params['n2_ha_ratio_max']
        nparams['vel'].value = round(p_vel)
        nparams['sig'].value = p_sig
        nparams['vel_b'].value = round(p_vel_b)
        nparams['sig_b'].value = p_sig_b        

        nparams['vel'].min = p_vel+refit_vel_min
        nparams['vel'].max = p_vel+refit_vel_max   

        nparams['sig'].min = init_params['sig_min']
        nparams['sig'].max = init_params['sig_max']
        
        nparams['vel_b'].min = p_vel_b+refit_vel_min
        nparams['vel_b'].max = p_vel_b+refit_vel_max   

        nparams['sig_b'].min = init_params['sig_b_min']
        nparams['sig_b'].max = init_params['sig_b_max']

        nparams['lam01'].value = 6548.04 
        nparams['lam01'].vary = False
        nparams['lam02'].value = 6562.80
        nparams['lam02'].vary = False
        nparams['lam03'].value = 6583.46
        nparams['lam03'].vary = False
        
        results[y_mask[k],x_mask[k]] = gmodel.fit(results[y_mask[k],x_mask[k]].data, nparams, x=lam_r)
        
    refit_pixs.close()
   
    result_list = np.ravel(results)
    
    return result_list

def hb_ha_n2_gaussians_cons(naxis1,naxis2,result_list,results_dir,lam_r,params,gmodel):

    results = np.reshape(result_list, (naxis2,naxis1))

    a_hb = np.zeros([naxis2,naxis1])
    a_ha = np.zeros([naxis2,naxis1])
    b_hb = np.zeros([naxis2,naxis1])
    b_ha = np.zeros([naxis2,naxis1])
    flux = np.zeros([naxis2,naxis1])
    ratio_hb = np.zeros([naxis2,naxis1])
    ratio_n2 = np.zeros([naxis2,naxis1])
    vel = np.zeros([naxis2,naxis1])
    sig = np.zeros([naxis2,naxis1])

    redchi = np.zeros([naxis2,naxis1])

    for i in np.arange(naxis1):
        for j in np.arange(naxis2):
            a_hb[j,i] = results[j,i].params['a_hb'].value
            a_ha[j,i] = results[j,i].params['a_ha'].value
            b_hb[j,i] = results[j,i].params['b_hb'].value
            b_ha[j,i] = results[j,i].params['b_ha'].value
            flux[j,i] = results[j,i].params['flux'].value
            ratio_hb[j,i] = results[j,i].params['ratio_hb'].value
            ratio_n2[j,i] = results[j,i].params['ratio_n2'].value
            vel[j,i] = results[j,i].params['vel'].value
            sig[j,i] = results[j,i].params['sig'].value

            redchi[j,i] = results[j,i].redchi

    redchi_mask = redchi*0

    #rc_limit = np.nanmedian(redchi)+(5*np.nanstd(redchi)) 
    rc_limit = redchi_limit

    redchi_mask[redchi>rc_limit] = 1
    
    y_mask, x_mask = np.where(redchi_mask == 1)

    radius = refit_radius

    a_hb[np.where(redchi_mask == 1)] = np.nan
    a_ha[np.where(redchi_mask == 1)] = np.nan
    b_hb[np.where(redchi_mask == 1)] = np.nan
    b_ha[np.where(redchi_mask == 1)] = np.nan
    flux[np.where(redchi_mask == 1)] = np.nan
    ratio_hb[np.where(redchi_mask == 1)] = np.nan
    ratio_n2[np.where(redchi_mask == 1)] = np.nan
    vel[np.where(redchi_mask == 1)] = np.nan
    sig[np.where(redchi_mask == 1)] = np.nan
    
    refit_pixs = open(results_dir+'/refit_pix.txt','w')

    for k in np.arange(len(y_mask)):
        print("Progress {:2.1%}".format(float(k)/float(len(y_mask))), end="\r")

        refit_pixs.write('x = '+str(x_mask[k])+' y = '+str(y_mask[k])+'\n')
    
        p_a_hb = np.nanmedian(a_hb[y_mask[k]-radius:y_mask[k]+radius,x_mask[k]-radius:x_mask[k]+radius])
        p_a_ha = np.nanmedian(a_ha[y_mask[k]-radius:y_mask[k]+radius,x_mask[k]-radius:x_mask[k]+radius])
        p_b_hb = np.nanmedian(b_hb[y_mask[k]-radius:y_mask[k]+radius,x_mask[k]-radius:x_mask[k]+radius])
        p_b_ha = np.nanmedian(b_ha[y_mask[k]-radius:y_mask[k]+radius,x_mask[k]-radius:x_mask[k]+radius])
        p_flux = np.nanmedian(flux[y_mask[k]-radius:y_mask[k]+radius,x_mask[k]-radius:x_mask[k]+radius])
        p_ratio_hb = np.nanmedian(ratio_hb[y_mask[k]-radius:y_mask[k]+radius,x_mask[k]-radius:x_mask[k]+radius])
        p_ratio_n2 = np.nanmedian(ratio_n2[y_mask[k]-radius:y_mask[k]+radius,x_mask[k]-radius:x_mask[k]+radius])
        p_vel = np.nanmedian(vel[y_mask[k]-radius:y_mask[k]+radius,x_mask[k]-radius:x_mask[k]+radius])
        p_sig = np.nanmedian(sig[y_mask[k]-radius:y_mask[k]+radius,x_mask[k]-radius:x_mask[k]+radius])
        
        if ~np.isfinite(p_a_hb):
            p_a_hb = results[y_mask[k],x_mask[k]].params['a_hb'].value
        if ~np.isfinite(p_a_ha):
            p_a_ha = results[y_mask[k],x_mask[k]].params['a_ha'].value
        if ~np.isfinite(p_b_hb):
            p_b_hb = results[y_mask[k],x_mask[k]].params['b_hb'].value
        if ~np.isfinite(p_b_ha):
            p_b_ha = results[y_mask[k],x_mask[k]].params['b_ha'].value
        if ~np.isfinite(p_flux):
            p_flux = results[y_mask[k],x_mask[k]].params['flux'].value
        if ~np.isfinite(p_ratio_hb):
            p_ratio_hb = results[y_mask[k],x_mask[k]].params['ratio_hb'].value
        if ~np.isfinite(p_ratio_n2):
            p_ratio_n2 = results[y_mask[k],x_mask[k]].params['ratio_n2'].value
        if ~np.isfinite(p_vel):
            p_vel = results[y_mask[k],x_mask[k]].params['vel'].value
        if ~np.isfinite(p_sig):
            p_sig = results[y_mask[k],x_mask[k]].params['sig'].value
        
        nparams = gmodel.make_params()
        
        nparams['a_hb'].value = p_a_hb 
        nparams['a_ha'].value = p_a_ha 
        nparams['b_hb'].value = p_b_hb
        nparams['b_ha'].value = p_b_ha
        nparams['flux'].value = init_params['flux']
        nparams['flux'].min = init_params['flux_min']
        nparams['ratio_hb'].value = p_ratio_hb
        nparams['ratio_hb'].min = init_params['ha_hb_ratio_min']
        nparams['ratio_hb'].max = init_params['ha_hb_ratio_max']
        nparams['ratio_n2'].value = p_ratio_n2
        nparams['ratio_n2'].min = init_params['n2_ha_ratio_min']
        nparams['ratio_n2'].max = init_params['n2_ha_ratio_max']
        nparams['vel'].value = round(p_vel)
        nparams['sig'].value = p_sig

        nparams['vel'].min = p_vel+refit_vel_min
        nparams['vel'].max = p_vel+refit_vel_max   

        nparams['sig'].min = init_params['sig_min']
        nparams['sig'].max = init_params['sig_max']
        
        nparams['lam0_hb'].value = 4861.325
        nparams['lam0_hb'].vary = False
        nparams['lam01'].value = 6548.04 
        nparams['lam01'].vary = False
        nparams['lam02'].value = 6562.80
        nparams['lam02'].vary = False
        nparams['lam03'].value = 6583.46
        nparams['lam03'].vary = False
        
        results[y_mask[k],x_mask[k]] = gmodel.fit(results[y_mask[k],x_mask[k]].data, nparams, x=lam_r)
        
    refit_pixs.close()
   
    result_list = np.ravel(results)
    
    return result_list
