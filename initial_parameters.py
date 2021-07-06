import numpy as np
import matplotlib.pyplot as plt
from demon_config import *

def hb(gmodel,lam,cube,SN,c,results_dir,aut_ini):
    
    params = gmodel.make_params()

    params['a'].value = init_params['a'] 
    params['b'].value = init_params['b']
    params['flux'].value = init_params['flux']
    params['flux'].min = init_params['flux_min']
    params['vel'].value = init_params['vel']
    params['sig'].value = init_params['sig']
    params['lam0'].value = 4861.325
    params['lam0'].vary = False
    
    params['sig'].min = init_params['sig_min']
    
    params['sig'].max = init_params['sig_max']
    
    if (aut_ini == 'yes'):
        lam_i = (np.abs(lam - (params['vel']*params['lam0']/c+params['lam0']-70.))).argmin()
        lam_f = (np.abs(lam - (params['vel']*params['lam0']/c+params['lam0']+70.))).argmin()
        
        if SL_flag == 'yes':
            f_res = cube[1].data[lam_i:lam_f,:,:]-cube[5].data[lam_i:lam_f,:,:]
        if SL_flag == 'no':
            f_res = cube[1].data[lam_i:lam_f,:,:]
        lam_r = lam[lam_i:lam_f]
        
        f_res_int_eSN = np.nanmean(f_res*np.exp(SN[np.newaxis,:,:]),axis=(1,2))
        f_res_int_eSN = f_res_int_eSN*np.max(np.nanmean(f_res,axis=(1,2)))/np.max(f_res_int_eSN)
        
        initial_fit = gmodel.fit(f_res_int_eSN, params, x=lam_r)
        
        fig = plt.figure(figsize=(15,3))
        initial_fit.plot_fit()
        fig.savefig(results_dir+'/initial_fit.pdf')
        
        params = initial_fit.params
        
        params['flux'].value = init_params['flux']
    
    params['vel'].min = params['vel'].value+init_params['vel_min']
    
    params['vel'].max = params['vel'].value+init_params['vel_max']
    
    lam_i = (np.abs(lam - (params['vel']*params['lam0']/c+params['lam0']-70.))).argmin()
    lam_f = (np.abs(lam - (params['vel']*params['lam0']/c+params['lam0']+70.))).argmin()
    if SL_flag == 'yes':
        f_res = cube[1].data[lam_i:lam_f,:,:]-cube[5].data[lam_i:lam_f,:,:]
    if SL_flag == 'no':
        f_res = cube[1].data[lam_i:lam_f,:,:]
    lam_r = lam[lam_i:lam_f]
    
    return params,f_res,lam_r,SN

def o3(gmodel,lam,cube,SN,c,results_dir,aut_ini):
    
    params = gmodel.make_params()

    params['a'].value = init_params['a'] 
    params['b'].value = init_params['b']
    params['flux1'].value = init_params['flux']
    params['flux1'].min = init_params['flux_min']
    params['flux2'].value = init_params['flux']
    params['flux2'].min = init_params['flux_min']
    params['vel1'].value = init_params['vel']
    params['vel2'].value = init_params['vel'] 
    params['sig1'].value = init_params['sig']
    params['sig2'].value = init_params['sig'] 
    params['lam01'].value = 4958.911
    params['lam01'].vary = False
    params['lam02'].value = 5006.843
    params['lam02'].vary = False
    
    params['sig1'].min = init_params['sig_min']
    params['sig2'].min = init_params['sig_min']
    
    params['sig1'].max = init_params['sig_max']
    params['sig2'].max = init_params['sig_max']
    
    if (aut_ini == 'yes'):
        lam_i = (np.abs(lam - ( (((params['vel1']*params['lam01'])+(params['vel2']*params['lam02']))/2.) /c+ ((params['lam01']+params['lam02'])/2.) -70.))).argmin()
        lam_f = (np.abs(lam - ( (((params['vel1']*params['lam01'])+(params['vel2']*params['lam02']))/2.) /c+ ((params['lam01']+params['lam02'])/2.) +70.))).argmin()
        
        if SL_flag == 'yes':
            f_res = cube[1].data[lam_i:lam_f,:,:]-cube[5].data[lam_i:lam_f,:,:]
        if SL_flag == 'no':
            f_res = cube[1].data[lam_i:lam_f,:,:]
        lam_r = lam[lam_i:lam_f]
        
        f_res_int_eSN = np.nanmean(f_res*np.exp(SN[np.newaxis,:,:]),axis=(1,2))
        f_res_int_eSN = f_res_int_eSN*np.max(np.nanmean(f_res,axis=(1,2)))/np.max(f_res_int_eSN)
        
        initial_fit = gmodel.fit(f_res_int_eSN, params, x=lam_r)
        
        fig = plt.figure(figsize=(15,3))
        initial_fit.plot_fit()
        fig.savefig(results_dir+'/initial_fit.pdf')
        
        params = initial_fit.params
        
        params['flux1'].value = init_params['flux']
        params['flux2'].value = init_params['flux']
    
    params['vel1'].min = params['vel1'].value+init_params['vel_min']
    params['vel2'].min = params['vel2'].value+init_params['vel_min']
    
    params['vel1'].max = params['vel1'].value+init_params['vel_max']
    params['vel2'].max = params['vel2'].value+init_params['vel_max']
    
    lam_i = (np.abs(lam - ( (((params['vel1']*params['lam01'])+(params['vel2']*params['lam02']))/2.) /c+ ((params['lam01']+params['lam02'])/2.) -70.))).argmin()
    lam_f = (np.abs(lam - ( (((params['vel1']*params['lam01'])+(params['vel2']*params['lam02']))/2.) /c+ ((params['lam01']+params['lam02'])/2.) +70.))).argmin()
    if SL_flag == 'yes':
        f_res = cube[1].data[lam_i:lam_f,:,:]-cube[5].data[lam_i:lam_f,:,:]
    if SL_flag == 'no':
        f_res = cube[1].data[lam_i:lam_f,:,:]
    lam_r = lam[lam_i:lam_f]
    
    return params,f_res,lam_r,SN

def o3_cons(gmodel,lam,cube,SN,c,results_dir,aut_ini):
    
    params = gmodel.make_params()

    params['a'].value = init_params['a'] 
    params['b'].value = init_params['b']
    params['flux'].value = init_params['flux']
    params['flux'].min = init_params['flux_min']
    params['vel'].value = init_params['vel']
    params['sig'].value = init_params['sig']
    params['lam01'].value = 4958.911
    params['lam01'].vary = False
    params['lam02'].value = 5006.843
    params['lam02'].vary = False
    
    params['sig'].min = init_params['sig_min']
    
    params['sig'].max = init_params['sig_max']
    
    if (aut_ini == 'yes'):
        lam_i = (np.abs(lam - ( (((params['vel']*params['lam01'])+(params['vel']*params['lam02']))/2.) /c+ ((params['lam01']+params['lam02'])/2.) -70.))).argmin()
        lam_f = (np.abs(lam - ( (((params['vel']*params['lam01'])+(params['vel']*params['lam02']))/2.) /c+ ((params['lam01']+params['lam02'])/2.) +70.))).argmin()
        
        if SL_flag == 'yes':
            f_res = cube[1].data[lam_i:lam_f,:,:]-cube[5].data[lam_i:lam_f,:,:]
        if SL_flag == 'no':
            f_res = cube[1].data[lam_i:lam_f,:,:]
        lam_r = lam[lam_i:lam_f]
        
        f_res_int_eSN = np.nanmean(f_res*np.exp(SN[np.newaxis,:,:]),axis=(1,2))
        f_res_int_eSN = f_res_int_eSN*np.max(np.nanmean(f_res,axis=(1,2)))/np.max(f_res_int_eSN)
        
        initial_fit = gmodel.fit(f_res_int_eSN, params, x=lam_r)
        
        fig = plt.figure(figsize=(15,3))
        initial_fit.plot_fit()
        fig.savefig(results_dir+'/initial_fit.pdf')
        
        params = initial_fit.params
        
        params['flux'].value = init_params['flux']
    
    params['vel'].min = params['vel'].value+init_params['vel_min']
    
    params['vel'].max = params['vel'].value+init_params['vel_max']
    
    lam_i = (np.abs(lam - ( (((params['vel']*params['lam01'])+(params['vel']*params['lam02']))/2.) /c+ ((params['lam01']+params['lam02'])/2.) -70.))).argmin()
    lam_f = (np.abs(lam - ( (((params['vel']*params['lam01'])+(params['vel']*params['lam02']))/2.) /c+ ((params['lam01']+params['lam02'])/2.) +70.))).argmin()
    if SL_flag == 'yes':
        f_res = cube[1].data[lam_i:lam_f,:,:]-cube[5].data[lam_i:lam_f,:,:]
    if SL_flag == 'no':
        f_res = cube[1].data[lam_i:lam_f,:,:]
    lam_r = lam[lam_i:lam_f]
    
    return params,f_res,lam_r,SN

def o1(gmodel,lam,cube,SN,c,results_dir,aut_ini):
    
    params = gmodel.make_params()

    params['a'].value = init_params['a'] 
    params['b'].value = init_params['b']
    params['flux1'].value = init_params['flux']
    params['flux1'].min = init_params['flux_min']
    params['flux2'].value = init_params['flux']
    params['flux2'].min = init_params['flux_min']
    params['vel1'].value = init_params['vel']
    params['vel2'].value = init_params['vel'] 
    params['sig1'].value = init_params['sig']
    params['sig2'].value = init_params['sig'] 
    params['lam01'].value = 6300.304
    params['lam01'].vary = False
    params['lam02'].value = 6363.777
    params['lam02'].vary = False
    
    params['sig1'].min = init_params['sig_min']
    params['sig2'].min = init_params['sig_min']
    
    params['sig1'].max = init_params['sig_max']
    params['sig2'].max = init_params['sig_max']
    
    if (aut_ini == 'yes'):
        lam_i = (np.abs(lam - ( (((params['vel1']*params['lam01'])+(params['vel2']*params['lam02']))/2.) /c+ ((params['lam01']+params['lam02'])/2.) -70.))).argmin()
        lam_f = (np.abs(lam - ( (((params['vel1']*params['lam01'])+(params['vel2']*params['lam02']))/2.) /c+ ((params['lam01']+params['lam02'])/2.) +70.))).argmin()
        
        if SL_flag == 'yes':
            f_res = cube[1].data[lam_i:lam_f,:,:]-cube[5].data[lam_i:lam_f,:,:]
        if SL_flag == 'no':
            f_res = cube[1].data[lam_i:lam_f,:,:]
        lam_r = lam[lam_i:lam_f]
        
        f_res_int_eSN = np.nanmean(f_res*np.exp(SN[np.newaxis,:,:]),axis=(1,2))
        f_res_int_eSN = f_res_int_eSN*np.max(np.nanmean(f_res,axis=(1,2)))/np.max(f_res_int_eSN)
        
        initial_fit = gmodel.fit(f_res_int_eSN, params, x=lam_r)
        
        fig = plt.figure(figsize=(15,3))
        initial_fit.plot_fit()
        fig.savefig(results_dir+'/initial_fit.pdf')
        
        params = initial_fit.params
        
        params['flux1'].value = init_params['flux']
        params['flux2'].value = init_params['flux']
    
    params['vel1'].min = params['vel1'].value+init_params['vel_min']
    params['vel2'].min = params['vel2'].value+init_params['vel_min']
    
    params['vel1'].max = params['vel1'].value+init_params['vel_max']
    params['vel2'].max = params['vel2'].value+init_params['vel_max']
    
    lam_i = (np.abs(lam - ( (((params['vel1']*params['lam01'])+(params['vel2']*params['lam02']))/2.) /c+ ((params['lam01']+params['lam02'])/2.) -70.))).argmin()
    lam_f = (np.abs(lam - ( (((params['vel1']*params['lam01'])+(params['vel2']*params['lam02']))/2.) /c+ ((params['lam01']+params['lam02'])/2.) +70.))).argmin()
    if SL_flag == 'yes':
        f_res = cube[1].data[lam_i:lam_f,:,:]-cube[5].data[lam_i:lam_f,:,:]
    if SL_flag == 'no':
        f_res = cube[1].data[lam_i:lam_f,:,:]
    lam_r = lam[lam_i:lam_f]
    
    return params,f_res,lam_r,SN

def o1_cons(gmodel,lam,cube,SN,c,results_dir,aut_ini):
    
    params = gmodel.make_params()

    params['a'].value = init_params['a'] 
    params['b'].value = init_params['b']
    params['flux'].value = init_params['flux']
    params['flux'].min = init_params['flux_min']
    params['vel'].value = init_params['vel']
    params['sig'].value = init_params['sig']
    params['lam01'].value = 6300.304
    params['lam01'].vary = False
    params['lam02'].value = 6363.777
    params['lam02'].vary = False
    
    params['sig'].min = init_params['sig_min']
    
    params['sig'].max = init_params['sig_max']
    
    if (aut_ini == 'yes'):
        lam_i = (np.abs(lam - ( (((params['vel']*params['lam01'])+(params['vel']*params['lam02']))/2.) /c+ ((params['lam01']+params['lam02'])/2.) -70.))).argmin()
        lam_f = (np.abs(lam - ( (((params['vel']*params['lam01'])+(params['vel']*params['lam02']))/2.) /c+ ((params['lam01']+params['lam02'])/2.) +70.))).argmin()
        
        if SL_flag == 'yes':
            f_res = cube[1].data[lam_i:lam_f,:,:]-cube[5].data[lam_i:lam_f,:,:]
        if SL_flag == 'no':
            f_res = cube[1].data[lam_i:lam_f,:,:]
        lam_r = lam[lam_i:lam_f]
        
        f_res_int_eSN = np.nanmean(f_res*np.exp(SN[np.newaxis,:,:]),axis=(1,2))
        f_res_int_eSN = f_res_int_eSN*np.max(np.nanmean(f_res,axis=(1,2)))/np.max(f_res_int_eSN)
        
        initial_fit = gmodel.fit(f_res_int_eSN, params, x=lam_r)
        
        fig = plt.figure(figsize=(15,3))
        initial_fit.plot_fit()
        fig.savefig(results_dir+'/initial_fit.pdf')
        
        params = initial_fit.params
        
        params['flux'].value = init_params['flux']
    
    params['vel'].min = params['vel'].value+init_params['vel_min']
    
    params['vel'].max = params['vel'].value+init_params['vel_max']
    
    lam_i = (np.abs(lam - ( (((params['vel']*params['lam01'])+(params['vel']*params['lam02']))/2.) /c+ ((params['lam01']+params['lam02'])/2.) -70.))).argmin()
    lam_f = (np.abs(lam - ( (((params['vel']*params['lam01'])+(params['vel']*params['lam02']))/2.) /c+ ((params['lam01']+params['lam02'])/2.) +70.))).argmin()
    if SL_flag == 'yes':
        f_res = cube[1].data[lam_i:lam_f,:,:]-cube[5].data[lam_i:lam_f,:,:]
    if SL_flag == 'no':
        f_res = cube[1].data[lam_i:lam_f,:,:]
    lam_r = lam[lam_i:lam_f]
    
    return params,f_res,lam_r,SN

def ha_n2(gmodel,lam,cube,SN,c,results_dir,aut_ini):
    
    params = gmodel.make_params()

    params['a'].value = init_params['a']
    params['b'].value = init_params['b']
    params['flux1'].value = init_params['flux']
    params['flux1'].min = init_params['flux_min']
    params['flux2'].value = init_params['flux']
    params['flux2'].min = init_params['flux_min']
    params['flux3'].value = init_params['flux'] 
    params['flux3'].min = init_params['flux_min']
    params['vel1'].value = init_params['vel']
    params['vel2'].value = init_params['vel']
    params['vel3'].value = init_params['vel']
    params['sig1'].value = init_params['sig']
    params['sig2'].value = init_params['sig']
    params['sig3'].value = init_params['sig']
    params['lam01'].value = 6548.04 
    params['lam01'].vary = False
    params['lam02'].value = 6562.80
    params['lam02'].vary = False
    params['lam03'].value = 6583.46
    params['lam03'].vary = False
    
    params['sig1'].min = init_params['sig_min']
    params['sig2'].min = init_params['sig_min']
    params['sig3'].min = init_params['sig_min']
    
    params['sig1'].max = init_params['sig_max']
    params['sig2'].max = init_params['sig_max']
    params['sig3'].max = init_params['sig_max']
    
    if (aut_ini == 'yes'):
        lam_i = (np.abs(lam - (params['vel2']*params['lam02']/c+params['lam02']-70.))).argmin()
        lam_f = (np.abs(lam - (params['vel2']*params['lam02']/c+params['lam02']+70.))).argmin()

        if SL_flag == 'yes':
            f_res = cube[1].data[lam_i:lam_f,:,:]-cube[5].data[lam_i:lam_f,:,:]
        if SL_flag == 'no':
            f_res = cube[1].data[lam_i:lam_f,:,:]
        lam_r = lam[lam_i:lam_f]

        f_res_int_eSN = np.nanmean(f_res*np.exp(SN[np.newaxis,:,:]),axis=(1,2))
        f_res_int_eSN = f_res_int_eSN*np.max(np.nanmean(f_res,axis=(1,2)))/np.max(f_res_int_eSN)

        initial_fit = gmodel.fit(f_res_int_eSN, params, x=lam_r)

        fig = plt.figure(figsize=(15,3))
        initial_fit.plot_fit()
        fig.savefig(results_dir+'/initial_fit.pdf')

        params = initial_fit.params

        params['flux1'].value = init_params['flux']
        params['flux2'].value = init_params['flux']
        params['flux3'].value = init_params['flux']
    
    params['vel1'].min = params['vel1'].value+init_params['vel_min']
    params['vel2'].min = params['vel2'].value+init_params['vel_min']
    params['vel3'].min = params['vel3'].value+init_params['vel_min']
    
    params['vel1'].max = params['vel1'].value+init_params['vel_max']
    params['vel2'].max = params['vel2'].value+init_params['vel_max']
    params['vel3'].max = params['vel3'].value+init_params['vel_max']
    
    lam_i = (np.abs(lam - (params['vel2']*params['lam02']/c+params['lam02']-70.))).argmin()
    lam_f = (np.abs(lam - (params['vel2']*params['lam02']/c+params['lam02']+70.))).argmin()
    if SL_flag == 'yes':
        f_res = cube[1].data[lam_i:lam_f,:,:]-cube[5].data[lam_i:lam_f,:,:]
    if SL_flag == 'no':
        f_res = cube[1].data[lam_i:lam_f,:,:]
    lam_r = lam[lam_i:lam_f]
    
    return params,f_res,lam_r,SN

def ha_n2_cons(gmodel,lam,cube,SN,c,results_dir,aut_ini):
    
    params = gmodel.make_params()

    params['a'].value = init_params['a']
    params['b'].value = init_params['b']
    params['flux'].value = init_params['flux']
    params['flux'].min = init_params['flux_min']
    params['ratio'].value = init_params['n2_ha_ratio']
    params['ratio'].min = init_params['n2_ha_ratio_min']
    params['ratio'].max = init_params['n2_ha_ratio_max']
    params['vel'].value = init_params['vel']
    params['sig'].value = init_params['sig']
    params['lam01'].value = 6548.04
    params['lam01'].vary = False
    params['lam02'].value = 6562.80
    params['lam02'].vary = False
    params['lam03'].value = 6583.46
    params['lam03'].vary = False
    
    params['sig'].min = init_params['sig_min']
    
    params['sig'].max = init_params['sig_max']
    
    if (aut_ini == 'yes'):
        lam_i = (np.abs(lam - (params['vel']*params['lam02']/c+params['lam02']-70.))).argmin()
        lam_f = (np.abs(lam - (params['vel']*params['lam02']/c+params['lam02']+70.))).argmin()

        if SL_flag == 'yes':
            f_res = cube[1].data[lam_i:lam_f,:,:]-cube[5].data[lam_i:lam_f,:,:]
        if SL_flag == 'no':
            f_res = cube[1].data[lam_i:lam_f,:,:]
        lam_r = lam[lam_i:lam_f]

        SN_exp = np.exp(SN[np.newaxis,:,:])
        SN_exp[~np.isfinite(SN_exp)] = np.nan
        f_res_int_eSN = np.nanmean(f_res*SN_exp,axis=(1,2))
        f_res_int_eSN = f_res_int_eSN*np.max(np.nanmean(f_res,axis=(1,2)))/np.max(f_res_int_eSN)

        initial_fit = gmodel.fit(f_res_int_eSN, params, x=lam_r)

        fig = plt.figure(figsize=(15,3))
        initial_fit.plot_fit()
        fig.savefig(results_dir+'/initial_fit.pdf')

        params = initial_fit.params

        params['flux'].value = init_params['flux']
        
    params['vel'].min = params['vel'].value+init_params['vel_min']
    
    params['vel'].max = params['vel'].value+init_params['vel_max']
    
    lam_i = (np.abs(lam - (params['vel']*params['lam02']/c+params['lam02']-70.))).argmin()
    lam_f = (np.abs(lam - (params['vel']*params['lam02']/c+params['lam02']+70.))).argmin()
    if SL_flag == 'yes':
        f_res = cube[1].data[lam_i:lam_f,:,:]-cube[5].data[lam_i:lam_f,:,:]
    if SL_flag == 'no':
        f_res = cube[1].data[lam_i:lam_f,:,:]
    lam_r = lam[lam_i:lam_f]
    
    return params,f_res,lam_r,SN

def ha_n2_2g_cons(gmodel,lam,cube,SN,c,results_dir,aut_ini):
    
    params = gmodel.make_params()

    params['a'].value = init_params['a']
    params['b'].value = init_params['b']
    params['flux'].value = init_params['flux']
    params['flux'].min = init_params['flux_min']
    params['flux_b'].value = init_params['flux']
    params['flux_b'].min = init_params['flux_min']
    params['ratio'].value = init_params['n2_ha_ratio']
    params['ratio'].min = init_params['n2_ha_ratio_min']
    params['ratio'].max = init_params['n2_ha_ratio_max']
    params['vel'].value = init_params['vel']
    params['sig'].value = init_params['sig']
    params['vel_b'].value = init_params['vel']
    params['sig_b'].value = init_params['sig_b']
    params['lam01'].value = 6548.04
    params['lam01'].vary = False
    params['lam02'].value = 6562.80
    params['lam02'].vary = False
    params['lam03'].value = 6583.46
    params['lam03'].vary = False
    
    params['sig'].min = init_params['sig_min']
    params['sig'].max = init_params['sig_max']
    #params['sig_b'].min = init_params['sig_b_min']
    params['sig_b'].set(expr='> sig')
    params['sig_b'].max = init_params['sig_b_max']
    
    if (aut_ini == 'yes'):
        lam_i = (np.abs(lam - (params['vel']*params['lam02']/c+params['lam02']-70.))).argmin()
        lam_f = (np.abs(lam - (params['vel']*params['lam02']/c+params['lam02']+70.))).argmin()

        if SL_flag == 'yes':
            f_res = cube[1].data[lam_i:lam_f,:,:]-cube[5].data[lam_i:lam_f,:,:]
        if SL_flag == 'no':
            f_res = cube[1].data[lam_i:lam_f,:,:]
        lam_r = lam[lam_i:lam_f]

        SN_exp = np.exp(SN[np.newaxis,:,:])
        SN_exp[~np.isfinite(SN_exp)] = np.nan
        f_res_int_eSN = np.nanmean(f_res*SN_exp,axis=(1,2))
        f_res_int_eSN = f_res_int_eSN*np.max(np.nanmean(f_res,axis=(1,2)))/np.max(f_res_int_eSN)

        initial_fit = gmodel.fit(f_res_int_eSN, params, x=lam_r)

        fig = plt.figure(figsize=(15,3))
        initial_fit.plot_fit()
        fig.savefig(results_dir+'/initial_fit.pdf')

        params = initial_fit.params

        params['flux'].value = init_params['flux']
        params['flux_b'].value = init_params['flux']
        
    params['vel'].min = params['vel'].value+init_params['vel_min']
    params['vel'].max = params['vel'].value+init_params['vel_max']
    params['vel_b'].min = params['vel_b'].value+init_params['vel_min']
    params['vel_b'].max = params['vel_b'].value+init_params['vel_max']
    
    lam_i = (np.abs(lam - (params['vel']*params['lam02']/c+params['lam02']-70.))).argmin()
    lam_f = (np.abs(lam - (params['vel']*params['lam02']/c+params['lam02']+70.))).argmin()
    if SL_flag == 'yes':
        f_res = cube[1].data[lam_i:lam_f,:,:]-cube[5].data[lam_i:lam_f,:,:]
    if SL_flag == 'no':
        f_res = cube[1].data[lam_i:lam_f,:,:]
    lam_r = lam[lam_i:lam_f]
    
    return params,f_res,lam_r,SN

def hb_ha_n2_cons(gmodel,lam,cube,SN,c,results_dir,aut_ini):
    
    params = gmodel.make_params()

    params['a_hb'].value = init_params['a']
    params['a_ha'].value = init_params['a']
    params['b_hb'].value = init_params['b']
    params['b_ha'].value = init_params['b']
    params['flux'].value = init_params['flux']
    params['flux'].min = init_params['flux_min']
    params['ratio_hb'].value = init_params['ha_hb_ratio']
    params['ratio_hb'].min = init_params['ha_hb_ratio_min']
    params['ratio_hb'].max = init_params['ha_hb_ratio_max']
    params['ratio_n2'].value = init_params['n2_ha_ratio']
    params['ratio_n2'].min = init_params['n2_ha_ratio_min']
    params['ratio_n2'].max = init_params['n2_ha_ratio_max']
    params['vel'].value = init_params['vel']
    params['sig'].value = init_params['sig']
    params['lam0_hb'].value = 4861.325
    params['lam0_hb'].vary = False
    params['lam01'].value = 6548.04
    params['lam01'].vary = False
    params['lam02'].value = 6562.80
    params['lam02'].vary = False
    params['lam03'].value = 6583.46
    params['lam03'].vary = False
    
    params['sig'].min = init_params['sig_min']
    
    params['sig'].max = init_params['sig_max']
    
    if (aut_ini == 'yes'):
        lam_i_hb = (np.abs(lam - (params['vel']*params['lam0_hb']/c+params['lam0_hb']-70.))).argmin()
        lam_f_hb = (np.abs(lam - (params['vel']*params['lam0_hb']/c+params['lam0_hb']+70.))).argmin()
        lam_i_ha = (np.abs(lam - (params['vel']*params['lam02']/c+params['lam02']-70.))).argmin()
        lam_f_ha = (np.abs(lam - (params['vel']*params['lam02']/c+params['lam02']+70.))).argmin()

        if SL_flag == 'yes':
            f_res = np.concatenate((cube[1].data[lam_i_hb:lam_f_hb,:,:]-cube[5].data[lam_i_hb:lam_f_hb,:,:],cube[1].data[lam_i_ha:lam_f_ha,:,:]-cube[5].data[lam_i_ha:lam_f_ha,:,:]))
        if SL_flag == 'no':
            f_res = np.concatenate((cube[1].data[lam_i_hb:lam_f_hb,:,:],cube[1].data[lam_i_ha:lam_f_ha,:,:]))
        lam_r = np.concatenate((lam[lam_i_hb:lam_f_hb],lam[lam_i_ha:lam_f_ha]))

        f_res_int_eSN = np.nanmean(f_res*np.exp(SN[np.newaxis,:,:]),axis=(1,2))
        f_res_int_eSN = f_res_int_eSN*np.max(np.nanmean(f_res,axis=(1,2)))/np.max(f_res_int_eSN)

        initial_fit = gmodel.fit(f_res_int_eSN, params, x=lam_r)

        fig = plt.figure(figsize=(15,3))
        initial_fit.plot_fit()
        fig.savefig(results_dir+'/initial_fit.pdf')

        params = initial_fit.params

        params['flux'].value = init_params['flux']
        
    params['vel'].min = params['vel'].value+init_params['vel_min']
    
    params['vel'].max = params['vel'].value+init_params['vel_max']
    
    lam_i_hb = (np.abs(lam - (params['vel']*params['lam0_hb']/c+params['lam0_hb']-70.))).argmin()
    lam_f_hb = (np.abs(lam - (params['vel']*params['lam0_hb']/c+params['lam0_hb']+70.))).argmin()
    lam_i_ha = (np.abs(lam - (params['vel']*params['lam02']/c+params['lam02']-70.))).argmin()
    lam_f_ha = (np.abs(lam - (params['vel']*params['lam02']/c+params['lam02']+70.))).argmin()
    if SL_flag == 'yes':
        f_res = np.concatenate((cube[1].data[lam_i_hb:lam_f_hb,:,:]-cube[5].data[lam_i_hb:lam_f_hb,:,:],cube[1].data[lam_i_ha:lam_f_ha,:,:]-cube[5].data[lam_i_ha:lam_f_ha,:,:]))
    if SL_flag == 'no':
        f_res = np.concatenate((cube[1].data[lam_i_hb:lam_f_hb,:,:],cube[1].data[lam_i_ha:lam_f_ha,:,:]))
    lam_r = np.concatenate((lam[lam_i_hb:lam_f_hb],lam[lam_i_ha:lam_f_ha]))
    
    return params,f_res,lam_r,SN

def s2(gmodel,lam,cube,SN,c,results_dir,aut_ini):
    
    params = gmodel.make_params()

    params['a'].value = init_params['a'] 
    params['b'].value = init_params['b']
    params['flux1'].value = init_params['flux']
    params['flux1'].min = init_params['flux_min']
    params['flux2'].value = init_params['flux']
    params['flux2'].min = init_params['flux_min']
    params['vel1'].value = init_params['vel']
    params['vel2'].value = init_params['vel'] 
    params['sig1'].value = init_params['sig']
    params['sig2'].value = init_params['sig'] 
    params['lam01'].value = 6716.44
    params['lam01'].vary = False
    params['lam02'].value = 6730.81
    params['lam02'].vary = False
    
    params['sig1'].min = init_params['sig_min']
    params['sig2'].min = init_params['sig_min']
    
    params['sig1'].max = init_params['sig_max']
    params['sig2'].max = init_params['sig_max']
    
    if (aut_ini == 'yes'):
        lam_i = (np.abs(lam - ( (((params['vel1']*params['lam01'])+(params['vel2']*params['lam02']))/2.) /c+ ((params['lam01']+params['lam02'])/2.) -70.))).argmin()
        lam_f = (np.abs(lam - ( (((params['vel1']*params['lam01'])+(params['vel2']*params['lam02']))/2.) /c+ ((params['lam01']+params['lam02'])/2.) +70.))).argmin()
        
        if SL_flag == 'yes':
            f_res = cube[1].data[lam_i:lam_f,:,:]-cube[5].data[lam_i:lam_f,:,:]
        if SL_flag == 'no':
            f_res = cube[1].data[lam_i:lam_f,:,:]
        lam_r = lam[lam_i:lam_f]
        
        f_res_int_eSN = np.nanmean(f_res*np.exp(SN[np.newaxis,:,:]),axis=(1,2))
        f_res_int_eSN = f_res_int_eSN*np.max(np.nanmean(f_res,axis=(1,2)))/np.max(f_res_int_eSN)
        
        initial_fit = gmodel.fit(f_res_int_eSN, params, x=lam_r)
        
        fig = plt.figure(figsize=(15,3))
        initial_fit.plot_fit()
        fig.savefig(results_dir+'/initial_fit.pdf')
        
        params = initial_fit.params
        
        params['flux1'].value = init_params['flux']
        params['flux2'].value = init_params['flux']
    
    params['vel1'].min = params['vel1'].value+init_params['vel_min']
    params['vel2'].min = params['vel2'].value+init_params['vel_min']
    
    params['vel1'].max = params['vel1'].value+init_params['vel_max']
    params['vel2'].max = params['vel2'].value+init_params['vel_max']

    
    lam_i = (np.abs(lam - ( (((params['vel1']*params['lam01'])+(params['vel2']*params['lam02']))/2.) /c+ ((params['lam01']+params['lam02'])/2.) -70.))).argmin()
    lam_f = (np.abs(lam - ( (((params['vel1']*params['lam01'])+(params['vel2']*params['lam02']))/2.) /c+ ((params['lam01']+params['lam02'])/2.) +70.))).argmin()
    if SL_flag == 'yes':
        f_res = cube[1].data[lam_i:lam_f,:,:]-cube[5].data[lam_i:lam_f,:,:]
    if SL_flag == 'no':
        f_res = cube[1].data[lam_i:lam_f,:,:]
    lam_r = lam[lam_i:lam_f]
    
    return params,f_res,lam_r,SN

def s2_cons(gmodel,lam,cube,SN,c,results_dir,aut_ini):
    
    params = gmodel.make_params()

    params['a'].value = init_params['a'] 
    params['b'].value = init_params['b']
    params['flux'].value = init_params['flux']
    params['flux'].min = init_params['flux_min']
    params['ratio'].value = init_params['s2_ratio']
    params['ratio'].min = init_params['s2_ratio_min']
    params['ratio'].max = init_params['s2_ratio_max']
    params['vel'].value = init_params['vel']
    params['sig'].value = init_params['sig']
    params['lam01'].value = 6716.44
    params['lam01'].vary = False
    params['lam02'].value = 6730.81
    params['lam02'].vary = False
    
    params['sig'].min = init_params['sig_min']
    
    params['sig'].max = init_params['sig_max']
    
    if (aut_ini == 'yes'):
        lam_i = (np.abs(lam - ( (((params['vel']*params['lam01'])+(params['vel']*params['lam02']))/2.) /c+ ((params['lam01']+params['lam02'])/2.) -70.))).argmin()
        lam_f = (np.abs(lam - ( (((params['vel']*params['lam01'])+(params['vel']*params['lam02']))/2.) /c+ ((params['lam01']+params['lam02'])/2.) +70.))).argmin()
        
        if SL_flag == 'yes':
            f_res = cube[1].data[lam_i:lam_f,:,:]-cube[5].data[lam_i:lam_f,:,:]
        if SL_flag == 'no':
            f_res = cube[1].data[lam_i:lam_f,:,:]
        lam_r = lam[lam_i:lam_f]
        
        f_res_int_eSN = np.nanmean(f_res*np.exp(SN[np.newaxis,:,:]),axis=(1,2))
        f_res_int_eSN = f_res_int_eSN*np.max(np.nanmean(f_res,axis=(1,2)))/np.max(f_res_int_eSN)
        
        initial_fit = gmodel.fit(f_res_int_eSN, params, x=lam_r)
        
        fig = plt.figure(figsize=(15,3))
        initial_fit.plot_fit()
        fig.savefig(results_dir+'/initial_fit.pdf')
        
        params = initial_fit.params
        
        params['flux'].value = init_params['flux']
    
    params['vel'].min = params['vel'].value+init_params['vel_min']
    
    params['vel'].max = params['vel'].value+init_params['vel_max']
    
    lam_i = (np.abs(lam - ( (((params['vel']*params['lam01'])+(params['vel']*params['lam02']))/2.) /c+ ((params['lam01']+params['lam02'])/2.) -70.))).argmin()
    lam_f = (np.abs(lam - ( (((params['vel']*params['lam01'])+(params['vel']*params['lam02']))/2.) /c+ ((params['lam01']+params['lam02'])/2.) +70.))).argmin()
    if SL_flag == 'yes':
        f_res = cube[1].data[lam_i:lam_f,:,:]-cube[5].data[lam_i:lam_f,:,:]
    if SL_flag == 'no':
        f_res = cube[1].data[lam_i:lam_f,:,:]
    lam_r = lam[lam_i:lam_f]
    
    return params,f_res,lam_r,SN
