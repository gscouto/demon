import numpy as np
from astropy.io import fits
from line_fit_config import *
from lmfit import Model, fit_report
import fit_functions
import matplotlib.pyplot as plt
import os 
import multiprocessing as mp

def two_gaussians_cons_o3(naxis1,naxis2,result_list,results_dir,lam_r,params):

    results = np.reshape(result_list, (naxis2,naxis1))

    ncomps_flag = np.zeros([naxis2,naxis1])
    bic_flag = np.zeros([naxis2,naxis1])

    for i in np.arange(naxis1):
        for j in np.arange(naxis2):
        
            res_max_lim = 5*np.std(np.array(results[j,i].data)[np.r_[0:20,len(lam_r)-20:len(lam_r)]])
            res_min_lim = -5*np.std(np.array(results[j,i].data)[np.r_[0:20,len(lam_r)-20:len(lam_r)]])

            gaussian_region = (results[j,i].best_fit > results[j,i].params['a'].value+(lam_r*results[j,i].params['b'].value))*~np.isclose(results[j,i].best_fit,results[j,i].params['a'].value+(lam_r*results[j,i].params['b'].value),rtol=1e-4)
            
            r_max = ((results[j,i].residual > res_max_lim)*gaussian_region)
            r_min = ((results[j,i].residual < res_min_lim)*gaussian_region)

            if (any(r_max[k]==r_max[k+1]==True for k in range(len(r_max)-1)) | any(r_min[k]==r_min[k+1]==True for k in range(len(r_min)-1))) and (results[j,i].params['flux'].stderr):                
                if (results[j,i].params['flux'].stderr/results[j,i].params['flux'].value < 0.5) & (results[j,i].params['sig'].stderr < 20) & (results[j,i].params['vel'].stderr < 20):
                    ncomps_flag[j,i] = 1
                    bic_flag[j,i] = 1
                    
                    os.makedirs(results_dir+'/more_comps_plots', exist_ok=True)
                    
                    fig = plt.figure(figsize=(7,3))
                    
                    plt.plot((results[j,i].data*0.)+res_max_lim,color='gray')
                    plt.plot((results[j,i].data*0.)+res_min_lim,color='gray')
                    plt.plot(results[j,i].data,color='black')
                    plt.plot(results[j,i].residual,color='blue')
                    plt.plot(results[j,i].best_fit,color='red')
                    plt.savefig(results_dir+'/more_comps_plots/fit_'+str(i)+'_'+str(j)+'.pdf',overwrite=True)   
                    
                    plt.close()
    
    gmodel = Model(fit_functions.two_gaussians_2g_cons_o3)
    
    nparams = gmodel.make_params()
    
    nparams['a'].value = init_params['a']
    nparams['b'].value = init_params['b']
    nparams['flux'].value = init_params['flux']
    nparams['flux'].min = init_params['flux_min']
    nparams['flux_b'].value = init_params['flux']
    nparams['flux_b'].min = init_params['flux_min']
    nparams.add('flux_delta',value=init_params['flux_delta'],min=init_params['flux_delta_min'],max=init_params['flux_delta_max'],vary=True)
    nparams['flux_b'].set(expr='flux/flux_delta')
    
    nparams['vel'].value = init_params['vel']
    nparams['sig'].value = init_params['sig']
    nparams['lam01'].value = 4958.911
    nparams['lam01'].vary = False
    nparams['lam02'].value = 5006.843
    nparams['lam02'].vary = False
    
    nparams['sig'].min = init_params['sig_min']
    nparams['sig'].max = init_params['sig_max']
    nparams.add('sig_delta',value=init_params['sig_delta'],min=init_params['sig_delta_min'],max=init_params['sig_delta_max'],vary=True)
    nparams['sig_b'].set(expr='sig_delta*sig')
    
    nparams['vel'].min = init_params['vel_min']
    nparams['vel'].max = init_params['vel_max']
    nparams.add('vel_delta',value=init_params['vel_delta'],min=init_params['vel_delta_min'],max=init_params['vel_delta_max'],vary=True)
    nparams['vel_b'].set(expr='vel+vel_delta')
    
    y_mask, x_mask = np.where(ncomps_flag == 1)
        
    global kloop
        
    def kloop(args):
        
        results_data, lam_r, nparams, k, l = args

        print("Progress {:2.1%}".format(float(k)/float(l)), end="\r")
        
        result = gmodel.fit(results_data, nparams, x=lam_r)
        
        return (result)
        
    with mp.Pool(n_proc) as pool:
        results_k = pool.starmap(kloop, zip((results[y_mask[k],x_mask[k]].data, lam_r, nparams, k, len(y_mask)) for k in np.arange(len(y_mask))))        

    for k in np.arange(len(y_mask)):
        if (results_k[k].bic < results[y_mask[k],x_mask[k]].bic):
            results[y_mask[k],x_mask[k]] = results_k[k]
            bic_flag[y_mask[k],x_mask[k]] = 2
        
    result_list = np.ravel(results)
        
    return result_list, ncomps_flag, bic_flag

def two_gaussians_cons_o1(naxis1,naxis2,result_list,results_dir,lam_r,params):

    results = np.reshape(result_list, (naxis2,naxis1))

    ncomps_flag = np.zeros([naxis2,naxis1])
    bic_flag = np.zeros([naxis2,naxis1])

    for i in np.arange(naxis1):
        for j in np.arange(naxis2):
        
            res_max_lim = 5*np.std(np.array(results[j,i].data)[np.r_[0:20,len(lam_r)-20:len(lam_r)]])
            res_min_lim = -5*np.std(np.array(results[j,i].data)[np.r_[0:20,len(lam_r)-20:len(lam_r)]])

            gaussian_region = (results[j,i].best_fit > results[j,i].params['a'].value+(lam_r*results[j,i].params['b'].value))*~np.isclose(results[j,i].best_fit,results[j,i].params['a'].value+(lam_r*results[j,i].params['b'].value),rtol=1e-4)
            
            r_max = ((results[j,i].residual > res_max_lim)*gaussian_region)
            r_min = ((results[j,i].residual < res_min_lim)*gaussian_region)

            if (any(r_max[k]==r_max[k+1]==True for k in range(len(r_max)-1)) | any(r_min[k]==r_min[k+1]==True for k in range(len(r_min)-1))) and (results[j,i].params['flux'].stderr):                    
                if (results[j,i].params['flux'].stderr/results[j,i].params['flux'].value < 0.5) & (results[j,i].params['sig'].stderr < 20) & (results[j,i].params['vel'].stderr < 20):
                    ncomps_flag[j,i] = 1
                    bic_flag[j,i] = 1
                    
                    os.makedirs(results_dir+'/more_comps_plots', exist_ok=True)
                    
                    fig = plt.figure(figsize=(7,3))
                    
                    plt.plot((results[j,i].data*0.)+res_max_lim,color='gray')
                    plt.plot((results[j,i].data*0.)+res_min_lim,color='gray')
                    plt.plot(results[j,i].data,color='black')
                    plt.plot(results[j,i].residual,color='blue')
                    plt.plot(results[j,i].best_fit,color='red')
                    plt.savefig(results_dir+'/more_comps_plots/fit_'+str(i)+'_'+str(j)+'.pdf',overwrite=True)   
                    
                    plt.close()
    
    gmodel = Model(fit_functions.two_gaussians_2g_cons_o1)
    
    nparams = gmodel.make_params()
    
    nparams['a'].value = init_params['a']
    nparams['b'].value = init_params['b']
    nparams['flux'].value = init_params['flux']
    nparams['flux'].min = init_params['flux_min']
    nparams['flux_b'].value = init_params['flux']
    nparams['flux_b'].min = init_params['flux_min']
    nparams.add('flux_delta',value=init_params['flux_delta'],min=init_params['flux_delta_min'],max=init_params['flux_delta_max'],vary=True)
    nparams['flux_b'].set(expr='flux/flux_delta')
    
    nparams['vel'].value = init_params['vel']
    nparams['sig'].value = init_params['sig']
    nparams['lam01'].value = 6300.304
    nparams['lam01'].vary = False
    nparams['lam02'].value = 6363.777
    nparams['lam02'].vary = False
    
    nparams['sig'].min = init_params['sig_min']
    nparams['sig'].max = init_params['sig_max']
    nparams.add('sig_delta',value=init_params['sig_delta'],min=init_params['sig_delta_min'],max=init_params['sig_delta_max'],vary=True)
    nparams['sig_b'].set(expr='sig_delta*sig')
    
    nparams['vel'].min = init_params['vel_min']
    nparams['vel'].max = init_params['vel_max']
    nparams.add('vel_delta',value=init_params['vel_delta'],min=init_params['vel_delta_min'],max=init_params['vel_delta_max'],vary=True)
    nparams['vel_b'].set(expr='vel+vel_delta')
    
    y_mask, x_mask = np.where(ncomps_flag == 1)
        
    global kloop
        
    def kloop(args):
        
        results_data, lam_r, nparams, k, l = args

        print("Progress {:2.1%}".format(float(k)/float(l)), end="\r")
        
        result = gmodel.fit(results_data, nparams, x=lam_r)
        
        return (result)
        
    with mp.Pool(n_proc) as pool:
        results_k = pool.starmap(kloop, zip((results[y_mask[k],x_mask[k]].data, lam_r, nparams, k, len(y_mask)) for k in np.arange(len(y_mask))))        

    for k in np.arange(len(y_mask)):
        if (results_k[k].bic < results[y_mask[k],x_mask[k]].bic):
            results[y_mask[k],x_mask[k]] = results_k[k]
            bic_flag[y_mask[k],x_mask[k]] = 2
        
    result_list = np.ravel(results)
        
    return result_list, ncomps_flag, bic_flag

def two_gaussians_cons_s2(naxis1,naxis2,result_list,results_dir,lam_r,params):

    results = np.reshape(result_list, (naxis2,naxis1))

    ncomps_flag = np.zeros([naxis2,naxis1])
    bic_flag = np.zeros([naxis2,naxis1])

    for i in np.arange(naxis1):
        for j in np.arange(naxis2):
        
            res_max_lim = 5*np.std(np.array(results[j,i].data)[np.r_[0:20,len(lam_r)-20:len(lam_r)]])
            res_min_lim = -5*np.std(np.array(results[j,i].data)[np.r_[0:20,len(lam_r)-20:len(lam_r)]])

            gaussian_region = (results[j,i].best_fit > results[j,i].params['a'].value+(lam_r*results[j,i].params['b'].value))*~np.isclose(results[j,i].best_fit,results[j,i].params['a'].value+(lam_r*results[j,i].params['b'].value),rtol=1e-4)
            
            r_max = ((results[j,i].residual > res_max_lim)*gaussian_region)
            r_min = ((results[j,i].residual < res_min_lim)*gaussian_region)

            if (any(r_max[k]==r_max[k+1]==True for k in range(len(r_max)-1)) | any(r_min[k]==r_min[k+1]==True for k in range(len(r_min)-1))) and (results[j,i].params['flux'].stderr):            
                if (results[j,i].params['flux'].stderr/results[j,i].params['flux'].value < 0.5) & (results[j,i].params['sig'].stderr < 20) & (results[j,i].params['vel'].stderr < 20):
                    ncomps_flag[j,i] = 1
                    bic_flag[j,i] = 1
                    
                    os.makedirs(results_dir+'/more_comps_plots', exist_ok=True)
                    
                    fig = plt.figure(figsize=(7,3))
                    
                    plt.plot((results[j,i].data*0.)+res_max_lim,color='gray')
                    plt.plot((results[j,i].data*0.)+res_min_lim,color='gray')
                    plt.plot(results[j,i].data,color='black')
                    plt.plot(results[j,i].residual,color='blue')
                    plt.plot(results[j,i].best_fit,color='red')
                    plt.savefig(results_dir+'/more_comps_plots/fit_'+str(i)+'_'+str(j)+'.pdf',overwrite=True)   
                    
                    plt.close()
    
    gmodel = Model(fit_functions.two_gaussians_2g_cons_s2)
    
    nparams = gmodel.make_params()
    
    nparams['a'].value = init_params['a']
    nparams['b'].value = init_params['b']
    nparams['flux'].value = init_params['flux']
    nparams['flux'].min = init_params['flux_min']
    nparams['flux_b'].value = init_params['flux']
    nparams['flux_b'].min = init_params['flux_min']
    nparams.add('flux_delta',value=init_params['flux_delta'],min=init_params['flux_delta_min'],max=init_params['flux_delta_max'],vary=True)
    nparams['flux_b'].set(expr='flux/flux_delta')
    
    nparams['ratio'].value = init_params['s2_ratio']
    nparams['ratio'].min = init_params['s2_ratio_min']
    nparams['ratio'].max = init_params['s2_ratio_max']
    nparams['ratio_b'].value = init_params['s2_ratio']
    nparams['ratio_b'].min = init_params['s2_ratio_min']
    nparams['ratio_b'].max = init_params['s2_ratio_max']
    nparams['vel'].value = init_params['vel']
    nparams['sig'].value = init_params['sig']
    nparams['lam01'].value = 6716.44
    nparams['lam01'].vary = False
    nparams['lam02'].value = 6730.81
    nparams['lam02'].vary = False
    
    nparams['sig'].min = init_params['sig_min']
    nparams['sig'].max = init_params['sig_max']
    nparams.add('sig_delta',value=init_params['sig_delta'],min=init_params['sig_delta_min'],max=init_params['sig_delta_max'],vary=True)
    nparams['sig_b'].set(expr='sig_delta*sig')
    
    nparams['vel'].min = init_params['vel_min']
    nparams['vel'].max = init_params['vel_max']
    nparams.add('vel_delta',value=init_params['vel_delta'],min=init_params['vel_delta_min'],max=init_params['vel_delta_max'],vary=True)
    nparams['vel_b'].set(expr='vel+vel_delta')
    
    y_mask, x_mask = np.where(ncomps_flag == 1)
        
    global kloop
        
    def kloop(args):
        
        results_data, lam_r, nparams, k, l = args

        print("Progress {:2.1%}".format(float(k)/float(l)), end="\r")
        
        result = gmodel.fit(results_data, nparams, x=lam_r)
        
        return (result)
        
    with mp.Pool(n_proc) as pool:
        results_k = pool.starmap(kloop, zip((results[y_mask[k],x_mask[k]].data, lam_r, nparams, k, len(y_mask)) for k in np.arange(len(y_mask))))        

    for k in np.arange(len(y_mask)):
        if (results_k[k].bic < results[y_mask[k],x_mask[k]].bic):
            results[y_mask[k],x_mask[k]] = results_k[k]
            bic_flag[y_mask[k],x_mask[k]] = 2
        
    result_list = np.ravel(results)
        
    return result_list, ncomps_flag, bic_flag

def three_gaussians_cons(naxis1,naxis2,result_list,results_dir,lam_r,params):

    results = np.reshape(result_list, (naxis2,naxis1))

    ncomps_flag = np.zeros([naxis2,naxis1])
    bic_flag = np.zeros([naxis2,naxis1])

    for i in np.arange(naxis1):
        for j in np.arange(naxis2):
        
            res_max_lim = 5*np.std(np.array(results[j,i].data)[np.r_[0:20,len(lam_r)-20:len(lam_r)]])
            res_min_lim = -5*np.std(np.array(results[j,i].data)[np.r_[0:20,len(lam_r)-20:len(lam_r)]])
            
            line_peak = np.where(results[j,i].best_fit == np.nanmax(results[j,i].best_fit))[0][0]

            gaussian_region = (results[j,i].best_fit > results[j,i].params['a'].value+(lam_r*results[j,i].params['b'].value))*~np.isclose(results[j,i].best_fit,results[j,i].params['a'].value+(lam_r*results[j,i].params['b'].value),rtol=1e-4)
            
            r_max = ((results[j,i].residual > res_max_lim)*gaussian_region)[line_peak-20:line_peak+20]
            r_min = ((results[j,i].residual < res_min_lim)*gaussian_region)[line_peak-20:line_peak+20]

            if (any(r_max[k]==r_max[k+1]==True for k in range(len(r_max)-1)) | any(r_min[k]==r_min[k+1]==True for k in range(len(r_min)-1))) and (results[j,i].params['flux'].stderr):
                if (results[j,i].params['flux'].stderr/results[j,i].params['flux'].value < 0.5) & (results[j,i].params['sig'].stderr < 20) & (results[j,i].params['vel'].stderr < 20):
                    ncomps_flag[j,i] = 1
                    bic_flag[j,i] = 1
                    
                    os.makedirs(results_dir+'/more_comps_plots', exist_ok=True)
                    
                    fig = plt.figure(figsize=(7,3))
                    
                    plt.plot((results[j,i].data*0.)+res_max_lim,color='gray')
                    plt.plot((results[j,i].data*0.)+res_min_lim,color='gray')
                    plt.plot(results[j,i].data,color='black')
                    plt.plot(results[j,i].residual,color='blue')
                    plt.plot(results[j,i].best_fit,color='red')
                    plt.savefig(results_dir+'/more_comps_plots/fit_'+str(i)+'_'+str(j)+'.pdf',overwrite=True)   
                    
                    plt.close()
    
    gmodel = Model(fit_functions.three_gaussians_2g_cons)
    
    nparams = gmodel.make_params()
    
    nparams['a'].value = init_params['a']
    nparams['b'].value = init_params['b']
    nparams['flux'].value = init_params['flux']
    nparams['flux'].min = init_params['flux_min']
    nparams['flux_b'].value = init_params['flux']
    nparams['flux_b'].min = init_params['flux_min']
    nparams.add('flux_delta',value=init_params['flux_delta'],min=init_params['flux_delta_min'],max=init_params['flux_delta_max'],vary=True)
    nparams['flux_b'].set(expr='flux/flux_delta')
    
    nparams['ratio'].value = init_params['n2_ha_ratio']
    nparams['ratio'].min = init_params['n2_ha_ratio_min']
    nparams['ratio'].max = init_params['n2_ha_ratio_max']
    nparams['ratio_b'].value = init_params['n2_ha_ratio']
    nparams['ratio_b'].min = init_params['n2_ha_ratio_min']
    nparams['ratio_b'].max = init_params['n2_ha_ratio_max']
    nparams['vel'].value = init_params['vel']
    nparams['sig'].value = init_params['sig']
    nparams['lam01'].value = 6548.04
    nparams['lam01'].vary = False
    nparams['lam02'].value = 6562.80
    nparams['lam02'].vary = False
    nparams['lam03'].value = 6583.46
    nparams['lam03'].vary = False
    
    nparams['sig'].min = init_params['sig_min']
    nparams['sig'].max = init_params['sig_max']
    nparams.add('sig_delta',value=init_params['sig_delta'],min=init_params['sig_delta_min'],max=init_params['sig_delta_max'],vary=True)
    nparams['sig_b'].set(expr='sig_delta*sig')
    
    nparams['vel'].min = init_params['vel_min']
    nparams['vel'].max = init_params['vel_max']
    nparams.add('vel_delta',value=init_params['vel_delta'],min=init_params['vel_delta_min'],max=init_params['vel_delta_max'],vary=True)
    nparams['vel_b'].set(expr='vel+vel_delta')
    
    y_mask, x_mask = np.where(ncomps_flag == 1)
        
    global kloop
        
    def kloop(args):
        
        results_data, lam_r, nparams, k, l = args

        print("Progress {:2.1%}".format(float(k)/float(l)), end="\r")
        
        result = gmodel.fit(results_data, nparams, x=lam_r)
        
        return (result)
        
    with mp.Pool(n_proc) as pool:
        results_k = pool.starmap(kloop, zip((results[y_mask[k],x_mask[k]].data, lam_r, nparams, k, len(y_mask)) for k in np.arange(len(y_mask))))        

    for k in np.arange(len(y_mask)):
        if (results_k[k].bic < results[y_mask[k],x_mask[k]].bic):
            results[y_mask[k],x_mask[k]] = results_k[k]
            bic_flag[y_mask[k],x_mask[k]] = 2
        
    result_list = np.ravel(results)
        
    return result_list, ncomps_flag, bic_flag


def hb_ha_n2_gaussians_cons(naxis1,naxis2,result_list,results_dir,lam_r,params):

    results = np.reshape(result_list, (naxis2,naxis1))

    ncomps_flag = np.zeros([naxis2,naxis1])
    bic_flag = np.zeros([naxis2,naxis1])

    for i in np.arange(naxis1):
        for j in np.arange(naxis2):
        
            #res_max_lim = np.mean(np.array(results[j,i].data)[np.r_[int(len(lam_r)/2):int(len(lam_r)/2)+20,len(lam_r)-20:len(lam_r)]])+5*np.std(np.array(results[j,i].data)[np.r_[int(len(lam_r)/2):int(len(lam_r)/2)+20,len(lam_r)-20:len(lam_r)]])        
            res_max_lim = 5*np.std(np.array(results[j,i].data)[np.r_[int(len(lam_r)/2):int(len(lam_r)/2)+20,len(lam_r)-20:len(lam_r)]])
            res_min_lim = -5*np.std(np.array(results[j,i].data)[np.r_[int(len(lam_r)/2):int(len(lam_r)/2)+20,len(lam_r)-20:len(lam_r)]])
            
            #if (i == 197) & (j == 194):
                #print(any(results[j,i].residual > res_max_lim))
                #print(any(results[j,i].residual < res_min_lim))
                #print(results[j,i].params['flux'].stderr)
                #print(results[j,i].params['flux'].stderr/results[j,i].params['flux'].value)
                #print(results[j,i].params['sig'].stderr < 20)
                #print(results[j,i].params['vel'].stderr < 20)
                
                #i = 171
                #j = 161
                
                #plt.plot(lam_r[int(len(lam_r)/2):],results[j,i].params['a_ha'].value+(lam_r[int(len(lam_r)/2):]*results[j,i].params['b_ha'].value),color='blue')
                #plt.plot(lam_r[int(len(lam_r)/2):],results[j,i].best_fit[int(len(lam_r)/2):],color='red')
                #plt.savefig('teste.pdf')

                #plt.plot((results[j,i].data*0.)+res_max_lim,color='gray')
                #plt.plot((results[j,i].data*0.)+res_min_lim,color='gray')
                #plt.plot(results[j,i].data,color='black')
                #plt.plot(results[j,i].residual,color='blue')
                #plt.plot(results[j,i].best_fit,color='red')
                #plt.xlim([int(len(lam_r)/2),len(lam_r)])
                #plt.savefig('teste.pdf')         

            line_peak = np.where(results[j,i].best_fit == np.nanmax(results[j,i].best_fit))[0][0]
            
            gaussian_region = (results[j,i].best_fit > results[j,i].params['a_ha'].value+(lam_r*results[j,i].params['b_ha'].value))*~np.isclose(results[j,i].best_fit,results[j,i].params['a_ha'].value+(lam_r*results[j,i].params['b_ha'].value),rtol=1e-4)
            
            r_max = ((results[j,i].residual > res_max_lim)*gaussian_region)[line_peak-20:line_peak+20]
            r_min = ((results[j,i].residual < res_min_lim)*gaussian_region)[line_peak-20:line_peak+20]

            if (any(r_max[k]==r_max[k+1]==True for k in range(len(r_max)-1)) | any(r_min[k]==r_min[k+1]==True for k in range(len(r_min)-1))) and (results[j,i].params['flux'].stderr):
                if (results[j,i].params['flux'].stderr/results[j,i].params['flux'].value < 0.5) & (results[j,i].params['sig'].stderr < 20) & (results[j,i].params['vel'].stderr < 20):
                    ncomps_flag[j,i] = 1
                    bic_flag[j,i] = 1
                    
                    os.makedirs(results_dir+'/more_comps_plots', exist_ok=True)
                    
                    fig = plt.figure(figsize=(7,3))
                    
                    plt.plot((results[j,i].data*0.)+res_max_lim,color='gray')
                    plt.plot((results[j,i].data*0.)+res_min_lim,color='gray')
                    plt.plot(results[j,i].data,color='black')
                    plt.plot(results[j,i].residual,color='blue')
                    plt.plot(results[j,i].best_fit,color='red')
                    plt.xlim([int(len(lam_r)/2),len(lam_r)])
                    plt.savefig(results_dir+'/more_comps_plots/fit_'+str(i)+'_'+str(j)+'.pdf',overwrite=True)   
                    
                    plt.close()
    
    gmodel = Model(fit_functions.hb_ha_n2_gaussians_2g_cons)
    
    nparams = gmodel.make_params()
    
    nparams['a_hb'].value = init_params['a']
    nparams['b_hb'].value = init_params['b']
    nparams['a_ha'].value = init_params['a']
    nparams['b_ha'].value = init_params['b']
    nparams['flux'].value = init_params['flux']
    nparams['flux'].min = init_params['flux_min']
    nparams['flux_b'].value = init_params['flux']
    nparams['flux_b'].min = init_params['flux_min']
    nparams.add('flux_delta',value=init_params['flux_delta'],min=init_params['flux_delta_min'],max=init_params['flux_delta_max'],vary=True)
    nparams['flux_b'].set(expr='flux/flux_delta')
    
    nparams['ratio_n2'].value = init_params['n2_ha_ratio']
    nparams['ratio_n2'].min = init_params['n2_ha_ratio_min']
    nparams['ratio_n2'].max = init_params['n2_ha_ratio_max']
    nparams['ratio_n2_b'].value = init_params['n2_ha_ratio']
    nparams['ratio_n2_b'].min = init_params['n2_ha_ratio_min']
    nparams['ratio_n2_b'].max = init_params['n2_ha_ratio_max']
    nparams['ratio_hb'].value = init_params['ha_hb_ratio']
    nparams['ratio_hb'].min = init_params['ha_hb_ratio_min']
    nparams['ratio_hb'].max = init_params['ha_hb_ratio_max']
    nparams['ratio_hb_b'].value = init_params['ha_hb_ratio']
    nparams['ratio_hb_b'].min = init_params['ha_hb_ratio_min']
    nparams['ratio_hb_b'].max = init_params['ha_hb_ratio_max']
    nparams['ratio_hb_b'].set(expr='ratio_hb')
    nparams['ratio_n2_b'].set(expr='ratio_n2')
    
    nparams['vel'].value = init_params['vel']
    nparams['sig'].value = init_params['sig']
    nparams['lam0_hb'].value = 4861.325
    nparams['lam0_hb'].vary = False
    nparams['lam01'].value = 6548.04
    nparams['lam01'].vary = False
    nparams['lam02'].value = 6562.80
    nparams['lam02'].vary = False
    nparams['lam03'].value = 6583.46
    nparams['lam03'].vary = False
    
    nparams['sig'].min = init_params['sig_min']
    nparams['sig'].max = init_params['sig_max']
    nparams.add('sig_delta',value=init_params['sig_delta'],min=init_params['sig_delta_min'],max=init_params['sig_delta_max'],vary=True)
    nparams['sig_b'].set(expr='sig_delta*sig')
    
    nparams['vel'].min = init_params['vel_min']
    nparams['vel'].max = init_params['vel_max']
    nparams.add('vel_delta',value=init_params['vel_delta'],min=init_params['vel_delta_min'],max=init_params['vel_delta_max'],vary=True)
    nparams['vel_b'].set(expr='vel+vel_delta')
    
    y_mask, x_mask = np.where(ncomps_flag == 1)
    
    global kloop
        
    def kloop(args):
        
        results_data, lam_r, nparams, k, l = args

        print("Progress {:2.1%}".format(float(k)/float(l)), end="\r")
        
        result = gmodel.fit(results_data, nparams, x=lam_r)
        
        return (result)
        
    with mp.Pool(n_proc) as pool:
        results_k = pool.starmap(kloop, zip((results[y_mask[k],x_mask[k]].data, lam_r, nparams, k, len(y_mask)) for k in np.arange(len(y_mask))))        

    for k in np.arange(len(y_mask)):
        if (results_k[k].bic < results[y_mask[k],x_mask[k]].bic):
            results[y_mask[k],x_mask[k]] = results_k[k]
            bic_flag[y_mask[k],x_mask[k]] = 2
        
    result_list = np.ravel(results)
        
    return result_list, ncomps_flag, bic_flag