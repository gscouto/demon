import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import initial_parameters
import fit_functions
import output
import refit
import more_comps
from demon_config import *

from astropy.io import fits
from astropy.table import Table, vstack,hstack
import numpy as np
import matplotlib.pyplot as plt
import tables
from lmfit import Model, fit_report
from lmfit.model import save_modelresult
import time
from astropy.constants import c
import multiprocessing as mp
import csv
from datetime import datetime
import os 
import sys
import warnings
import gc
import psutil

def forloop(args):
    
    xf, yf,params= args
    
    if ((xf/10.).is_integer()) & ((yf/10.).is_integer()) & (xf == yf):
        print('x = '+str(xf), '| y = '+str(yf))
    
    result = gmodel.fit(f_res[:,yf,xf], params, x=lam_r)

    return (result)

print('---------------------------------------')
print('')
print('Datacube EMissiON-line fitter (DEMON)')
print('')
print('--------------------------------------')

warnings.filterwarnings("ignore")

c = c.value/1000.

ilines = lines_to_fit

for g in ids:
    
    print('')
    print('Preparing to fit lines for galaxy '+g)
    print('')
        
    lines_to_fit = ilines

    if SL_flag == 'yes':
        cube = fits.open(cube_path+g+'_starlighted.fits')
    if SL_flag == 'no':
        cube = fits.open(cube_path+g+'.fits')

    crval3 = cube[1].header['CRVAL3']*1e10
    cdelt3 = cube[1].header['CDELT3']*cube[1].header['PC3_3']*1e10
    naxis1 = cube[1].header['NAXIS1']
    naxis2 = cube[1].header['NAXIS2']
    naxis3 = cube[1].header['NAXIS3']

    lam = (np.arange(naxis3)*cdelt3)+crval3

    lam_i_SN =  np.abs(lam - lami_SN).argmin()
    lam_f_SN =  np.abs(lam - lamf_SN).argmin()

    lam_i_cont1 =  np.abs(lam - lami_cont1).argmin()
    lam_f_cont1 =  np.abs(lam - lamf_cont1).argmin()

    lam_i_cont2 =  np.abs(lam - lami_cont2).argmin()
    lam_f_cont2 =  np.abs(lam - lamf_cont2).argmin()

    if SL_flag == 'yes':
        SN_spec = cube[1].data[lam_i_SN:lam_f_SN,:,:]-cube[5].data[lam_i_SN:lam_f_SN,:,:]
        cont1_spec = cube[1].data[lam_i_cont1:lam_f_cont1,:,:]-cube[5].data[lam_i_cont1:lam_f_cont1,:,:]
        cont2_spec = cube[1].data[lam_i_cont2:lam_f_cont2,:,:]-cube[5].data[lam_i_cont2:lam_f_cont2,:,:]
    if SL_flag == 'no':
        SN_spec = cube[1].data[lam_i_SN:lam_f_SN,:,:]
        cont1_spec = cube[1].data[lam_i_cont1:lam_f_cont1,:,:]
        cont2_spec = cube[1].data[lam_i_cont2:lam_f_cont2,:,:]        

    SN = (np.nanmean(SN_spec,axis=0) - (np.nanmean([np.nanmean(cont1_spec,axis=0),np.nanmean(cont2_spec,axis=0)],axis=0)))

    if lines_to_fit == 'all':
        for lines_to_fit in ['ha_n2','hb','o3','s2','o1']:
        
            run_time = datetime.now().strftime('%Y-%m-%d_%H.%M.%S')

            results_dir=results_path+g+'/'+lines_to_fit+'_'+run_time
            
            os.makedirs(results_dir, exist_ok=True)
        
            if lines_to_fit == 'ha_n2':
                print('Fitting the following emission lines: N2 and Ha')
                print('')
                gmodel = Model(fit_functions.three_gaussians)
                params,f_res,lam_r,SN = initial_parameters.ha_n2(gmodel,lam,cube,SN,c,results_dir,aut_ini)

            elif lines_to_fit == 'hb':
                print('')
                print('Fitting the following emission lines: Hb')
                print('')
                gmodel = Model(fit_functions.one_gaussian)
                params,f_res,lam_r,SN = initial_parameters.hb(gmodel,lam,cube,SN,c,results_dir,aut_ini)

            elif lines_to_fit == 'o3':
                print('')
                print('Fitting the following emission lines: O3')
                print('')
                gmodel = Model(fit_functions.two_gaussians)
                params,f_res,lam_r,SN = initial_parameters.o3(gmodel,lam,cube,SN,c,results_dir,aut_ini)
                
            elif lines_to_fit == 's2':
                print('')
                print('Fitting the following emission lines: S2')
                print('')
                gmodel = Model(fit_functions.two_gaussians)
                params,f_res,lam_r,SN = initial_parameters.s2(gmodel,lam,cube,SN,c,results_dir,aut_ini)
                
            elif lines_to_fit == 'o1':
                print('')
                print('Fitting the following emission lines: O1')
                print('')
                gmodel = Model(fit_functions.two_gaussians)
                params,f_res,lam_r,SN = initial_parameters.o1(gmodel,lam,cube,SN,c,results_dir,aut_ini)
                
            with open(results_dir+'/ini_params.txt','w') as params_file:
                for name, par in params.items():
                    params_file.write(str(par)+'\n')
        
            start = time.time()

            with mp.Pool(n_proc) as pool:
                result_list = pool.starmap(forloop, zip((xf, yf, params) for yf in np.arange(naxis2) for xf in np.arange(naxis1)))

            if refit_flag == 'yes':

                print('')
                print('Redoing bad fits...')            
                if lines_to_fit == 'ha_n2':
                    result_list = refit.three_gaussians(naxis1,naxis2,result_list,results_dir,lam_r,params,gmodel)
                    
                elif lines_to_fit == 'hb':
                    result_list = refit.one_gaussian(naxis1,naxis2,result_list,results_dir,lam_r,params,gmodel)
                    
                elif (lines_to_fit == 'o3') | (lines_to_fit == 'o1') | (lines_to_fit == 's2'):
                    result_list = refit.two_gaussians(naxis1,naxis2,result_list,results_dir,lam_r,params,gmodel)

            print('')
            print('Finished fitting the datacube. Wrapping up...')
            print('')

            if lines_to_fit == 'ha_n2':
                output.three_gaussians(naxis1,naxis2,result_list,results_dir,lam_r)

            elif lines_to_fit == 'hb':
                output.one_gaussian(naxis1,naxis2,result_list,results_dir,lam_r)
                
            elif (lines_to_fit == 'o3') | (lines_to_fit == 's2') | (lines_to_fit == 'o1'):
                output.two_gaussians(naxis1,naxis2,result_list,results_dir,lam_r)
                
            fits.writeto(results_dir+'/SN.fits',SN,overwrite=True)
            
            end = time.time()

            runtime = end-start
            hrs = runtime/60./60.
            mins = (hrs-int(hrs))*60.
            secs = round((mins-int(mins))*60.)
            
            print('')
            print('This run took '+str(int(hrs))+'h'+str(int(mins))+'m'+str(secs)+'s.')
            print('')
            
            del f_res
            del result_list
            
            gc.collect()

    else:
        
        if lines_to_fit == 'all_cons':
            for lines_to_fit in np.array(['hb_ha_n2_cons','o3_cons','s2_cons','o1_cons']):
            
                run_time = datetime.now().strftime('%Y-%m-%d_%H.%M.%S')

                if morec_flag == 'yes':
                    results_dir=results_path+g+'/'+lines_to_fit+'_2g'+'_'+run_time                
                else:    
                    results_dir=results_path+g+'/'+lines_to_fit+'_'+run_time
                
                os.makedirs(results_dir, exist_ok=True)
            
                if lines_to_fit == 'hb_ha_n2_cons':
                    print('Fitting the following emission lines: Hb + N2 and Ha')
                    print('')
                    gmodel = Model(fit_functions.hb_ha_n2_gaussians_cons)
                    params,f_res,lam_r,SN = initial_parameters.hb_ha_n2_cons(gmodel,lam,cube,SN,c,results_dir,aut_ini)
                    
                elif lines_to_fit == 'o3_cons':
                    print('')
                    print('Fitting the following emission lines: O3')
                    print('')
                    gmodel = Model(fit_functions.two_gaussians_cons_o3)
                    params,f_res,lam_r,SN = initial_parameters.o3_cons(gmodel,lam,cube,SN,c,results_dir,aut_ini)
                    
                elif lines_to_fit == 'o1_cons':
                    print('')
                    print('Fitting the following emission lines: O1')
                    print('')
                    gmodel = Model(fit_functions.two_gaussians_cons_o1)
                    params,f_res,lam_r,SN = initial_parameters.o1_cons(gmodel,lam,cube,SN,c,results_dir,aut_ini)
                    
                elif lines_to_fit == 's2_cons':
                    print('')
                    print('Fitting the following emission lines: S2')
                    print('')
                    gmodel = Model(fit_functions.two_gaussians_cons_s2)
                    params,f_res,lam_r,SN = initial_parameters.s2_cons(gmodel,lam,cube,SN,c,results_dir,aut_ini)

                params_file = open(results_dir+'/ini_params.txt','w')
                for name, par in params.items():
                    params_file.write(str(par)+'\n')
                params_file.close()
            
                start = time.time()

                with mp.Pool(n_proc) as pool:
                    result_list = pool.starmap(forloop, zip((xf, yf, params) for yf in np.arange(naxis2) for xf in np.arange(naxis1)))
                
                if refit_flag == 'yes':
                
                    print('')
                    print('Redoing bad fits...')
                    if lines_to_fit == 'hb_ha_n2_cons':
                        result_list = refit.hb_ha_n2_gaussians_cons(naxis1,naxis2,result_list,results_dir,lam_r,params,gmodel)
                        
                    elif (lines_to_fit == 'o3_cons') | (lines_to_fit == 'o1_cons'):
                        result_list = refit.two_gaussians_cons(naxis1,naxis2,result_list,results_dir,lam_r,params,gmodel)

                    elif lines_to_fit == 's2_cons':
                        result_list = refit.two_gaussians_cons_s2(naxis1,naxis2,result_list,results_dir,lam_r,params,gmodel)
                    
                if morec_flag == 'yes':
                    print('')
                    print('Redoing fits with more than one component...')

                    if lines_to_fit == 'o3_cons':
                        result_list, ncomps_flag, bic_flag = more_comps.two_gaussians_cons_o3(naxis1,naxis2,result_list,results_dir,lam_r,params)

                    elif lines_to_fit == 'o1_cons':
                        result_list, ncomps_flag, bic_flag = more_comps.two_gaussians_cons_o1(naxis1,naxis2,result_list,results_dir,lam_r,params)
                    
                    elif lines_to_fit == 's2_cons':
                        result_list, ncomps_flag, bic_flag = more_comps.two_gaussians_cons_s2(naxis1,naxis2,result_list,results_dir,lam_r,params)
                    
                    elif lines_to_fit == 'ha_n2_cons':
                        result_list, ncomps_flag, bic_flag = more_comps.three_gaussians_cons(naxis1,naxis2,result_list,results_dir,lam_r,params)
                        
                    elif lines_to_fit == 'hb_ha_n2_cons':
                        result_list, ncomps_flag, bic_flag = more_comps.hb_ha_n2_gaussians_cons(naxis1,naxis2,result_list,results_dir,lam_r,params)
                
                    print('')
                    print('Finished fitting the datacube. Wrapping up...')
                    print('')
                
                    if lines_to_fit == 'o3_cons':
                        output.two_gaussians_2g_cons_o3(naxis1,naxis2,result_list,results_dir,lam_r,ncomps_flag,bic_flag,cube[0].header)

                    elif lines_to_fit == 'o1_cons':
                        output.two_gaussians_2g_cons_o1(naxis1,naxis2,result_list,results_dir,lam_r,ncomps_flag,bic_flag,cube[0].header)

                    elif lines_to_fit == 's2_cons':
                        output.two_gaussians_2g_cons_s2(naxis1,naxis2,result_list,results_dir,lam_r,ncomps_flag,bic_flag,cube[0].header)
                
                    elif lines_to_fit == 'ha_n2_cons':
                        output.three_gaussians_2g_cons(naxis1,naxis2,result_list,results_dir,lam_r,ncomps_flag,bic_flag,cube[0].header)
                        
                    elif lines_to_fit == 'hb_ha_n2_cons':
                        output.hb_ha_n2_gaussians_2g_cons(naxis1,naxis2,result_list,results_dir,lam_r,ncomps_flag,bic_flag,cube[0].header)
                
                else:

                    print('')
                    print('Finished fitting the datacube. Wrapping up...')
                    print('')

                    if lines_to_fit == 'hb_ha_n2_cons':
                        output.hb_ha_n2_gaussians_cons(naxis1,naxis2,result_list,results_dir,lam_r,cube[0].header)
                        
                    elif (lines_to_fit == 'o3_cons') | (lines_to_fit == 'o1_cons'):
                        output.two_gaussians_cons(naxis1,naxis2,result_list,results_dir,lam_r,cube[0].header)

                    elif lines_to_fit == 's2_cons':
                        output.two_gaussians_cons_s2(naxis1,naxis2,result_list,results_dir,lam_r,cube[0].header)

                fits.writeto(results_dir+'/SN.fits',SN,overwrite=True)
                
                end = time.time()

                runtime = end-start
                hrs = runtime/60./60.
                mins = (hrs-int(hrs))*60.
                secs = round((mins-int(mins))*60.)
                
                print('')
                print('This run took '+str(int(hrs))+'h'+str(int(mins))+'m'+str(secs)+'s.')
                print('')
                    
                del f_res
                del result_list
                
                gc.collect()
              
        else:

            run_time = datetime.now().strftime('%Y-%m-%d_%H.%M.%S')

            if morec_flag == 'yes':
                results_dir=results_path+g+'/'+lines_to_fit+'_2g'+'_'+run_time                
            else:    
                results_dir=results_path+g+'/'+lines_to_fit+'_'+run_time
            
            os.makedirs(results_dir, exist_ok=True)

            if lines_to_fit == 'ha_n2':
                print('Fitting the following emission lines: N2 and Ha')
                print('')
                gmodel = Model(fit_functions.three_gaussians)
                params,f_res,lam_r,SN = initial_parameters.ha_n2(gmodel,lam,cube,SN,c,results_dir,aut_ini)
                
            elif lines_to_fit == 'ha_n2_cons':
                print('Fitting the following emission lines: N2 and Ha')
                print('')
                gmodel = Model(fit_functions.three_gaussians_cons)
                params,f_res,lam_r,SN = initial_parameters.ha_n2_cons(gmodel,lam,cube,SN,c,results_dir,aut_ini)
                
            elif lines_to_fit == 'ha_n2_2g_cons':
                print('Fitting the following emission lines: N2 and Ha (2 gaussians each)')
                print('')
                gmodel = Model(fit_functions.three_gaussians_2g_cons)
                params,f_res,lam_r,SN = initial_parameters.ha_n2_2g_cons(gmodel,lam,cube,SN,c,results_dir,aut_ini)
                
            elif lines_to_fit == 'hb_ha_n2_cons':
                print('Fitting the following emission lines: Hb + N2 and Ha')
                print('')
                gmodel = Model(fit_functions.hb_ha_n2_gaussians_cons)
                params,f_res,lam_r,SN = initial_parameters.hb_ha_n2_cons(gmodel,lam,cube,SN,c,results_dir,aut_ini)

            elif lines_to_fit == 'hb':
                print('')
                print('Fitting the following emission lines: Hb')
                print('')
                gmodel = Model(fit_functions.one_gaussian)
                params,f_res,lam_r,SN = initial_parameters.hb(gmodel,lam,cube,SN,c,results_dir,aut_ini)

            elif lines_to_fit == 'o3':
                print('')
                print('Fitting the following emission lines: O3')
                print('')
                gmodel = Model(fit_functions.two_gaussians)
                params,f_res,lam_r,SN = initial_parameters.o3(gmodel,lam,cube,SN,c,results_dir,aut_ini)
                
            elif lines_to_fit == 'o3_cons':
                print('')
                print('Fitting the following emission lines: O3')
                print('')
                gmodel = Model(fit_functions.two_gaussians_cons_o3)
                params,f_res,lam_r,SN = initial_parameters.o3_cons(gmodel,lam,cube,SN,c,results_dir,aut_ini)
                
            elif lines_to_fit == 's2':
                print('')
                print('Fitting the following emission lines: S2')
                print('')
                gmodel = Model(fit_functions.two_gaussians)
                params,f_res,lam_r,SN = initial_parameters.s2(gmodel,lam,cube,SN,c,results_dir,aut_ini)
                
            elif lines_to_fit == 's2_cons':
                print('')
                print('Fitting the following emission lines: S2')
                print('')
                gmodel = Model(fit_functions.two_gaussians_cons_s2)
                params,f_res,lam_r,SN = initial_parameters.s2_cons(gmodel,lam,cube,SN,c,results_dir,aut_ini)
                
            elif lines_to_fit == 'o1':
                print('')
                print('Fitting the following emission lines: O1')
                print('')
                gmodel = Model(fit_functions.two_gaussians_o1)
                params,f_res,lam_r,SN = initial_parameters.o1(gmodel,lam,cube,SN,c,results_dir,aut_ini)
                
            elif lines_to_fit == 'o1_cons':
                print('')
                print('Fitting the following emission lines: O1')
                print('')
                gmodel = Model(fit_functions.two_gaussians_cons_o1)
                params,f_res,lam_r,SN = initial_parameters.o1_cons(gmodel,lam,cube,SN,c,results_dir,aut_ini)

            params_file = open(results_dir+'/ini_params.txt','w')
            for name, par in params.items():
                params_file.write(str(par)+'\n')
            params_file.close()

            start = time.time()         
                         
            with mp.Pool(n_proc) as pool:
                result_list = pool.starmap(forloop, zip((xf, yf, params) for yf in np.arange(naxis2) for xf in np.arange(naxis1)))

            if refit_flag == 'yes':

                print('')
                print('Redoing bad fits...')
                
                if lines_to_fit == 'ha_n2_cons':
                    result_list = refit.three_gaussians_cons(naxis1,naxis2,result_list,results_dir,lam_r,params,gmodel)
                    
                elif lines_to_fit == 'ha_n2':
                    result_list = refit.three_gaussians(naxis1,naxis2,result_list,results_dir,lam_r,params,gmodel)
                    
                elif lines_to_fit == 'ha_n2_2g_cons':
                    result_list = refit.three_gaussians_2g_cons(naxis1,naxis2,result_list,results_dir,lam_r,params,gmodel)
                    
                elif lines_to_fit == 'hb_ha_n2_cons':
                    result_list = refit.hb_ha_n2_gaussians_cons(naxis1,naxis2,result_list,results_dir,lam_r,params,gmodel)
                    
                elif lines_to_fit == 'hb':
                    result_list = refit.one_gaussian(naxis1,naxis2,result_list,results_dir,lam_r,params,gmodel)
                    
                elif (lines_to_fit == 'o3') | (lines_to_fit == 'o1') | (lines_to_fit == 's2'):
                    result_list = refit.two_gaussians(naxis1,naxis2,result_list,results_dir,lam_r,params,gmodel)
                    
                elif (lines_to_fit == 'o3_cons') | (lines_to_fit == 'o1_cons'):
                    result_list = refit.two_gaussians_cons(naxis1,naxis2,result_list,results_dir,lam_r,params,gmodel)

                elif lines_to_fit == 's2_cons':
                    result_list = refit.two_gaussians_cons_s2(naxis1,naxis2,result_list,results_dir,lam_r,params,gmodel)
                
            if morec_flag == 'yes':
                print('')
                print('Redoing fits with more than one component...')

                if lines_to_fit == 'o3_cons':
                    result_list, ncomps_flag, bic_flag = more_comps.two_gaussians_cons_o3(naxis1,naxis2,result_list,results_dir,lam_r,params)

                elif lines_to_fit == 'o1_cons':
                    result_list, ncomps_flag, bic_flag = more_comps.two_gaussians_cons_o1(naxis1,naxis2,result_list,results_dir,lam_r,params)
                
                elif lines_to_fit == 's2_cons':
                    result_list, ncomps_flag, bic_flag = more_comps.two_gaussians_cons_s2(naxis1,naxis2,result_list,results_dir,lam_r,params)
                
                elif lines_to_fit == 'ha_n2_cons':
                    result_list, ncomps_flag, bic_flag = more_comps.three_gaussians_cons(naxis1,naxis2,result_list,results_dir,lam_r,params)
                    
                elif lines_to_fit == 'hb_ha_n2_cons':
                    result_list, ncomps_flag, bic_flag = more_comps.hb_ha_n2_gaussians_cons(naxis1,naxis2,result_list,results_dir,lam_r,params)
            
                print('')
                print('Finished fitting the datacube. Wrapping up...')
                print('')
            
                if lines_to_fit == 'o3_cons':
                    output.two_gaussians_2g_cons_o3(naxis1,naxis2,result_list,results_dir,lam_r,ncomps_flag,bic_flag,cube[0].header)

                elif lines_to_fit == 'o1_cons':
                    output.two_gaussians_2g_cons_o1(naxis1,naxis2,result_list,results_dir,lam_r,ncomps_flag,bic_flag,cube[0].header)

                elif lines_to_fit == 's2_cons':
                    output.two_gaussians_2g_cons_s2(naxis1,naxis2,result_list,results_dir,lam_r,ncomps_flag,bic_flag,cube[0].header)
            
                elif lines_to_fit == 'ha_n2_cons':
                    output.three_gaussians_2g_cons(naxis1,naxis2,result_list,results_dir,lam_r,ncomps_flag,bic_flag,cube[0].header)
                    
                elif lines_to_fit == 'hb_ha_n2_cons':
                    output.hb_ha_n2_gaussians_2g_cons(naxis1,naxis2,result_list,results_dir,lam_r,ncomps_flag,bic_flag,cube[0].header)
            
            else:
                
                print('')
                print('Finished fitting the datacube. Wrapping up...')
                print('')
            
                if lines_to_fit == 'ha_n2':
                    output.three_gaussians(naxis1,naxis2,result_list,results_dir,lam_r,cube[0].header)
                    
                elif lines_to_fit == 'ha_n2_cons':
                    output.three_gaussians_cons(naxis1,naxis2,result_list,results_dir,lam_r,cube[0].header)
                    
                elif lines_to_fit == 'ha_n2_2g_cons':
                    output.three_gaussians_2g_cons(naxis1,naxis2,result_list,results_dir,lam_r,cube[0].header)
                    
                elif lines_to_fit == 'hb_ha_n2_cons':
                    output.hb_ha_n2_gaussians_cons(naxis1,naxis2,result_list,results_dir,lam_r,cube[0].header)

                elif lines_to_fit == 'hb':
                    output.one_gaussian(naxis1,naxis2,result_list,results_dir,lam_r,cube[0].header)
                    
                elif (lines_to_fit == 'o3') | (lines_to_fit == 's2') | (lines_to_fit == 'o1'):
                    output.two_gaussians(naxis1,naxis2,result_list,results_dir,lam_r,cube[0].header)
                    
                elif (lines_to_fit == 'o3_cons') | (lines_to_fit == 'o1_cons'):
                    output.two_gaussians_cons(naxis1,naxis2,result_list,results_dir,lam_r,cube[0].header)

                elif lines_to_fit == 's2_cons':
                    output.two_gaussians_cons_s2(naxis1,naxis2,result_list,results_dir,lam_r,cube[0].header)

            fits.writeto(results_dir+'/SN.fits',SN,overwrite=True)
            
            end = time.time()

            runtime = end-start
            hrs = runtime/60./60.
            mins = (hrs-int(hrs))*60.
            secs = round((mins-int(mins))*60.)
            
            print('')
            print('This run took '+str(int(hrs))+'h'+str(int(mins))+'m'+str(secs)+'s.')
            print('')
            
            del f_res
            del result_list        

            gc.collect()
            
    del cube

print('All done! So long, and thanks for all the fish! =)')
