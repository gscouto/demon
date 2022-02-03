import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import initial_parameters_alma
import fit_functions_alma
import output_alma
import refit
import more_comps
from demon_config_alma import *

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
    
    result = gmodel.fit(f_res[:,yf,xf], params, x=freq_r)

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

    cube = fits.open(cube_path+g)

    crval3 = cube[0].header['CRVAL3']
    cdelt3 = cube[0].header['CDELT3']
    naxis1 = cube[0].header['NAXIS1']
    naxis2 = cube[0].header['NAXIS2']
    naxis3 = cube[0].header['NAXIS3']
        
    # CONTINUAR AQUI

    freq = ((np.arange(naxis3)*cdelt3)+crval3)*1e-9

    freq_i_SN =  np.abs(freq - freqi_SN).argmin()
    freq_f_SN =  np.abs(freq - freqf_SN).argmin()

    freq_i_cont1 =  np.abs(freq - freqi_cont1).argmin()
    freq_f_cont1 =  np.abs(freq - freqf_cont1).argmin()

    freq_i_cont2 =  np.abs(freq - freqi_cont2).argmin()
    freq_f_cont2 =  np.abs(freq - freqf_cont2).argmin()

    SN_spec = cube[0].data[0,freq_f_SN:freq_i_SN,:,:]
    cont1_spec = cube[0].data[0,freq_f_cont1:freq_i_cont1,:,:]
    cont2_spec = cube[0].data[0,freq_f_cont2:freq_i_cont2,:,:]        

    SN = (np.nanmean(SN_spec,axis=0) - (np.nanmean([np.nanmean(cont1_spec,axis=0),np.nanmean(cont2_spec,axis=0)],axis=0)))

    run_time = datetime.now().strftime('%Y-%m-%d_%H.%M.%S')

    results_dir=results_path+g+'/'+lines_to_fit+'_'+run_time
    
    os.makedirs(results_dir, exist_ok=True)

    if lines_to_fit == 'co3-2':
        print('')
        print('Fitting the following emission lines: CO(3-2)')
        print('')
        gmodel = Model(fit_functions_alma.one_gaussian_co32)
        params,f_res,freq_r,SN = initial_parameters_alma.co32(gmodel,freq,cube,SN,c,results_dir,aut_ini)

    params_file = open(results_dir+'/ini_params.txt','w')
    for name, par in params.items():
        params_file.write(str(par)+'\n')
    params_file.close()

    start = time.time()         
                    
    with mp.Pool(n_proc) as pool:
        result_list = pool.starmap(forloop, zip((xf, yf, params) for yf in np.arange(naxis2) for xf in np.arange(naxis1)))
        
    print('')
    print('Finished fitting the datacube. Wrapping up...')
    print('')

    if lines_to_fit == 'co3-2':
        output_alma.one_gaussian(naxis1,naxis2,result_list,results_dir,freq_r,cube[0].header)

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
