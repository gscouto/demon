import numpy as np
from astropy.io import fits
import tables
import h5py
from astropy.constants import c
from demon_config_alma import *

c = c.value/1000.

def one_gaussian(naxis1,naxis2,result_list,results_dir,freq_r,hdr):

    results = np.reshape(result_list, (naxis2,naxis1))

    am = np.zeros([naxis2,naxis1])
    bm = np.zeros([naxis2,naxis1])
    fluxm = np.zeros([naxis2,naxis1])
    velm = np.zeros([naxis2,naxis1])
    sigm = np.zeros([naxis2,naxis1])

    am_err = np.zeros([naxis2,naxis1])
    bm_err = np.zeros([naxis2,naxis1])
    fluxm_err = np.zeros([naxis2,naxis1])
    velm_err = np.zeros([naxis2,naxis1])
    sigm_err = np.zeros([naxis2,naxis1])
    
    residualsm = np.zeros([naxis2,naxis1])
    redchi = np.zeros([naxis2,naxis1])
    redchi_w = np.zeros([naxis2,naxis1])
    bic = np.zeros([naxis2,naxis1])
    
    residual = np.zeros([len(results[0,0].residual),naxis2,naxis1])
    model = np.zeros([len(results[0,0].residual),naxis2,naxis1])
    data = np.zeros([len(results[0,0].residual),naxis2,naxis1])
    
    xm = np.zeros([naxis2,naxis1])
    ym = np.zeros([naxis2,naxis1])

    for i in np.arange(naxis1):
        for j in np.arange(naxis2):
            am[j,i] = results[j,i].params['a'].value
            bm[j,i] = results[j,i].params['b'].value
            fluxm[j,i] = results[j,i].params['flux'].value
            velm[j,i] = results[j,i].params['vel'].value
            sigm[j,i] = results[j,i].params['sig'].value

            am_err[j,i] = results[j,i].params['a'].stderr
            bm_err[j,i] = results[j,i].params['b'].stderr
            fluxm_err[j,i] = results[j,i].params['flux'].stderr
            velm_err[j,i] = results[j,i].params['vel'].stderr
            sigm_err[j,i] = results[j,i].params['sig'].stderr
            
            residualsm[j,i] = np.abs(np.sum(results[j,i].residual))
            redchi[j,i] = results[j,i].redchi
            redchi_w[j,i] = results[j,i].redchi*np.abs(residualsm[j,i])
            bic[j,i] = results[j,i].bic
            
            residual[:,j,i] = results[j,i].residual
            data[:,j,i] = results[j,i].data
            model[:,j,i] = results[j,i].best_fit
            
            xm[j,i] = i
            ym[j,i] = j

    fits.writeto(results_dir+'/fluxm.fits',fluxm,header=hdr,overwrite=True)
    fits.writeto(results_dir+'/velm.fits',velm,header=hdr,overwrite=True)
    fits.writeto(results_dir+'/sigm.fits',sigm,header=hdr,overwrite=True)

    fits.writeto(results_dir+'/flux_err.fits',fluxm_err,header=hdr,overwrite=True)
    fits.writeto(results_dir+'/vel_err.fits',velm_err,header=hdr,overwrite=True)
    fits.writeto(results_dir+'/sig_err.fits',sigm_err,header=hdr,overwrite=True)
    
    fits.writeto(results_dir+'/residuals.fits',residualsm,header=hdr,overwrite=True)
    fits.writeto(results_dir+'/redchi.fits',redchi,header=hdr,overwrite=True)
    fits.writeto(results_dir+'/redchi_w.fits',redchi_w,header=hdr,overwrite=True)
    fits.writeto(results_dir+'/bic.fits',bic,header=hdr,overwrite=True)

    xm = np.ravel(xm)
    ym = np.ravel(ym)

    am = np.ravel(am)
    bm = np.ravel(bm)
    fluxm = np.ravel(fluxm)
    velm = np.ravel(velm)
    sigm = np.ravel(sigm)

    am_err = np.ravel(am_err)
    bm_err = np.ravel(bm_err)
    fluxm_err = np.ravel(fluxm_err)
    velm_err = np.ravel(velm_err)
    sigm_err = np.ravel(sigm_err)
    
    residualsm = np.ravel(residualsm)

    class t_res(tables.IsDescription):
        
        x = tables.IntCol(shape=(len(result_list)))
        y = tables.IntCol(shape=(len(result_list)))
        
        a = tables.Float64Col(shape=(len(result_list)))
        b = tables.Float64Col(shape=(len(result_list)))
        flux = tables.Float64Col(shape=(len(result_list)))
        vel = tables.Float64Col(shape=(len(result_list)))
        sig = tables.Float64Col(shape=(len(result_list)))
        
        a_err = tables.Float64Col(shape=(len(result_list)))
        b_err = tables.Float64Col(shape=(len(result_list)))
        flux_err = tables.Float64Col(shape=(len(result_list)))
        vel_err = tables.Float64Col(shape=(len(result_list)))
        sig_err = tables.Float64Col(shape=(len(result_list)))
        
        residuals = tables.Float64Col(shape=(len(result_list)))
        
    hf = tables.open_file(results_dir+'/results.h5', 'w')

    th = hf.create_table('/','results', t_res)

    hr = th.row

    hr['x'] = xm
    hr['y'] = ym

    hr['a'] = am
    hr['b'] = bm
    hr['flux'] = fluxm
    hr['vel'] = velm
    hr['sig'] = sigm

    hr['a_err'] = am_err
    hr['b_err'] = bm_err
    hr['flux_err'] = fluxm_err
    hr['vel_err'] = velm_err
    hr['sig_err'] = sigm_err
    
    hr['residuals'] = residualsm

    hr.append()

    hf.flush()
    hf.close()

    ft = h5py.File(results_dir+'/fit.hdf5', 'w')
    ft.create_dataset('residual', data=residual)
    ft.create_dataset('model', data=model)
    ft.create_dataset('data', data=data)
    ft.create_dataset('freq', data=freq_r)
    ft.close()

    return
