import numpy as np
from astropy.io import fits
import tables
import h5py
from astropy.constants import c
from line_fit_config import *

c = c.value/1000.

def one_gaussian(naxis1,naxis2,result_list,results_dir,lam_r):

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
            
            residual[:,j,i] = results[j,i].residual
            data[:,j,i] = results[j,i].data
            model[:,j,i] = results[j,i].best_fit
            
            xm[j,i] = i
            ym[j,i] = j

    fits.writeto(results_dir+'/fluxm.fits',fluxm,overwrite=True)
    fits.writeto(results_dir+'/velm.fits',velm,overwrite=True)
    fits.writeto(results_dir+'/sigm.fits',sigm,overwrite=True)

    fits.writeto(results_dir+'/flux_err.fits',fluxm_err,overwrite=True)
    fits.writeto(results_dir+'/vel_err.fits',velm_err,overwrite=True)
    fits.writeto(results_dir+'/sig_err.fits',sigm_err,overwrite=True)
    
    fits.writeto(results_dir+'/residuals.fits',residualsm,overwrite=True)
    fits.writeto(results_dir+'/redchi.fits',redchi,overwrite=True)
    fits.writeto(results_dir+'/redchi_w.fits',redchi_w,overwrite=True)

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
    ft.create_dataset('lam', data=lam_r)
    ft.close()

    return

def two_gaussians(naxis1,naxis2,result_list,results_dir,lam_r):

    results = np.reshape(result_list, (naxis2,naxis1))

    am = np.zeros([naxis2,naxis1])
    bm = np.zeros([naxis2,naxis1])
    fluxm1 = np.zeros([naxis2,naxis1])
    fluxm2 = np.zeros([naxis2,naxis1])
    velm1 = np.zeros([naxis2,naxis1])
    velm2 = np.zeros([naxis2,naxis1])
    sigm1 = np.zeros([naxis2,naxis1])
    sigm2 = np.zeros([naxis2,naxis1])

    am_err = np.zeros([naxis2,naxis1])
    bm_err = np.zeros([naxis2,naxis1])
    fluxm1_err = np.zeros([naxis2,naxis1])
    fluxm2_err = np.zeros([naxis2,naxis1])
    velm1_err = np.zeros([naxis2,naxis1])
    velm2_err = np.zeros([naxis2,naxis1])
    sigm1_err = np.zeros([naxis2,naxis1])
    sigm2_err = np.zeros([naxis2,naxis1])
    
    residualsm = np.zeros([naxis2,naxis1])
    redchi = np.zeros([naxis2,naxis1])
    redchi_w = np.zeros([naxis2,naxis1])
    
    residual = np.zeros([len(results[0,0].residual),naxis2,naxis1])
    model = np.zeros([len(results[0,0].residual),naxis2,naxis1])
    data = np.zeros([len(results[0,0].residual),naxis2,naxis1])

    xm = np.zeros([naxis2,naxis1])
    ym = np.zeros([naxis2,naxis1])

    for i in np.arange(naxis1):
        for j in np.arange(naxis2):
            am[j,i] = results[j,i].params['a'].value
            bm[j,i] = results[j,i].params['b'].value
            fluxm1[j,i] = results[j,i].params['flux1'].value
            fluxm2[j,i] = results[j,i].params['flux2'].value
            velm1[j,i] = results[j,i].params['vel1'].value
            velm2[j,i] = results[j,i].params['vel2'].value
            sigm1[j,i] = results[j,i].params['sig1'].value
            sigm2[j,i] = results[j,i].params['sig2'].value

            am_err[j,i] = results[j,i].params['a'].stderr
            bm_err[j,i] = results[j,i].params['b'].stderr
            fluxm1_err[j,i] = results[j,i].params['flux1'].stderr
            fluxm2_err[j,i] = results[j,i].params['flux2'].stderr
            velm1_err[j,i] = results[j,i].params['vel1'].stderr
            velm2_err[j,i] = results[j,i].params['vel2'].stderr
            sigm1_err[j,i] = results[j,i].params['sig1'].stderr
            sigm2_err[j,i] = results[j,i].params['sig2'].stderr
            
            residualsm[j,i] = np.abs(np.sum(results[j,i].residual))
            redchi[j,i] = results[j,i].redchi
            redchi_w[j,i] = results[j,i].redchi*np.abs(residualsm[j,i])
            
            residual[:,j,i] = results[j,i].residual
            data[:,j,i] = results[j,i].data
            model[:,j,i] = results[j,i].best_fit
            
            xm[j,i] = i
            ym[j,i] = j

    fits.writeto(results_dir+'/fluxm1.fits',fluxm1,overwrite=True)
    fits.writeto(results_dir+'/fluxm2.fits',fluxm2,overwrite=True)
    fits.writeto(results_dir+'/velm1.fits',velm1,overwrite=True)
    fits.writeto(results_dir+'/velm2.fits',velm2,overwrite=True)
    fits.writeto(results_dir+'/sigm1.fits',sigm1,overwrite=True)
    fits.writeto(results_dir+'/sigm2.fits',sigm2,overwrite=True)

    fits.writeto(results_dir+'/flux1_err.fits',fluxm1_err,overwrite=True)
    fits.writeto(results_dir+'/flux2_err.fits',fluxm2_err,overwrite=True)
    fits.writeto(results_dir+'/vel1_err.fits',velm1_err,overwrite=True)
    fits.writeto(results_dir+'/vel2_err.fits',velm2_err,overwrite=True)
    fits.writeto(results_dir+'/sig1_err.fits',sigm1_err,overwrite=True)
    fits.writeto(results_dir+'/sig2_err.fits',sigm2_err,overwrite=True)
    
    fits.writeto(results_dir+'/residuals.fits',residualsm,overwrite=True)
    fits.writeto(results_dir+'/redchi.fits',redchi,overwrite=True)
    fits.writeto(results_dir+'/redchi_w.fits',redchi_w,overwrite=True)

    xm = np.ravel(xm)
    ym = np.ravel(ym)

    am = np.ravel(am)
    bm = np.ravel(bm)
    fluxm1 = np.ravel(fluxm1)
    fluxm2 = np.ravel(fluxm2)
    velm1 = np.ravel(velm1)
    velm2 = np.ravel(velm2)
    sigm1 = np.ravel(sigm1)
    sigm2 = np.ravel(sigm2)

    am_err = np.ravel(am_err)
    bm_err = np.ravel(bm_err)
    fluxm1_err = np.ravel(fluxm1_err)
    fluxm2_err = np.ravel(fluxm2_err)
    velm1_err = np.ravel(velm1_err)
    velm2_err = np.ravel(velm2_err)
    sigm1_err = np.ravel(sigm1_err)
    sigm2_err = np.ravel(sigm2_err)
    
    residualsm = np.ravel(residualsm)

    class t_res(tables.IsDescription):
        
        x = tables.IntCol(shape=(len(result_list)))
        y = tables.IntCol(shape=(len(result_list)))
        
        a = tables.Float64Col(shape=(len(result_list)))
        b = tables.Float64Col(shape=(len(result_list)))
        flux1 = tables.Float64Col(shape=(len(result_list)))
        flux2 = tables.Float64Col(shape=(len(result_list)))
        vel1 = tables.Float64Col(shape=(len(result_list)))
        vel2 = tables.Float64Col(shape=(len(result_list)))
        sig1 = tables.Float64Col(shape=(len(result_list)))
        sig2 = tables.Float64Col(shape=(len(result_list)))
        
        a_err = tables.Float64Col(shape=(len(result_list)))
        b_err = tables.Float64Col(shape=(len(result_list)))
        flux1_err = tables.Float64Col(shape=(len(result_list)))
        flux2_err = tables.Float64Col(shape=(len(result_list)))
        vel1_err = tables.Float64Col(shape=(len(result_list)))
        vel2_err = tables.Float64Col(shape=(len(result_list)))
        sig1_err = tables.Float64Col(shape=(len(result_list)))
        sig2_err = tables.Float64Col(shape=(len(result_list)))
        
        residuals = tables.Float64Col(shape=(len(result_list)))
        
    hf = tables.open_file(results_dir+'/results.h5', 'w')

    th = hf.create_table('/','results', t_res)

    hr = th.row

    hr['x'] = xm
    hr['y'] = ym

    hr['a'] = am
    hr['b'] = bm
    hr['flux1'] = fluxm1
    hr['flux2'] = fluxm2
    hr['vel1'] = velm1
    hr['vel2'] = velm2
    hr['sig1'] = sigm1
    hr['sig2'] = sigm2

    hr['a_err'] = am_err
    hr['b_err'] = bm_err
    hr['flux1_err'] = fluxm1_err
    hr['flux2_err'] = fluxm2_err
    hr['vel1_err'] = velm1_err
    hr['vel2_err'] = velm2_err
    hr['sig1_err'] = sigm1_err
    hr['sig2_err'] = sigm2_err

    hr['residuals'] = residualsm

    hr.append()

    hf.flush()
    hf.close()

    ft = h5py.File(results_dir+'/fit.hdf5', 'w')
    ft.create_dataset('residual', data=residual)
    ft.create_dataset('model', data=model)
    ft.create_dataset('data', data=data)
    ft.create_dataset('lam', data=lam_r)
    ft.close()

    return

def two_gaussians_cons(naxis1,naxis2,result_list,results_dir,lam_r):

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
            
            residual[:,j,i] = results[j,i].residual
            data[:,j,i] = results[j,i].data
            model[:,j,i] = results[j,i].best_fit
            
            xm[j,i] = i
            ym[j,i] = j

    fits.writeto(results_dir+'/fluxm.fits',fluxm,overwrite=True)
    fits.writeto(results_dir+'/velm.fits',velm,overwrite=True)
    fits.writeto(results_dir+'/sigm.fits',sigm,overwrite=True)

    fits.writeto(results_dir+'/flux_err.fits',fluxm_err,overwrite=True)
    fits.writeto(results_dir+'/vel_err.fits',velm_err,overwrite=True)
    fits.writeto(results_dir+'/sig_err.fits',sigm_err,overwrite=True)
    
    fits.writeto(results_dir+'/residuals.fits',residualsm,overwrite=True)
    fits.writeto(results_dir+'/redchi.fits',redchi,overwrite=True)
    fits.writeto(results_dir+'/redchi_w.fits',redchi_w,overwrite=True)

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
    ft.create_dataset('lam', data=lam_r)
    ft.close()

    return

def two_gaussians_2g_cons_o1(naxis1,naxis2,result_list,results_dir,lam_r,ncomps_flag,bic_flag):

    results = np.reshape(result_list, (naxis2,naxis1))

    am = np.zeros([naxis2,naxis1])
    bm = np.zeros([naxis2,naxis1])
    fluxm = np.zeros([naxis2,naxis1])
    fluxm_b = np.zeros([naxis2,naxis1])
    velm = np.zeros([naxis2,naxis1])
    sigm = np.zeros([naxis2,naxis1])
    velm_b = np.zeros([naxis2,naxis1])
    sigm_b = np.zeros([naxis2,naxis1])

    am_err = np.zeros([naxis2,naxis1])
    bm_err = np.zeros([naxis2,naxis1])
    fluxm_err = np.zeros([naxis2,naxis1])
    fluxm_b_err = np.zeros([naxis2,naxis1])
    velm_err = np.zeros([naxis2,naxis1])
    sigm_err = np.zeros([naxis2,naxis1])
    velm_b_err = np.zeros([naxis2,naxis1])
    sigm_b_err = np.zeros([naxis2,naxis1])
    
    residualsm = np.zeros([naxis2,naxis1])
    redchi = np.zeros([naxis2,naxis1])
    redchi_w = np.zeros([naxis2,naxis1])
    bic = np.zeros([naxis2,naxis1])
    
    residual = np.zeros([len(results[0,0].residual),naxis2,naxis1])
    model = np.zeros([len(results[0,0].residual),naxis2,naxis1])
    model_2g_n = np.zeros([len(results[0,0].residual),naxis2,naxis1])
    model_2g_b = np.zeros([len(results[0,0].residual),naxis2,naxis1])
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
            
            if (ncomps_flag[j,i] == 1) and (bic_flag[j,i] == 2):                
                fluxm_b[j,i] = results[j,i].params['flux_b'].value
                velm_b[j,i] = results[j,i].params['vel_b'].value
                sigm_b[j,i] = results[j,i].params['sig_b'].value
                
                fluxm_b_err[j,i] = results[j,i].params['flux_b'].stderr
                velm_b_err[j,i] = results[j,i].params['vel_b'].stderr
                sigm_b_err[j,i] = results[j,i].params['sig_b'].stderr
                
                model_2g_n[:,j,i] = (am[j,i] + (bm[j,i]*lam_r)) + (fluxm[j,i] / (np.sqrt(2*np.pi) * (np.sqrt((sigm[j,i]**2)+(inst_sigma_red**2))*results[j,i].params['lam01'].value/c))) * np.exp(-(lam_r-((velm[j,i]*results[j,i].params['lam01'].value/c)+results[j,i].params['lam01'].value))**2 / (2*(np.sqrt((sigm[j,i]**2) +(inst_sigma_red**2))*results[j,i].params['lam01'].value/c)**2)) +      ((fluxm[j,i]/3.05) / (np.sqrt(2*np.pi) * (np.sqrt((sigm[j,i]**2)+(inst_sigma_red**2))*results[j,i].params['lam02'].value/c))) * np.exp(-(lam_r-((velm[j,i]*results[j,i].params['lam02'].value/c)+results[j,i].params['lam02'].value))**2 / (2*(np.sqrt((sigm[j,i]**2)+(inst_sigma_red**2))*results[j,i].params['lam02'].value/c)**2))
                
                model_2g_b[:,j,i] = (am[j,i] + (bm[j,i]*lam_r)) + (fluxm_b[j,i] / (np.sqrt(2*np.pi) * (np.sqrt((sigm_b[j,i]**2)+(inst_sigma_red**2))*results[j,i].params['lam01'].value/c))) * np.exp(-(lam_r-((velm_b[j,i]*results[j,i].params['lam01'].value/c)+results[j,i].params['lam01'].value))**2 / (2*(np.sqrt((sigm_b[j,i]**2) +(inst_sigma_red**2))*results[j,i].params['lam01'].value/c)**2)) +      ((fluxm_b[j,i]/3.05) / (np.sqrt(2*np.pi) * (np.sqrt((sigm_b[j,i]**2)+(inst_sigma_red**2))*results[j,i].params['lam02'].value/c))) * np.exp(-(lam_r-((velm_b[j,i]*results[j,i].params['lam02'].value/c)+results[j,i].params['lam02'].value))**2 / (2*(np.sqrt((sigm_b[j,i]**2)+(inst_sigma_red**2))*results[j,i].params['lam02'].value/c)**2))
            
            residualsm[j,i] = np.abs(np.sum(results[j,i].residual))
            redchi[j,i] = results[j,i].redchi
            redchi_w[j,i] = results[j,i].redchi*np.abs(residualsm[j,i])
            bic[j,i] = results[j,i].bic
                        
            residual[:,j,i] = results[j,i].residual
            data[:,j,i] = results[j,i].data
            model[:,j,i] = results[j,i].best_fit
            
            xm[j,i] = i
            ym[j,i] = j
            

    fits.writeto(results_dir+'/fluxm.fits',fluxm,overwrite=True)
    fits.writeto(results_dir+'/fluxm_b.fits',fluxm_b,overwrite=True)
    fits.writeto(results_dir+'/velm.fits',velm,overwrite=True)
    fits.writeto(results_dir+'/sigm.fits',sigm,overwrite=True)
    fits.writeto(results_dir+'/velm_b.fits',velm_b,overwrite=True)
    fits.writeto(results_dir+'/sigm_b.fits',sigm_b,overwrite=True)

    fits.writeto(results_dir+'/flux_err.fits',fluxm_err,overwrite=True)
    fits.writeto(results_dir+'/flux_b_err.fits',fluxm_b_err,overwrite=True)
    fits.writeto(results_dir+'/vel_err.fits',velm_err,overwrite=True)
    fits.writeto(results_dir+'/sig_err.fits',sigm_err,overwrite=True)
    fits.writeto(results_dir+'/vel_b_err.fits',velm_b_err,overwrite=True)
    fits.writeto(results_dir+'/sig_b_err.fits',sigm_b_err,overwrite=True)
    
    fits.writeto(results_dir+'/residuals.fits',residualsm,overwrite=True)
    fits.writeto(results_dir+'/redchi.fits',redchi,overwrite=True)
    fits.writeto(results_dir+'/redchi_w.fits',redchi_w,overwrite=True)
    fits.writeto(results_dir+'/bic.fits',bic,overwrite=True)
    
    fits.writeto(results_dir+'/ncomps_flag.fits',ncomps_flag,overwrite=True)
    fits.writeto(results_dir+'/bic_flag.fits',bic_flag,overwrite=True)

    xm = np.ravel(xm)
    ym = np.ravel(ym)

    am = np.ravel(am)
    bm = np.ravel(bm)
    fluxm = np.ravel(fluxm)
    fluxm_b = np.ravel(fluxm_b)
    velm = np.ravel(velm)
    sigm = np.ravel(sigm)
    velm_b = np.ravel(velm_b)
    sigm_b = np.ravel(sigm_b)

    am_err = np.ravel(am_err)
    bm_err = np.ravel(bm_err)
    fluxm_err = np.ravel(fluxm_err)
    fluxm_b_err = np.ravel(fluxm_b_err)
    velm_err = np.ravel(velm_err)
    sigm_err = np.ravel(sigm_err)
    velm_b_err = np.ravel(velm_b_err)
    sigm_b_err = np.ravel(sigm_b_err)
    
    residualsm = np.ravel(residualsm)

    class t_res(tables.IsDescription):
        
        x = tables.IntCol(shape=(len(result_list)))
        y = tables.IntCol(shape=(len(result_list)))
        
        a = tables.Float64Col(shape=(len(result_list)))
        b = tables.Float64Col(shape=(len(result_list)))
        flux = tables.Float64Col(shape=(len(result_list)))
        flux_b = tables.Float64Col(shape=(len(result_list)))
        vel = tables.Float64Col(shape=(len(result_list)))
        sig = tables.Float64Col(shape=(len(result_list)))
        vel_b = tables.Float64Col(shape=(len(result_list)))
        sig_b = tables.Float64Col(shape=(len(result_list)))
        
        a_err = tables.Float64Col(shape=(len(result_list)))
        b_err = tables.Float64Col(shape=(len(result_list)))
        flux_err = tables.Float64Col(shape=(len(result_list)))
        flux_b_err = tables.Float64Col(shape=(len(result_list)))
        vel_err = tables.Float64Col(shape=(len(result_list)))
        sig_err = tables.Float64Col(shape=(len(result_list)))
        vel_b_err = tables.Float64Col(shape=(len(result_list)))
        sig_b_err = tables.Float64Col(shape=(len(result_list)))
        
        residuals = tables.Float64Col(shape=(len(result_list)))
        
    hf = tables.open_file(results_dir+'/results.h5', 'w')

    th = hf.create_table('/','results', t_res)

    hr = th.row

    hr['x'] = xm
    hr['y'] = ym

    hr['a'] = am
    hr['b'] = bm
    hr['flux'] = fluxm
    hr['flux_b'] = fluxm_b
    hr['vel'] = velm
    hr['sig'] = sigm
    hr['vel_b'] = velm_b
    hr['sig_b'] = sigm_b

    hr['a_err'] = am_err
    hr['b_err'] = bm_err
    hr['flux_err'] = fluxm_err
    hr['flux_b_err'] = fluxm_b_err
    hr['vel_err'] = velm_err
    hr['sig_err'] = sigm_err
    hr['vel_b_err'] = velm_b_err
    hr['sig_b_err'] = sigm_b_err
    
    hr['residuals'] = residualsm

    hr.append()

    hf.flush()
    hf.close()
    
    ft = h5py.File(results_dir+'/fit.hdf5', 'w')
    ft.create_dataset('residual', data=residual)
    ft.create_dataset('model', data=model)
    ft.create_dataset('model_2g_n', data=model_2g_n)
    ft.create_dataset('model_2g_b', data=model_2g_b)
    ft.create_dataset('data', data=data)
    ft.create_dataset('lam', data=lam_r)
    ft.close()

    return


def two_gaussians_2g_cons_o3(naxis1,naxis2,result_list,results_dir,lam_r,ncomps_flag,bic_flag):

    results = np.reshape(result_list, (naxis2,naxis1))

    am = np.zeros([naxis2,naxis1])
    bm = np.zeros([naxis2,naxis1])
    fluxm = np.zeros([naxis2,naxis1])
    fluxm_b = np.zeros([naxis2,naxis1])
    velm = np.zeros([naxis2,naxis1])
    sigm = np.zeros([naxis2,naxis1])
    velm_b = np.zeros([naxis2,naxis1])
    sigm_b = np.zeros([naxis2,naxis1])

    am_err = np.zeros([naxis2,naxis1])
    bm_err = np.zeros([naxis2,naxis1])
    fluxm_err = np.zeros([naxis2,naxis1])
    fluxm_b_err = np.zeros([naxis2,naxis1])
    velm_err = np.zeros([naxis2,naxis1])
    sigm_err = np.zeros([naxis2,naxis1])
    velm_b_err = np.zeros([naxis2,naxis1])
    sigm_b_err = np.zeros([naxis2,naxis1])
    
    residualsm = np.zeros([naxis2,naxis1])
    redchi = np.zeros([naxis2,naxis1])
    redchi_w = np.zeros([naxis2,naxis1])
    bic = np.zeros([naxis2,naxis1])
    
    residual = np.zeros([len(results[0,0].residual),naxis2,naxis1])
    model = np.zeros([len(results[0,0].residual),naxis2,naxis1])
    model_2g_n = np.zeros([len(results[0,0].residual),naxis2,naxis1])
    model_2g_b = np.zeros([len(results[0,0].residual),naxis2,naxis1])
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
            
            if (ncomps_flag[j,i] == 1) and (bic_flag[j,i] == 2):                
                fluxm_b[j,i] = results[j,i].params['flux_b'].value
                velm_b[j,i] = results[j,i].params['vel_b'].value
                sigm_b[j,i] = results[j,i].params['sig_b'].value
                
                fluxm_b_err[j,i] = results[j,i].params['flux_b'].stderr
                velm_b_err[j,i] = results[j,i].params['vel_b'].stderr
                sigm_b_err[j,i] = results[j,i].params['sig_b'].stderr
                
                model_2g_n[:,j,i] = (am[j,i] + (bm[j,i]*lam_r)) + ((fluxm[j,i]/2.94) / (np.sqrt(2*np.pi) * (np.sqrt((sigm[j,i]**2)+(inst_sigma_red**2))*results[j,i].params['lam01'].value/c))) * np.exp(-(lam_r-((velm[j,i]*results[j,i].params['lam01'].value/c)+results[j,i].params['lam01'].value))**2 / (2*(np.sqrt((sigm[j,i]**2) +(inst_sigma_red**2))*results[j,i].params['lam01'].value/c)**2)) +      (fluxm[j,i] / (np.sqrt(2*np.pi) * (np.sqrt((sigm[j,i]**2)+(inst_sigma_red**2))*results[j,i].params['lam02'].value/c))) * np.exp(-(lam_r-((velm[j,i]*results[j,i].params['lam02'].value/c)+results[j,i].params['lam02'].value))**2 / (2*(np.sqrt((sigm[j,i]**2)+(inst_sigma_red**2))*results[j,i].params['lam02'].value/c)**2))
                
                model_2g_b[:,j,i] = (am[j,i] + (bm[j,i]*lam_r)) + ((fluxm_b[j,i]/2.94) / (np.sqrt(2*np.pi) * (np.sqrt((sigm_b[j,i]**2)+(inst_sigma_red**2))*results[j,i].params['lam01'].value/c))) * np.exp(-(lam_r-((velm_b[j,i]*results[j,i].params['lam01'].value/c)+results[j,i].params['lam01'].value))**2 / (2*(np.sqrt((sigm_b[j,i]**2) +(inst_sigma_red**2))*results[j,i].params['lam01'].value/c)**2)) +      (fluxm_b[j,i] / (np.sqrt(2*np.pi) * (np.sqrt((sigm_b[j,i]**2)+(inst_sigma_red**2))*results[j,i].params['lam02'].value/c))) * np.exp(-(lam_r-((velm_b[j,i]*results[j,i].params['lam02'].value/c)+results[j,i].params['lam02'].value))**2 / (2*(np.sqrt((sigm_b[j,i]**2)+(inst_sigma_red**2))*results[j,i].params['lam02'].value/c)**2))
            
            residualsm[j,i] = np.abs(np.sum(results[j,i].residual))
            redchi[j,i] = results[j,i].redchi
            redchi_w[j,i] = results[j,i].redchi*np.abs(residualsm[j,i])
            bic[j,i] = results[j,i].bic
                        
            residual[:,j,i] = results[j,i].residual
            data[:,j,i] = results[j,i].data
            model[:,j,i] = results[j,i].best_fit
            
            xm[j,i] = i
            ym[j,i] = j
            

    fits.writeto(results_dir+'/fluxm.fits',fluxm,overwrite=True)
    fits.writeto(results_dir+'/fluxm_b.fits',fluxm_b,overwrite=True)
    fits.writeto(results_dir+'/velm.fits',velm,overwrite=True)
    fits.writeto(results_dir+'/sigm.fits',sigm,overwrite=True)
    fits.writeto(results_dir+'/velm_b.fits',velm_b,overwrite=True)
    fits.writeto(results_dir+'/sigm_b.fits',sigm_b,overwrite=True)

    fits.writeto(results_dir+'/flux_err.fits',fluxm_err,overwrite=True)
    fits.writeto(results_dir+'/flux_b_err.fits',fluxm_b_err,overwrite=True)
    fits.writeto(results_dir+'/vel_err.fits',velm_err,overwrite=True)
    fits.writeto(results_dir+'/sig_err.fits',sigm_err,overwrite=True)
    fits.writeto(results_dir+'/vel_b_err.fits',velm_b_err,overwrite=True)
    fits.writeto(results_dir+'/sig_b_err.fits',sigm_b_err,overwrite=True)
    
    fits.writeto(results_dir+'/residuals.fits',residualsm,overwrite=True)
    fits.writeto(results_dir+'/redchi.fits',redchi,overwrite=True)
    fits.writeto(results_dir+'/redchi_w.fits',redchi_w,overwrite=True)
    fits.writeto(results_dir+'/bic.fits',bic,overwrite=True)
    
    fits.writeto(results_dir+'/ncomps_flag.fits',ncomps_flag,overwrite=True)
    fits.writeto(results_dir+'/bic_flag.fits',bic_flag,overwrite=True)

    xm = np.ravel(xm)
    ym = np.ravel(ym)

    am = np.ravel(am)
    bm = np.ravel(bm)
    fluxm = np.ravel(fluxm)
    fluxm_b = np.ravel(fluxm_b)
    velm = np.ravel(velm)
    sigm = np.ravel(sigm)
    velm_b = np.ravel(velm_b)
    sigm_b = np.ravel(sigm_b)

    am_err = np.ravel(am_err)
    bm_err = np.ravel(bm_err)
    fluxm_err = np.ravel(fluxm_err)
    fluxm_b_err = np.ravel(fluxm_b_err)
    velm_err = np.ravel(velm_err)
    sigm_err = np.ravel(sigm_err)
    velm_b_err = np.ravel(velm_b_err)
    sigm_b_err = np.ravel(sigm_b_err)
    
    residualsm = np.ravel(residualsm)

    class t_res(tables.IsDescription):
        
        x = tables.IntCol(shape=(len(result_list)))
        y = tables.IntCol(shape=(len(result_list)))
        
        a = tables.Float64Col(shape=(len(result_list)))
        b = tables.Float64Col(shape=(len(result_list)))
        flux = tables.Float64Col(shape=(len(result_list)))
        flux_b = tables.Float64Col(shape=(len(result_list)))
        vel = tables.Float64Col(shape=(len(result_list)))
        sig = tables.Float64Col(shape=(len(result_list)))
        vel_b = tables.Float64Col(shape=(len(result_list)))
        sig_b = tables.Float64Col(shape=(len(result_list)))
        
        a_err = tables.Float64Col(shape=(len(result_list)))
        b_err = tables.Float64Col(shape=(len(result_list)))
        flux_err = tables.Float64Col(shape=(len(result_list)))
        flux_b_err = tables.Float64Col(shape=(len(result_list)))
        vel_err = tables.Float64Col(shape=(len(result_list)))
        sig_err = tables.Float64Col(shape=(len(result_list)))
        vel_b_err = tables.Float64Col(shape=(len(result_list)))
        sig_b_err = tables.Float64Col(shape=(len(result_list)))
        
        residuals = tables.Float64Col(shape=(len(result_list)))
        
    hf = tables.open_file(results_dir+'/results.h5', 'w')

    th = hf.create_table('/','results', t_res)

    hr = th.row

    hr['x'] = xm
    hr['y'] = ym

    hr['a'] = am
    hr['b'] = bm
    hr['flux'] = fluxm
    hr['flux_b'] = fluxm_b
    hr['vel'] = velm
    hr['sig'] = sigm
    hr['vel_b'] = velm_b
    hr['sig_b'] = sigm_b

    hr['a_err'] = am_err
    hr['b_err'] = bm_err
    hr['flux_err'] = fluxm_err
    hr['flux_b_err'] = fluxm_b_err
    hr['vel_err'] = velm_err
    hr['sig_err'] = sigm_err
    hr['vel_b_err'] = velm_b_err
    hr['sig_b_err'] = sigm_b_err
    
    hr['residuals'] = residualsm

    hr.append()

    hf.flush()
    hf.close()
    
    ft = h5py.File(results_dir+'/fit.hdf5', 'w')
    ft.create_dataset('residual', data=residual)
    ft.create_dataset('model', data=model)
    ft.create_dataset('model_2g_n', data=model_2g_n)
    ft.create_dataset('model_2g_b', data=model_2g_b)
    ft.create_dataset('data', data=data)
    ft.create_dataset('lam', data=lam_r)
    ft.close()

    return

def two_gaussians_cons_s2(naxis1,naxis2,result_list,results_dir,lam_r):

    results = np.reshape(result_list, (naxis2,naxis1))

    am = np.zeros([naxis2,naxis1])
    bm = np.zeros([naxis2,naxis1])
    fluxm1 = np.zeros([naxis2,naxis1])
    fluxm2 = np.zeros([naxis2,naxis1])
    velm = np.zeros([naxis2,naxis1])
    sigm = np.zeros([naxis2,naxis1])

    am_err = np.zeros([naxis2,naxis1])
    bm_err = np.zeros([naxis2,naxis1])
    fluxm1_err = np.zeros([naxis2,naxis1])
    fluxm2_err = np.zeros([naxis2,naxis1])
    velm_err = np.zeros([naxis2,naxis1])
    sigm_err = np.zeros([naxis2,naxis1])
    
    residualsm = np.zeros([naxis2,naxis1])
    redchi = np.zeros([naxis2,naxis1])
    redchi_w = np.zeros([naxis2,naxis1])
    
    residual = np.zeros([len(results[0,0].residual),naxis2,naxis1])
    model = np.zeros([len(results[0,0].residual),naxis2,naxis1])
    data = np.zeros([len(results[0,0].residual),naxis2,naxis1])

    xm = np.zeros([naxis2,naxis1])
    ym = np.zeros([naxis2,naxis1])

    for i in np.arange(naxis1):
        for j in np.arange(naxis2):
            am[j,i] = results[j,i].params['a'].value
            bm[j,i] = results[j,i].params['b'].value
            fluxm1[j,i] = results[j,i].params['flux'].value
            fluxm2[j,i] = results[j,i].params['flux'].value/results[j,i].params['ratio'].value
            velm[j,i] = results[j,i].params['vel'].value
            sigm[j,i] = results[j,i].params['sig'].value
            
            am_err[j,i] = results[j,i].params['a'].stderr
            bm_err[j,i] = results[j,i].params['b'].stderr
            fluxm1_err[j,i] = results[j,i].params['flux'].stderr
            fluxm2_err[j,i] = results[j,i].params['ratio'].stderr
            velm_err[j,i] = results[j,i].params['vel'].stderr
            sigm_err[j,i] = results[j,i].params['sig'].stderr
            
            residualsm[j,i] = np.abs(np.sum(results[j,i].residual))
            redchi[j,i] = results[j,i].redchi
            redchi_w[j,i] = results[j,i].redchi*np.abs(residualsm[j,i])
                        
            residual[:,j,i] = results[j,i].residual
            data[:,j,i] = results[j,i].data
            model[:,j,i] = results[j,i].best_fit
            
            xm[j,i] = i
            ym[j,i] = j

    fits.writeto(results_dir+'/fluxm1.fits',fluxm1,overwrite=True)
    fits.writeto(results_dir+'/fluxm2.fits',fluxm2,overwrite=True)
    fits.writeto(results_dir+'/velm.fits',velm,overwrite=True)
    fits.writeto(results_dir+'/sigm.fits',sigm,overwrite=True)

    fits.writeto(results_dir+'/flux1_err.fits',fluxm1_err,overwrite=True)
    fits.writeto(results_dir+'/flux2_err.fits',fluxm2_err,overwrite=True)
    fits.writeto(results_dir+'/vel_err.fits',velm_err,overwrite=True)
    fits.writeto(results_dir+'/sig_err.fits',sigm_err,overwrite=True)
    
    fits.writeto(results_dir+'/residuals.fits',residualsm,overwrite=True)
    fits.writeto(results_dir+'/redchi.fits',redchi,overwrite=True)
    fits.writeto(results_dir+'/redchi_w.fits',redchi_w,overwrite=True)

    xm = np.ravel(xm)
    ym = np.ravel(ym)

    am = np.ravel(am)
    bm = np.ravel(bm)
    fluxm1 = np.ravel(fluxm1)
    fluxm2 = np.ravel(fluxm2)
    velm = np.ravel(velm)
    sigm = np.ravel(sigm)

    am_err = np.ravel(am_err)
    bm_err = np.ravel(bm_err)
    fluxm1_err = np.ravel(fluxm1_err)
    fluxm2_err = np.ravel(fluxm2_err)
    velm_err = np.ravel(velm_err)
    sigm_err = np.ravel(sigm_err)
    
    residualsm = np.ravel(residualsm)

    class t_res(tables.IsDescription):
        
        x = tables.IntCol(shape=(len(result_list)))
        y = tables.IntCol(shape=(len(result_list)))
        
        a = tables.Float64Col(shape=(len(result_list)))
        b = tables.Float64Col(shape=(len(result_list)))
        flux1 = tables.Float64Col(shape=(len(result_list)))
        flux2 = tables.Float64Col(shape=(len(result_list)))
        vel = tables.Float64Col(shape=(len(result_list)))
        sig = tables.Float64Col(shape=(len(result_list)))
        
        a_err = tables.Float64Col(shape=(len(result_list)))
        b_err = tables.Float64Col(shape=(len(result_list)))
        flux1_err = tables.Float64Col(shape=(len(result_list)))
        flux2_err = tables.Float64Col(shape=(len(result_list)))
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
    hr['flux1'] = fluxm1
    hr['flux2'] = fluxm2
    hr['vel'] = velm
    hr['sig'] = sigm

    hr['a_err'] = am_err
    hr['b_err'] = bm_err
    hr['flux1_err'] = fluxm1_err
    hr['flux2_err'] = fluxm2_err
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
    ft.create_dataset('lam', data=lam_r)
    ft.close()

    return

def two_gaussians_2g_cons_s2(naxis1,naxis2,result_list,results_dir,lam_r,ncomps_flag,bic_flag):

    results = np.reshape(result_list, (naxis2,naxis1))

    am = np.zeros([naxis2,naxis1])
    bm = np.zeros([naxis2,naxis1])
    fluxm1 = np.zeros([naxis2,naxis1])
    fluxm2 = np.zeros([naxis2,naxis1])
    fluxm1_b = np.zeros([naxis2,naxis1])
    fluxm2_b = np.zeros([naxis2,naxis1])
    velm = np.zeros([naxis2,naxis1])
    sigm = np.zeros([naxis2,naxis1])
    velm_b = np.zeros([naxis2,naxis1])
    sigm_b = np.zeros([naxis2,naxis1])

    am_err = np.zeros([naxis2,naxis1])
    bm_err = np.zeros([naxis2,naxis1])
    fluxm1_err = np.zeros([naxis2,naxis1])
    fluxm2_err = np.zeros([naxis2,naxis1])
    fluxm1_b_err = np.zeros([naxis2,naxis1])
    fluxm2_b_err = np.zeros([naxis2,naxis1])
    velm_err = np.zeros([naxis2,naxis1])
    sigm_err = np.zeros([naxis2,naxis1])
    velm_b_err = np.zeros([naxis2,naxis1])
    sigm_b_err = np.zeros([naxis2,naxis1])
    
    residualsm = np.zeros([naxis2,naxis1])
    redchi = np.zeros([naxis2,naxis1])
    redchi_w = np.zeros([naxis2,naxis1])
    bic = np.zeros([naxis2,naxis1])
    
    residual = np.zeros([len(results[0,0].residual),naxis2,naxis1])
    model = np.zeros([len(results[0,0].residual),naxis2,naxis1])
    model_2g_n = np.zeros([len(results[0,0].residual),naxis2,naxis1])
    model_2g_b = np.zeros([len(results[0,0].residual),naxis2,naxis1])
    data = np.zeros([len(results[0,0].residual),naxis2,naxis1])

    xm = np.zeros([naxis2,naxis1])
    ym = np.zeros([naxis2,naxis1])

    for i in np.arange(naxis1):
        for j in np.arange(naxis2):
            am[j,i] = results[j,i].params['a'].value
            bm[j,i] = results[j,i].params['b'].value
            fluxm1[j,i] = results[j,i].params['flux'].value
            fluxm2[j,i] = results[j,i].params['flux'].value/results[j,i].params['ratio'].value
            velm[j,i] = results[j,i].params['vel'].value
            sigm[j,i] = results[j,i].params['sig'].value
            
            am_err[j,i] = results[j,i].params['a'].stderr
            bm_err[j,i] = results[j,i].params['b'].stderr
            fluxm1_err[j,i] = results[j,i].params['flux'].stderr
            fluxm2_err[j,i] = results[j,i].params['ratio'].stderr
            velm_err[j,i] = results[j,i].params['vel'].stderr
            sigm_err[j,i] = results[j,i].params['sig'].stderr
            
            if (ncomps_flag[j,i] == 1) and (bic_flag[j,i] == 2):                
                fluxm1_b[j,i] = results[j,i].params['flux_b'].value
                fluxm2_b[j,i] = results[j,i].params['flux_b'].value/results[j,i].params['ratio_b'].value
                velm_b[j,i] = results[j,i].params['vel_b'].value
                sigm_b[j,i] = results[j,i].params['sig_b'].value
                
                fluxm1_b_err[j,i] = results[j,i].params['flux_b'].stderr
                fluxm2_b_err[j,i] = results[j,i].params['ratio_b'].stderr
                velm_b_err[j,i] = results[j,i].params['vel_b'].stderr
                sigm_b_err[j,i] = results[j,i].params['sig_b'].stderr
                
                model_2g_n[:,j,i] = (am[j,i] + (bm[j,i]*lam_r)) + (fluxm1[j,i] / (np.sqrt(2*np.pi) * (np.sqrt((sigm[j,i]**2)+(inst_sigma_red**2))*results[j,i].params['lam01'].value/c))) * np.exp(-(lam_r-((velm[j,i]*results[j,i].params['lam01'].value/c)+results[j,i].params['lam01'].value))**2 / (2*(np.sqrt((sigm[j,i]**2) +(inst_sigma_red**2))*results[j,i].params['lam01'].value/c)**2)) +      (fluxm2[j,i] / (np.sqrt(2*np.pi) * (np.sqrt((sigm[j,i]**2)+(inst_sigma_red**2))*results[j,i].params['lam02'].value/c))) * np.exp(-(lam_r-((velm[j,i]*results[j,i].params['lam02'].value/c)+results[j,i].params['lam02'].value))**2 / (2*(np.sqrt((sigm[j,i]**2)+(inst_sigma_red**2))*results[j,i].params['lam02'].value/c)**2))
                
                model_2g_b[:,j,i] = (am[j,i] + (bm[j,i]*lam_r)) + (fluxm1_b[j,i] / (np.sqrt(2*np.pi) * (np.sqrt((sigm_b[j,i]**2)+(inst_sigma_red**2))*results[j,i].params['lam01'].value/c))) * np.exp(-(lam_r-((velm_b[j,i]*results[j,i].params['lam01'].value/c)+results[j,i].params['lam01'].value))**2 / (2*(np.sqrt((sigm_b[j,i]**2) +(inst_sigma_red**2))*results[j,i].params['lam01'].value/c)**2)) +      (fluxm2_b[j,i] / (np.sqrt(2*np.pi) * (np.sqrt((sigm_b[j,i]**2)+(inst_sigma_red**2))*results[j,i].params['lam02'].value/c))) * np.exp(-(lam_r-((velm_b[j,i]*results[j,i].params['lam02'].value/c)+results[j,i].params['lam02'].value))**2 / (2*(np.sqrt((sigm_b[j,i]**2)+(inst_sigma_red**2))*results[j,i].params['lam02'].value/c)**2))
            
            residualsm[j,i] = np.abs(np.sum(results[j,i].residual))
            redchi[j,i] = results[j,i].redchi
            redchi_w[j,i] = results[j,i].redchi*np.abs(residualsm[j,i])
            bic[j,i] = results[j,i].bic
                        
            residual[:,j,i] = results[j,i].residual
            data[:,j,i] = results[j,i].data
            model[:,j,i] = results[j,i].best_fit
            
            xm[j,i] = i
            ym[j,i] = j
            

    fits.writeto(results_dir+'/fluxm1.fits',fluxm1,overwrite=True)
    fits.writeto(results_dir+'/fluxm2.fits',fluxm2,overwrite=True)
    fits.writeto(results_dir+'/fluxm1_b.fits',fluxm1_b,overwrite=True)
    fits.writeto(results_dir+'/fluxm2_b.fits',fluxm2_b,overwrite=True)
    fits.writeto(results_dir+'/velm.fits',velm,overwrite=True)
    fits.writeto(results_dir+'/sigm.fits',sigm,overwrite=True)
    fits.writeto(results_dir+'/velm_b.fits',velm_b,overwrite=True)
    fits.writeto(results_dir+'/sigm_b.fits',sigm_b,overwrite=True)

    fits.writeto(results_dir+'/flux1_err.fits',fluxm1_err,overwrite=True)
    fits.writeto(results_dir+'/flux2_err.fits',fluxm2_err,overwrite=True)
    fits.writeto(results_dir+'/flux1_b_err.fits',fluxm1_b_err,overwrite=True)
    fits.writeto(results_dir+'/flux2_b_err.fits',fluxm2_b_err,overwrite=True)
    fits.writeto(results_dir+'/vel_err.fits',velm_err,overwrite=True)
    fits.writeto(results_dir+'/sig_err.fits',sigm_err,overwrite=True)
    fits.writeto(results_dir+'/vel_b_err.fits',velm_b_err,overwrite=True)
    fits.writeto(results_dir+'/sig_b_err.fits',sigm_b_err,overwrite=True)
    
    fits.writeto(results_dir+'/residuals.fits',residualsm,overwrite=True)
    fits.writeto(results_dir+'/redchi.fits',redchi,overwrite=True)
    fits.writeto(results_dir+'/redchi_w.fits',redchi_w,overwrite=True)
    fits.writeto(results_dir+'/bic.fits',bic,overwrite=True)
    
    fits.writeto(results_dir+'/ncomps_flag.fits',ncomps_flag,overwrite=True)
    fits.writeto(results_dir+'/bic_flag.fits',bic_flag,overwrite=True)

    xm = np.ravel(xm)
    ym = np.ravel(ym)

    am = np.ravel(am)
    bm = np.ravel(bm)
    fluxm1 = np.ravel(fluxm1)
    fluxm2 = np.ravel(fluxm2)
    fluxm1_b = np.ravel(fluxm1_b)
    fluxm2_b = np.ravel(fluxm2_b)
    velm = np.ravel(velm)
    sigm = np.ravel(sigm)
    velm_b = np.ravel(velm_b)
    sigm_b = np.ravel(sigm_b)

    am_err = np.ravel(am_err)
    bm_err = np.ravel(bm_err)
    fluxm1_err = np.ravel(fluxm1_err)
    fluxm2_err = np.ravel(fluxm2_err)
    fluxm1_b_err = np.ravel(fluxm1_b_err)
    fluxm2_b_err = np.ravel(fluxm2_b_err)
    velm_err = np.ravel(velm_err)
    sigm_err = np.ravel(sigm_err)
    velm_b_err = np.ravel(velm_b_err)
    sigm_b_err = np.ravel(sigm_b_err)
    
    residualsm = np.ravel(residualsm)

    class t_res(tables.IsDescription):
        
        x = tables.IntCol(shape=(len(result_list)))
        y = tables.IntCol(shape=(len(result_list)))
        
        a = tables.Float64Col(shape=(len(result_list)))
        b = tables.Float64Col(shape=(len(result_list)))
        flux1 = tables.Float64Col(shape=(len(result_list)))
        flux2 = tables.Float64Col(shape=(len(result_list)))
        flux1_b = tables.Float64Col(shape=(len(result_list)))
        flux2_b = tables.Float64Col(shape=(len(result_list)))
        vel = tables.Float64Col(shape=(len(result_list)))
        sig = tables.Float64Col(shape=(len(result_list)))
        vel_b = tables.Float64Col(shape=(len(result_list)))
        sig_b = tables.Float64Col(shape=(len(result_list)))
        
        a_err = tables.Float64Col(shape=(len(result_list)))
        b_err = tables.Float64Col(shape=(len(result_list)))
        flux1_err = tables.Float64Col(shape=(len(result_list)))
        flux2_err = tables.Float64Col(shape=(len(result_list)))
        flux1_b_err = tables.Float64Col(shape=(len(result_list)))
        flux2_b_err = tables.Float64Col(shape=(len(result_list)))
        vel_err = tables.Float64Col(shape=(len(result_list)))
        sig_err = tables.Float64Col(shape=(len(result_list)))
        vel_b_err = tables.Float64Col(shape=(len(result_list)))
        sig_b_err = tables.Float64Col(shape=(len(result_list)))
        
        residuals = tables.Float64Col(shape=(len(result_list)))
        
    hf = tables.open_file(results_dir+'/results.h5', 'w')

    th = hf.create_table('/','results', t_res)

    hr = th.row

    hr['x'] = xm
    hr['y'] = ym

    hr['a'] = am
    hr['b'] = bm
    hr['flux1'] = fluxm1
    hr['flux2'] = fluxm2
    hr['flux1_b'] = fluxm1_b
    hr['flux2_b'] = fluxm2_b
    hr['vel'] = velm
    hr['sig'] = sigm
    hr['vel_b'] = velm_b
    hr['sig_b'] = sigm_b

    hr['a_err'] = am_err
    hr['b_err'] = bm_err
    hr['flux1_err'] = fluxm1_err
    hr['flux2_err'] = fluxm2_err
    hr['flux1_b_err'] = fluxm1_b_err
    hr['flux2_b_err'] = fluxm2_b_err
    hr['vel_err'] = velm_err
    hr['sig_err'] = sigm_err
    hr['vel_b_err'] = velm_b_err
    hr['sig_b_err'] = sigm_b_err
    
    hr['residuals'] = residualsm

    hr.append()

    hf.flush()
    hf.close()
    
    ft = h5py.File(results_dir+'/fit.hdf5', 'w')
    ft.create_dataset('residual', data=residual)
    ft.create_dataset('model', data=model)
    ft.create_dataset('model_2g_n', data=model_2g_n)
    ft.create_dataset('model_2g_b', data=model_2g_b)
    ft.create_dataset('data', data=data)
    ft.create_dataset('lam', data=lam_r)
    ft.close()

    return


def three_gaussians(naxis1,naxis2,result_list,results_dir,lam_r):

    results = np.reshape(result_list, (naxis2,naxis1))

    am = np.zeros([naxis2,naxis1])
    bm = np.zeros([naxis2,naxis1])
    fluxm1 = np.zeros([naxis2,naxis1])
    fluxm2 = np.zeros([naxis2,naxis1])
    fluxm3 = np.zeros([naxis2,naxis1])
    velm1 = np.zeros([naxis2,naxis1])
    velm2 = np.zeros([naxis2,naxis1])
    velm3 = np.zeros([naxis2,naxis1])
    sigm1 = np.zeros([naxis2,naxis1])
    sigm2 = np.zeros([naxis2,naxis1])
    sigm3 = np.zeros([naxis2,naxis1])

    am_err = np.zeros([naxis2,naxis1])
    bm_err = np.zeros([naxis2,naxis1])
    fluxm1_err = np.zeros([naxis2,naxis1])
    fluxm2_err = np.zeros([naxis2,naxis1])
    fluxm3_err = np.zeros([naxis2,naxis1])
    velm1_err = np.zeros([naxis2,naxis1])
    velm2_err = np.zeros([naxis2,naxis1])
    velm3_err = np.zeros([naxis2,naxis1])
    sigm1_err = np.zeros([naxis2,naxis1])
    sigm2_err = np.zeros([naxis2,naxis1])
    sigm3_err = np.zeros([naxis2,naxis1])
    
    residualsm = np.zeros([naxis2,naxis1])
    redchi = np.zeros([naxis2,naxis1])
    redchi_w = np.zeros([naxis2,naxis1])
    
    residual = np.zeros([len(results[0,0].residual),naxis2,naxis1])
    model = np.zeros([len(results[0,0].residual),naxis2,naxis1])
    data = np.zeros([len(results[0,0].residual),naxis2,naxis1])

    xm = np.zeros([naxis2,naxis1])
    ym = np.zeros([naxis2,naxis1])

    for i in np.arange(naxis1):
        for j in np.arange(naxis2):
            am[j,i] = results[j,i].params['a'].value
            bm[j,i] = results[j,i].params['b'].value
            fluxm1[j,i] = results[j,i].params['flux1'].value
            fluxm2[j,i] = results[j,i].params['flux2'].value
            fluxm3[j,i] = results[j,i].params['flux3'].value
            velm1[j,i] = results[j,i].params['vel1'].value
            velm2[j,i] = results[j,i].params['vel2'].value
            velm3[j,i] = results[j,i].params['vel3'].value
            sigm1[j,i] = results[j,i].params['sig1'].value
            sigm2[j,i] = results[j,i].params['sig2'].value
            sigm3[j,i] = results[j,i].params['sig3'].value

            am_err[j,i] = results[j,i].params['a'].stderr
            bm_err[j,i] = results[j,i].params['b'].stderr
            fluxm1_err[j,i] = results[j,i].params['flux1'].stderr
            fluxm2_err[j,i] = results[j,i].params['flux2'].stderr
            fluxm3_err[j,i] = results[j,i].params['flux3'].stderr
            velm1_err[j,i] = results[j,i].params['vel1'].stderr
            velm2_err[j,i] = results[j,i].params['vel2'].stderr
            velm3_err[j,i] = results[j,i].params['vel3'].stderr
            sigm1_err[j,i] = results[j,i].params['sig1'].stderr
            sigm2_err[j,i] = results[j,i].params['sig2'].stderr
            sigm3_err[j,i] = results[j,i].params['sig3'].stderr
            
            residualsm[j,i] = np.abs(np.sum(results[j,i].residual))
            redchi[j,i] = results[j,i].redchi
            redchi_w[j,i] = results[j,i].redchi*np.abs(residualsm[j,i])
            
            residual[:,j,i] = results[j,i].residual
            data[:,j,i] = results[j,i].data
            model[:,j,i] = results[j,i].best_fit
            
            xm[j,i] = i
            ym[j,i] = j

    fits.writeto(results_dir+'/fluxm1.fits',fluxm1,overwrite=True)
    fits.writeto(results_dir+'/fluxm2.fits',fluxm2,overwrite=True)
    fits.writeto(results_dir+'/fluxm3.fits',fluxm3,overwrite=True)
    fits.writeto(results_dir+'/velm1.fits',velm1,overwrite=True)
    fits.writeto(results_dir+'/velm2.fits',velm2,overwrite=True)
    fits.writeto(results_dir+'/velm3.fits',velm3,overwrite=True)
    fits.writeto(results_dir+'/sigm1.fits',sigm1,overwrite=True)
    fits.writeto(results_dir+'/sigm2.fits',sigm2,overwrite=True)
    fits.writeto(results_dir+'/sigm3.fits',sigm3,overwrite=True)

    fits.writeto(results_dir+'/flux1_err.fits',fluxm1_err,overwrite=True)
    fits.writeto(results_dir+'/flux2_err.fits',fluxm2_err,overwrite=True)
    fits.writeto(results_dir+'/flux3_err.fits',fluxm3_err,overwrite=True)
    fits.writeto(results_dir+'/vel1_err.fits',velm1_err,overwrite=True)
    fits.writeto(results_dir+'/vel2_err.fits',velm2_err,overwrite=True)
    fits.writeto(results_dir+'/vel3_err.fits',velm3_err,overwrite=True)
    fits.writeto(results_dir+'/sig1_err.fits',sigm1_err,overwrite=True)
    fits.writeto(results_dir+'/sig2_err.fits',sigm2_err,overwrite=True)
    fits.writeto(results_dir+'/sig3_err.fits',sigm3_err,overwrite=True)
    
    fits.writeto(results_dir+'/residuals.fits',residualsm,overwrite=True)
    fits.writeto(results_dir+'/redchi.fits',redchi,overwrite=True)
    fits.writeto(results_dir+'/redchi_w.fits',redchi_w,overwrite=True)

    xm = np.ravel(xm)
    ym = np.ravel(ym)

    am = np.ravel(am)
    bm = np.ravel(bm)
    fluxm1 = np.ravel(fluxm1)
    fluxm2 = np.ravel(fluxm2)
    fluxm3 = np.ravel(fluxm3)
    velm1 = np.ravel(velm1)
    velm2 = np.ravel(velm2)
    velm3 = np.ravel(velm3)
    sigm1 = np.ravel(sigm1)
    sigm2 = np.ravel(sigm2)
    sigm3 = np.ravel(sigm3)

    am_err = np.ravel(am_err)
    bm_err = np.ravel(bm_err)
    fluxm1_err = np.ravel(fluxm1_err)
    fluxm2_err = np.ravel(fluxm2_err)
    fluxm3_err = np.ravel(fluxm3_err)
    velm1_err = np.ravel(velm1_err)
    velm2_err = np.ravel(velm2_err)
    velm3_err = np.ravel(velm3_err)
    sigm1_err = np.ravel(sigm1_err)
    sigm2_err = np.ravel(sigm2_err)
    sigm3_err = np.ravel(sigm3_err)
    
    residualsm = np.ravel(residualsm)

    class t_res(tables.IsDescription):
        
        x = tables.IntCol(shape=(len(result_list)))
        y = tables.IntCol(shape=(len(result_list)))
        
        a = tables.Float64Col(shape=(len(result_list)))
        b = tables.Float64Col(shape=(len(result_list)))
        flux1 = tables.Float64Col(shape=(len(result_list)))
        flux2 = tables.Float64Col(shape=(len(result_list)))
        flux3 = tables.Float64Col(shape=(len(result_list)))
        vel1 = tables.Float64Col(shape=(len(result_list)))
        vel2 = tables.Float64Col(shape=(len(result_list)))
        vel3 = tables.Float64Col(shape=(len(result_list)))
        sig1 = tables.Float64Col(shape=(len(result_list)))
        sig2 = tables.Float64Col(shape=(len(result_list)))
        sig3 = tables.Float64Col(shape=(len(result_list)))
        
        a_err = tables.Float64Col(shape=(len(result_list)))
        b_err = tables.Float64Col(shape=(len(result_list)))
        flux1_err = tables.Float64Col(shape=(len(result_list)))
        flux2_err = tables.Float64Col(shape=(len(result_list)))
        flux3_err = tables.Float64Col(shape=(len(result_list)))
        vel1_err = tables.Float64Col(shape=(len(result_list)))
        vel2_err = tables.Float64Col(shape=(len(result_list)))
        vel3_err = tables.Float64Col(shape=(len(result_list)))
        sig1_err = tables.Float64Col(shape=(len(result_list)))
        sig2_err = tables.Float64Col(shape=(len(result_list)))
        sig3_err = tables.Float64Col(shape=(len(result_list)))
        
        residuals = tables.Float64Col(shape=(len(result_list)))
        
    hf = tables.open_file(results_dir+'/results.h5', 'w')

    th = hf.create_table('/','results', t_res)

    hr = th.row

    hr['x'] = xm
    hr['y'] = ym

    hr['a'] = am
    hr['b'] = bm
    hr['flux1'] = fluxm1
    hr['flux2'] = fluxm2
    hr['flux3'] = fluxm3
    hr['vel1'] = velm1
    hr['vel2'] = velm2
    hr['vel3'] = velm3
    hr['sig1'] = sigm1
    hr['sig2'] = sigm2
    hr['sig3'] = sigm3

    hr['a_err'] = am_err
    hr['b_err'] = bm_err
    hr['flux1_err'] = fluxm1_err
    hr['flux2_err'] = fluxm2_err
    hr['flux3_err'] = fluxm3_err
    hr['vel1_err'] = velm1_err
    hr['vel2_err'] = velm2_err
    hr['vel3_err'] = velm3_err
    hr['sig1_err'] = sigm1_err
    hr['sig2_err'] = sigm2_err
    hr['sig3_err'] = sigm3_err
    
    hr['residuals'] = residualsm

    hr.append()

    hf.flush()
    hf.close()
    
    ft = h5py.File(results_dir+'/fit.hdf5', 'w')
    ft.create_dataset('residual', data=residual)
    ft.create_dataset('model', data=model)
    ft.create_dataset('data', data=data)
    ft.create_dataset('lam', data=lam_r)
    ft.close()

    return

def three_gaussians_cons(naxis1,naxis2,result_list,results_dir,lam_r):

    results = np.reshape(result_list, (naxis2,naxis1))

    am = np.zeros([naxis2,naxis1])
    bm = np.zeros([naxis2,naxis1])
    fluxm2 = np.zeros([naxis2,naxis1])
    fluxm3 = np.zeros([naxis2,naxis1])
    velm = np.zeros([naxis2,naxis1])
    sigm = np.zeros([naxis2,naxis1])

    am_err = np.zeros([naxis2,naxis1])
    bm_err = np.zeros([naxis2,naxis1])
    fluxm2_err = np.zeros([naxis2,naxis1])
    fluxm3_err = np.zeros([naxis2,naxis1])
    velm_err = np.zeros([naxis2,naxis1])
    sigm_err = np.zeros([naxis2,naxis1])
    
    residualsm = np.zeros([naxis2,naxis1])
    redchi = np.zeros([naxis2,naxis1])
    redchi_w = np.zeros([naxis2,naxis1])
    
    residual = np.zeros([len(results[0,0].residual),naxis2,naxis1])
    model = np.zeros([len(results[0,0].residual),naxis2,naxis1])
    data = np.zeros([len(results[0,0].residual),naxis2,naxis1])

    xm = np.zeros([naxis2,naxis1])
    ym = np.zeros([naxis2,naxis1])

    for i in np.arange(naxis1):
        for j in np.arange(naxis2):
            am[j,i] = results[j,i].params['a'].value
            bm[j,i] = results[j,i].params['b'].value
            fluxm2[j,i] = results[j,i].params['flux'].value
            fluxm3[j,i] = results[j,i].params['ratio'].value*results[j,i].params['flux'].value
            velm[j,i] = results[j,i].params['vel'].value
            sigm[j,i] = results[j,i].params['sig'].value
            
            am_err[j,i] = results[j,i].params['a'].stderr
            bm_err[j,i] = results[j,i].params['b'].stderr
            fluxm2_err[j,i] = results[j,i].params['flux'].stderr
            fluxm3_err[j,i] = results[j,i].params['ratio'].stderr
            velm_err[j,i] = results[j,i].params['vel'].stderr
            sigm_err[j,i] = results[j,i].params['sig'].stderr
            
            residualsm[j,i] = np.abs(np.sum(results[j,i].residual))
            redchi[j,i] = results[j,i].redchi
            redchi_w[j,i] = results[j,i].redchi*np.abs(residualsm[j,i])
                        
            residual[:,j,i] = results[j,i].residual
            data[:,j,i] = results[j,i].data
            model[:,j,i] = results[j,i].best_fit
            
            xm[j,i] = i
            ym[j,i] = j

    fits.writeto(results_dir+'/fluxm2.fits',fluxm2,overwrite=True)
    fits.writeto(results_dir+'/fluxm3.fits',fluxm3,overwrite=True)
    fits.writeto(results_dir+'/velm.fits',velm,overwrite=True)
    fits.writeto(results_dir+'/sigm.fits',sigm,overwrite=True)

    fits.writeto(results_dir+'/flux2_err.fits',fluxm2_err,overwrite=True)
    fits.writeto(results_dir+'/flux3_err.fits',fluxm3_err,overwrite=True)
    fits.writeto(results_dir+'/vel_err.fits',velm_err,overwrite=True)
    fits.writeto(results_dir+'/sig_err.fits',sigm_err,overwrite=True)
    
    fits.writeto(results_dir+'/residuals.fits',residualsm,overwrite=True)
    fits.writeto(results_dir+'/redchi.fits',redchi,overwrite=True)
    fits.writeto(results_dir+'/redchi_w.fits',redchi_w,overwrite=True)

    xm = np.ravel(xm)
    ym = np.ravel(ym)

    am = np.ravel(am)
    bm = np.ravel(bm)
    fluxm2 = np.ravel(fluxm2)
    fluxm3 = np.ravel(fluxm3)
    velm = np.ravel(velm)
    sigm = np.ravel(sigm)

    am_err = np.ravel(am_err)
    bm_err = np.ravel(bm_err)
    fluxm2_err = np.ravel(fluxm2_err)
    fluxm3_err = np.ravel(fluxm3_err)
    velm_err = np.ravel(velm_err)
    sigm_err = np.ravel(sigm_err)
    
    residualsm = np.ravel(residualsm)

    class t_res(tables.IsDescription):
        
        x = tables.IntCol(shape=(len(result_list)))
        y = tables.IntCol(shape=(len(result_list)))
        
        a = tables.Float64Col(shape=(len(result_list)))
        b = tables.Float64Col(shape=(len(result_list)))
        flux2 = tables.Float64Col(shape=(len(result_list)))
        flux3 = tables.Float64Col(shape=(len(result_list)))
        vel = tables.Float64Col(shape=(len(result_list)))
        sig = tables.Float64Col(shape=(len(result_list)))
        
        a_err = tables.Float64Col(shape=(len(result_list)))
        b_err = tables.Float64Col(shape=(len(result_list)))
        flux2_err = tables.Float64Col(shape=(len(result_list)))
        flux3_err = tables.Float64Col(shape=(len(result_list)))
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
    hr['flux2'] = fluxm2
    hr['flux3'] = fluxm3
    hr['vel'] = velm
    hr['sig'] = sigm

    hr['a_err'] = am_err
    hr['b_err'] = bm_err
    hr['flux2_err'] = fluxm2_err
    hr['flux3_err'] = fluxm3_err
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
    ft.create_dataset('lam', data=lam_r)
    ft.close()

    return

def three_gaussians_2g_cons(naxis1,naxis2,result_list,results_dir,lam_r,ncomps_flag,bic_flag):

    results = np.reshape(result_list, (naxis2,naxis1))

    am = np.zeros([naxis2,naxis1])
    bm = np.zeros([naxis2,naxis1])
    fluxm2 = np.zeros([naxis2,naxis1])
    fluxm3 = np.zeros([naxis2,naxis1])
    fluxm2_b = np.zeros([naxis2,naxis1])
    fluxm3_b = np.zeros([naxis2,naxis1])
    velm = np.zeros([naxis2,naxis1])
    sigm = np.zeros([naxis2,naxis1])
    velm_b = np.zeros([naxis2,naxis1])
    sigm_b = np.zeros([naxis2,naxis1])

    am_err = np.zeros([naxis2,naxis1])
    bm_err = np.zeros([naxis2,naxis1])
    fluxm2_err = np.zeros([naxis2,naxis1])
    fluxm3_err = np.zeros([naxis2,naxis1])
    fluxm2_b_err = np.zeros([naxis2,naxis1])
    fluxm3_b_err = np.zeros([naxis2,naxis1])
    velm_err = np.zeros([naxis2,naxis1])
    sigm_err = np.zeros([naxis2,naxis1])
    velm_b_err = np.zeros([naxis2,naxis1])
    sigm_b_err = np.zeros([naxis2,naxis1])
    
    residualsm = np.zeros([naxis2,naxis1])
    redchi = np.zeros([naxis2,naxis1])
    redchi_w = np.zeros([naxis2,naxis1])
    bic = np.zeros([naxis2,naxis1])
    
    residual = np.zeros([len(results[0,0].residual),naxis2,naxis1])
    model = np.zeros([len(results[0,0].residual),naxis2,naxis1])
    model_2g_n = np.zeros([len(results[0,0].residual),naxis2,naxis1])
    model_2g_b = np.zeros([len(results[0,0].residual),naxis2,naxis1])
    data = np.zeros([len(results[0,0].residual),naxis2,naxis1])

    xm = np.zeros([naxis2,naxis1])
    ym = np.zeros([naxis2,naxis1])

    for i in np.arange(naxis1):
        for j in np.arange(naxis2):
            am[j,i] = results[j,i].params['a'].value
            bm[j,i] = results[j,i].params['b'].value
            fluxm2[j,i] = results[j,i].params['flux'].value
            fluxm3[j,i] = results[j,i].params['ratio'].value*results[j,i].params['flux'].value
            velm[j,i] = results[j,i].params['vel'].value
            sigm[j,i] = results[j,i].params['sig'].value
            
            am_err[j,i] = results[j,i].params['a'].stderr
            bm_err[j,i] = results[j,i].params['b'].stderr
            fluxm2_err[j,i] = results[j,i].params['flux'].stderr
            fluxm3_err[j,i] = results[j,i].params['ratio'].stderr
            velm_err[j,i] = results[j,i].params['vel'].stderr
            sigm_err[j,i] = results[j,i].params['sig'].stderr
            
            if (ncomps_flag[j,i] == 1) and (bic_flag[j,i] == 2):                
                fluxm2_b[j,i] = results[j,i].params['flux_b'].value
                fluxm3_b[j,i] = results[j,i].params['ratio_b'].value*results[j,i].params['flux_b'].value
                velm_b[j,i] = results[j,i].params['vel_b'].value
                sigm_b[j,i] = results[j,i].params['sig_b'].value
                
                fluxm2_b_err[j,i] = results[j,i].params['flux_b'].stderr
                fluxm3_b_err[j,i] = results[j,i].params['ratio_b'].stderr
                velm_b_err[j,i] = results[j,i].params['vel_b'].stderr
                sigm_b_err[j,i] = results[j,i].params['sig_b'].stderr
                
                model_2g_n[:,j,i] = (am[j,i] + (bm[j,i]*lam_r)) + ((fluxm3[j,i]/2.94) / (np.sqrt(2*np.pi) * (np.sqrt((sigm[j,i]**2)+(inst_sigma_red**2))*results[j,i].params['lam01'].value/c))) * np.exp(-(lam_r-((velm[j,i]*results[j,i].params['lam01'].value/c)+results[j,i].params['lam01'].value))**2 / (2*(np.sqrt((sigm[j,i]**2) +(inst_sigma_red**2))*results[j,i].params['lam01'].value/c)**2)) + (fluxm2[j,i] / (np.sqrt(2*np.pi) * (np.sqrt((sigm[j,i]**2)+(inst_sigma_red**2))*results[j,i].params['lam02'].value/c))) * np.exp(-(lam_r-((velm[j,i]*results[j,i].params['lam02'].value/c)+results[j,i].params['lam02'].value))**2 / (2*(np.sqrt((sigm[j,i]**2)+(inst_sigma_red**2))*results[j,i].params['lam02'].value/c)**2)) + (fluxm3[j,i] / (np.sqrt(2*np.pi) * (np.sqrt((sigm[j,i]**2)+(inst_sigma_red**2))*results[j,i].params['lam03'].value/c))) * np.exp(-(lam_r-((velm[j,i]*results[j,i].params['lam03'].value/c)+results[j,i].params['lam03'].value))**2 / (2*(np.sqrt((sigm[j,i]**2)+(inst_sigma_red**2))*results[j,i].params['lam03'].value/c)**2))
                
                model_2g_b[:,j,i] = (am[j,i] + (bm[j,i]*lam_r)) + ((fluxm3_b[j,i]/2.94) / (np.sqrt(2*np.pi) * (np.sqrt((sigm_b[j,i]**2)+(inst_sigma_red**2))*results[j,i].params['lam01'].value/c))) * np.exp(-(lam_r-((velm_b[j,i]*results[j,i].params['lam01'].value/c)+results[j,i].params['lam01'].value))**2 / (2*(np.sqrt((sigm_b[j,i]**2) +(inst_sigma_red**2))*results[j,i].params['lam01'].value/c)**2)) + (fluxm2_b[j,i] / (np.sqrt(2*np.pi) * (np.sqrt((sigm_b[j,i]**2)+(inst_sigma_red**2))*results[j,i].params['lam02'].value/c))) * np.exp(-(lam_r-((velm_b[j,i]*results[j,i].params['lam02'].value/c)+results[j,i].params['lam02'].value))**2 / (2*(np.sqrt((sigm_b[j,i]**2)+(inst_sigma_red**2))*results[j,i].params['lam02'].value/c)**2)) + (fluxm3_b[j,i] / (np.sqrt(2*np.pi) * (np.sqrt((sigm_b[j,i]**2)+(inst_sigma_red**2))*results[j,i].params['lam03'].value/c))) * np.exp(-(lam_r-((velm_b[j,i]*results[j,i].params['lam03'].value/c)+results[j,i].params['lam03'].value))**2 / (2*(np.sqrt((sigm_b[j,i]**2)+(inst_sigma_red**2))*results[j,i].params['lam03'].value/c)**2))
            
            residualsm[j,i] = np.abs(np.sum(results[j,i].residual))
            redchi[j,i] = results[j,i].redchi
            redchi_w[j,i] = results[j,i].redchi*np.abs(residualsm[j,i])
            bic[j,i] = results[j,i].bic
                        
            residual[:,j,i] = results[j,i].residual
            data[:,j,i] = results[j,i].data
            model[:,j,i] = results[j,i].best_fit
            
            xm[j,i] = i
            ym[j,i] = j
            

    fits.writeto(results_dir+'/fluxm2.fits',fluxm2,overwrite=True)
    fits.writeto(results_dir+'/fluxm3.fits',fluxm3,overwrite=True)
    fits.writeto(results_dir+'/fluxm2_b.fits',fluxm2_b,overwrite=True)
    fits.writeto(results_dir+'/fluxm3_b.fits',fluxm3_b,overwrite=True)
    fits.writeto(results_dir+'/velm.fits',velm,overwrite=True)
    fits.writeto(results_dir+'/sigm.fits',sigm,overwrite=True)
    fits.writeto(results_dir+'/velm_b.fits',velm_b,overwrite=True)
    fits.writeto(results_dir+'/sigm_b.fits',sigm_b,overwrite=True)

    fits.writeto(results_dir+'/flux2_err.fits',fluxm2_err,overwrite=True)
    fits.writeto(results_dir+'/flux3_err.fits',fluxm3_err,overwrite=True)
    fits.writeto(results_dir+'/flux2_b_err.fits',fluxm2_b_err,overwrite=True)
    fits.writeto(results_dir+'/flux3_b_err.fits',fluxm3_b_err,overwrite=True)
    fits.writeto(results_dir+'/vel_err.fits',velm_err,overwrite=True)
    fits.writeto(results_dir+'/sig_err.fits',sigm_err,overwrite=True)
    fits.writeto(results_dir+'/vel_b_err.fits',velm_b_err,overwrite=True)
    fits.writeto(results_dir+'/sig_b_err.fits',sigm_b_err,overwrite=True)
    
    fits.writeto(results_dir+'/residuals.fits',residualsm,overwrite=True)
    fits.writeto(results_dir+'/redchi.fits',redchi,overwrite=True)
    fits.writeto(results_dir+'/redchi_w.fits',redchi_w,overwrite=True)
    fits.writeto(results_dir+'/bic.fits',bic,overwrite=True)
    
    fits.writeto(results_dir+'/ncomps_flag.fits',ncomps_flag,overwrite=True)
    fits.writeto(results_dir+'/bic_flag.fits',bic_flag,overwrite=True)

    xm = np.ravel(xm)
    ym = np.ravel(ym)

    am = np.ravel(am)
    bm = np.ravel(bm)
    fluxm2 = np.ravel(fluxm2)
    fluxm3 = np.ravel(fluxm3)
    fluxm2_b = np.ravel(fluxm2_b)
    fluxm3_b = np.ravel(fluxm3_b)
    velm = np.ravel(velm)
    sigm = np.ravel(sigm)
    velm_b = np.ravel(velm_b)
    sigm_b = np.ravel(sigm_b)

    am_err = np.ravel(am_err)
    bm_err = np.ravel(bm_err)
    fluxm2_err = np.ravel(fluxm2_err)
    fluxm3_err = np.ravel(fluxm3_err)
    fluxm2_b_err = np.ravel(fluxm2_b_err)
    fluxm3_b_err = np.ravel(fluxm3_b_err)
    velm_err = np.ravel(velm_err)
    sigm_err = np.ravel(sigm_err)
    velm_b_err = np.ravel(velm_b_err)
    sigm_b_err = np.ravel(sigm_b_err)
    
    residualsm = np.ravel(residualsm)

    class t_res(tables.IsDescription):
        
        x = tables.IntCol(shape=(len(result_list)))
        y = tables.IntCol(shape=(len(result_list)))
        
        a = tables.Float64Col(shape=(len(result_list)))
        b = tables.Float64Col(shape=(len(result_list)))
        flux2 = tables.Float64Col(shape=(len(result_list)))
        flux3 = tables.Float64Col(shape=(len(result_list)))
        flux2_b = tables.Float64Col(shape=(len(result_list)))
        flux3_b = tables.Float64Col(shape=(len(result_list)))
        vel = tables.Float64Col(shape=(len(result_list)))
        sig = tables.Float64Col(shape=(len(result_list)))
        vel_b = tables.Float64Col(shape=(len(result_list)))
        sig_b = tables.Float64Col(shape=(len(result_list)))
        
        a_err = tables.Float64Col(shape=(len(result_list)))
        b_err = tables.Float64Col(shape=(len(result_list)))
        flux2_err = tables.Float64Col(shape=(len(result_list)))
        flux3_err = tables.Float64Col(shape=(len(result_list)))
        flux2_b_err = tables.Float64Col(shape=(len(result_list)))
        flux3_b_err = tables.Float64Col(shape=(len(result_list)))
        vel_err = tables.Float64Col(shape=(len(result_list)))
        sig_err = tables.Float64Col(shape=(len(result_list)))
        vel_b_err = tables.Float64Col(shape=(len(result_list)))
        sig_b_err = tables.Float64Col(shape=(len(result_list)))
        
        residuals = tables.Float64Col(shape=(len(result_list)))
        
    hf = tables.open_file(results_dir+'/results.h5', 'w')

    th = hf.create_table('/','results', t_res)

    hr = th.row

    hr['x'] = xm
    hr['y'] = ym

    hr['a'] = am
    hr['b'] = bm
    hr['flux2'] = fluxm2
    hr['flux3'] = fluxm3
    hr['flux2_b'] = fluxm2_b
    hr['flux3_b'] = fluxm3_b
    hr['vel'] = velm
    hr['sig'] = sigm
    hr['vel_b'] = velm_b
    hr['sig_b'] = sigm_b

    hr['a_err'] = am_err
    hr['b_err'] = bm_err
    hr['flux2_err'] = fluxm2_err
    hr['flux3_err'] = fluxm3_err
    hr['flux2_b_err'] = fluxm2_b_err
    hr['flux3_b_err'] = fluxm3_b_err
    hr['vel_err'] = velm_err
    hr['sig_err'] = sigm_err
    hr['vel_b_err'] = velm_b_err
    hr['sig_b_err'] = sigm_b_err
    
    hr['residuals'] = residualsm

    hr.append()

    hf.flush()
    hf.close()
    
    ft = h5py.File(results_dir+'/fit.hdf5', 'w')
    ft.create_dataset('residual', data=residual)
    ft.create_dataset('model', data=model)
    ft.create_dataset('model_2g_n', data=model_2g_n)
    ft.create_dataset('model_2g_b', data=model_2g_b)
    ft.create_dataset('data', data=data)
    ft.create_dataset('lam', data=lam_r)
    ft.close()

    return

def hb_ha_n2_gaussians_cons(naxis1,naxis2,result_list,results_dir,lam_r):

    results = np.reshape(result_list, (naxis2,naxis1))

    am_hb = np.zeros([naxis2,naxis1])
    am_ha = np.zeros([naxis2,naxis1])
    bm_hb = np.zeros([naxis2,naxis1])
    bm_ha = np.zeros([naxis2,naxis1])
    fluxm1 = np.zeros([naxis2,naxis1])
    fluxm2 = np.zeros([naxis2,naxis1])
    fluxm3 = np.zeros([naxis2,naxis1])
    velm = np.zeros([naxis2,naxis1])
    sigm = np.zeros([naxis2,naxis1])

    am_hb_err = np.zeros([naxis2,naxis1])
    am_ha_err = np.zeros([naxis2,naxis1])
    bm_hb_err = np.zeros([naxis2,naxis1])
    bm_ha_err = np.zeros([naxis2,naxis1])
    fluxm1_err = np.zeros([naxis2,naxis1])
    fluxm2_err = np.zeros([naxis2,naxis1])
    fluxm3_err = np.zeros([naxis2,naxis1])
    velm_err = np.zeros([naxis2,naxis1])
    sigm_err = np.zeros([naxis2,naxis1])
    
    residualsm = np.zeros([naxis2,naxis1])
    redchi = np.zeros([naxis2,naxis1])
    redchi_w = np.zeros([naxis2,naxis1])
    
    residual = np.zeros([len(results[0,0].residual),naxis2,naxis1])
    model = np.zeros([len(results[0,0].residual),naxis2,naxis1])
    data = np.zeros([len(results[0,0].residual),naxis2,naxis1])

    xm = np.zeros([naxis2,naxis1])
    ym = np.zeros([naxis2,naxis1])

    for i in np.arange(naxis1):
        for j in np.arange(naxis2):
            am_hb[j,i] = results[j,i].params['a_hb'].value
            am_ha[j,i] = results[j,i].params['a_ha'].value
            bm_hb[j,i] = results[j,i].params['b_hb'].value
            bm_ha[j,i] = results[j,i].params['b_ha'].value
            fluxm1[j,i] = results[j,i].params['flux'].value/results[j,i].params['ratio_hb'].value
            fluxm2[j,i] = results[j,i].params['flux'].value
            fluxm3[j,i] = results[j,i].params['ratio_n2'].value*results[j,i].params['flux'].value
            velm[j,i] = results[j,i].params['vel'].value
            sigm[j,i] = results[j,i].params['sig'].value
            
            am_hb_err[j,i] = results[j,i].params['a_hb'].stderr
            am_ha_err[j,i] = results[j,i].params['a_ha'].stderr
            bm_hb_err[j,i] = results[j,i].params['b_hb'].stderr
            bm_ha_err[j,i] = results[j,i].params['b_ha'].stderr
            fluxm1_err[j,i] = results[j,i].params['ratio_hb'].stderr
            fluxm2_err[j,i] = results[j,i].params['flux'].stderr
            fluxm3_err[j,i] = results[j,i].params['ratio_n2'].stderr
            velm_err[j,i] = results[j,i].params['vel'].stderr
            sigm_err[j,i] = results[j,i].params['sig'].stderr
            
            residualsm[j,i] = np.abs(np.sum(results[j,i].residual))
            redchi[j,i] = results[j,i].redchi
            redchi_w[j,i] = results[j,i].redchi*np.abs(residualsm[j,i])
                        
            residual[:,j,i] = results[j,i].residual
            data[:,j,i] = results[j,i].data
            model[:,j,i] = results[j,i].best_fit
            
            xm[j,i] = i
            ym[j,i] = j

    fits.writeto(results_dir+'/fluxm1.fits',fluxm1,overwrite=True)
    fits.writeto(results_dir+'/fluxm2.fits',fluxm2,overwrite=True)
    fits.writeto(results_dir+'/fluxm3.fits',fluxm3,overwrite=True)
    fits.writeto(results_dir+'/velm.fits',velm,overwrite=True)
    fits.writeto(results_dir+'/sigm.fits',sigm,overwrite=True)

    fits.writeto(results_dir+'/flux1_err.fits',fluxm1_err,overwrite=True)
    fits.writeto(results_dir+'/flux2_err.fits',fluxm2_err,overwrite=True)
    fits.writeto(results_dir+'/flux3_err.fits',fluxm3_err,overwrite=True)
    fits.writeto(results_dir+'/vel_err.fits',velm_err,overwrite=True)
    fits.writeto(results_dir+'/sig_err.fits',sigm_err,overwrite=True)
    
    fits.writeto(results_dir+'/residuals.fits',residualsm,overwrite=True)
    fits.writeto(results_dir+'/redchi.fits',redchi,overwrite=True)
    fits.writeto(results_dir+'/redchi_w.fits',redchi_w,overwrite=True)
    fits.writeto(results_dir+'/bic.fits',bic,overwrite=True)
    
    fits.writeto(results_dir+'/ncomps_flag.fits',ncomps_flag,overwrite=True)
    fits.writeto(results_dir+'/bic_flag.fits',bic_flag,overwrite=True)

    xm = np.ravel(xm)
    ym = np.ravel(ym)

    am_hb = np.ravel(am_hb)
    am_ha = np.ravel(am_ha)
    bm_hb = np.ravel(bm_hb)
    bm_ha = np.ravel(bm_ha)
    fluxm1 = np.ravel(fluxm1)
    fluxm2 = np.ravel(fluxm2)
    fluxm3 = np.ravel(fluxm3)
    velm = np.ravel(velm)
    sigm = np.ravel(sigm)

    am_hb_err = np.ravel(am_hb_err)
    am_ha_err = np.ravel(am_ha_err)
    bm_hb_err = np.ravel(bm_hb_err)
    bm_ha_err = np.ravel(bm_ha_err)
    fluxm1_err = np.ravel(fluxm1_err)
    fluxm2_err = np.ravel(fluxm2_err)
    fluxm3_err = np.ravel(fluxm3_err)
    velm_err = np.ravel(velm_err)
    sigm_err = np.ravel(sigm_err)
    
    residualsm = np.ravel(residualsm)

    class t_res(tables.IsDescription):
        
        x = tables.IntCol(shape=(len(result_list)))
        y = tables.IntCol(shape=(len(result_list)))
        
        a_hb = tables.Float64Col(shape=(len(result_list)))
        a_ha = tables.Float64Col(shape=(len(result_list)))
        b_hb = tables.Float64Col(shape=(len(result_list)))
        b_ha = tables.Float64Col(shape=(len(result_list)))
        flux1 = tables.Float64Col(shape=(len(result_list)))
        flux2 = tables.Float64Col(shape=(len(result_list)))
        flux3 = tables.Float64Col(shape=(len(result_list)))
        vel = tables.Float64Col(shape=(len(result_list)))
        sig = tables.Float64Col(shape=(len(result_list)))
        
        a_hb_err = tables.Float64Col(shape=(len(result_list)))
        a_ha_err = tables.Float64Col(shape=(len(result_list)))
        b_hb_err = tables.Float64Col(shape=(len(result_list)))
        b_ha_err = tables.Float64Col(shape=(len(result_list)))
        flux1_err = tables.Float64Col(shape=(len(result_list)))
        flux2_err = tables.Float64Col(shape=(len(result_list)))
        flux3_err = tables.Float64Col(shape=(len(result_list)))
        vel_err = tables.Float64Col(shape=(len(result_list)))
        sig_err = tables.Float64Col(shape=(len(result_list)))
        
        residuals = tables.Float64Col(shape=(len(result_list)))
        
    hf = tables.open_file(results_dir+'/results.h5', 'w')

    th = hf.create_table('/','results', t_res)

    hr = th.row

    hr['x'] = xm
    hr['y'] = ym

    hr['a_hb'] = am_hb
    hr['a_ha'] = am_ha
    hr['b_hb'] = bm_hb
    hr['b_ha'] = bm_ha
    hr['flux1'] = fluxm1
    hr['flux2'] = fluxm2
    hr['flux3'] = fluxm3
    hr['vel'] = velm
    hr['sig'] = sigm

    hr['a_hb_err'] = am_hb_err
    hr['a_ha_err'] = am_ha_err
    hr['b_hb_err'] = bm_hb_err
    hr['b_ha_err'] = bm_ha_err
    hr['flux1_err'] = fluxm1_err
    hr['flux2_err'] = fluxm2_err
    hr['flux3_err'] = fluxm3_err
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
    ft.create_dataset('lam', data=lam_r)
    ft.close()

    return

def hb_ha_n2_gaussians_2g_cons(naxis1,naxis2,result_list,results_dir,lam_r,ncomps_flag,bic_flag):

    results = np.reshape(result_list, (naxis2,naxis1))

    am_hb = np.zeros([naxis2,naxis1])
    am_ha = np.zeros([naxis2,naxis1])
    bm_hb = np.zeros([naxis2,naxis1])
    bm_ha = np.zeros([naxis2,naxis1])
    fluxm1 = np.zeros([naxis2,naxis1])
    fluxm2 = np.zeros([naxis2,naxis1])
    fluxm3 = np.zeros([naxis2,naxis1])
    fluxm1_b = np.zeros([naxis2,naxis1])
    fluxm2_b = np.zeros([naxis2,naxis1])
    fluxm3_b = np.zeros([naxis2,naxis1])
    velm = np.zeros([naxis2,naxis1])
    sigm = np.zeros([naxis2,naxis1])
    velm_b = np.zeros([naxis2,naxis1])
    sigm_b = np.zeros([naxis2,naxis1])

    am_hb_err = np.zeros([naxis2,naxis1])
    am_ha_err = np.zeros([naxis2,naxis1])
    bm_hb_err = np.zeros([naxis2,naxis1])
    bm_ha_err = np.zeros([naxis2,naxis1])
    fluxm1_err = np.zeros([naxis2,naxis1])
    fluxm2_err = np.zeros([naxis2,naxis1])
    fluxm3_err = np.zeros([naxis2,naxis1])
    fluxm1_b_err = np.zeros([naxis2,naxis1])
    fluxm2_b_err = np.zeros([naxis2,naxis1])
    fluxm3_b_err = np.zeros([naxis2,naxis1])
    velm_err = np.zeros([naxis2,naxis1])
    sigm_err = np.zeros([naxis2,naxis1])
    velm_b_err = np.zeros([naxis2,naxis1])
    sigm_b_err = np.zeros([naxis2,naxis1])
    
    residualsm = np.zeros([naxis2,naxis1])
    redchi = np.zeros([naxis2,naxis1])
    redchi_w = np.zeros([naxis2,naxis1])
    bic = np.zeros([naxis2,naxis1])
    
    residual = np.zeros([len(results[0,0].residual),naxis2,naxis1])
    model = np.zeros([len(results[0,0].residual),naxis2,naxis1])
    model_2g_n = np.zeros([len(results[0,0].residual),naxis2,naxis1])
    model_2g_b = np.zeros([len(results[0,0].residual),naxis2,naxis1])
    data = np.zeros([len(results[0,0].residual),naxis2,naxis1])

    xm = np.zeros([naxis2,naxis1])
    ym = np.zeros([naxis2,naxis1])

    for i in np.arange(naxis1):
        for j in np.arange(naxis2):
            am_hb[j,i] = results[j,i].params['a_hb'].value
            am_ha[j,i] = results[j,i].params['a_ha'].value
            bm_hb[j,i] = results[j,i].params['b_hb'].value
            bm_ha[j,i] = results[j,i].params['b_ha'].value
            fluxm1[j,i] = results[j,i].params['flux'].value/results[j,i].params['ratio_hb'].value
            fluxm2[j,i] = results[j,i].params['flux'].value
            fluxm3[j,i] = results[j,i].params['ratio_n2'].value*results[j,i].params['flux'].value
            velm[j,i] = results[j,i].params['vel'].value
            sigm[j,i] = results[j,i].params['sig'].value
            
            am_hb_err[j,i] = results[j,i].params['a_hb'].stderr
            am_ha_err[j,i] = results[j,i].params['a_ha'].stderr
            bm_hb_err[j,i] = results[j,i].params['b_hb'].stderr
            bm_ha_err[j,i] = results[j,i].params['b_ha'].stderr
            fluxm1_err[j,i] = results[j,i].params['ratio_hb'].stderr
            fluxm2_err[j,i] = results[j,i].params['flux'].stderr
            fluxm3_err[j,i] = results[j,i].params['ratio_n2'].stderr
            velm_err[j,i] = results[j,i].params['vel'].stderr
            sigm_err[j,i] = results[j,i].params['sig'].stderr
            
            x_hb = lam_r[0:int(len(lam_r)/2)]
            x_ha = lam_r[int(len(lam_r)/2):]
            
            if (ncomps_flag[j,i] == 1) and (bic_flag[j,i] == 2):         
                fluxm1_b[j,i] = results[j,i].params['flux_b'].value/results[j,i].params['ratio_hb_b'].value
                fluxm2_b[j,i] = results[j,i].params['flux_b'].value
                fluxm3_b[j,i] = results[j,i].params['ratio_n2_b'].value*results[j,i].params['flux_b'].value
                velm_b[j,i] = results[j,i].params['vel_b'].value
                sigm_b[j,i] = results[j,i].params['sig_b'].value
                
                fluxm1_b_err[j,i] = results[j,i].params['ratio_hb_b'].stderr
                fluxm2_b_err[j,i] = results[j,i].params['flux_b'].stderr
                fluxm3_b_err[j,i] = results[j,i].params['ratio_n2_b'].stderr
                velm_b_err[j,i] = results[j,i].params['vel_b'].stderr
                sigm_b_err[j,i] = results[j,i].params['sig_b'].stderr
                
                model_2g_n[:,j,i] = np.concatenate(( ((am_hb[j,i] + (bm_hb[j,i]*x_hb)) + \
                
                (fluxm1[j,i] / (np.sqrt(2*np.pi) * (np.sqrt((sigm[j,i]**2)+(inst_sigma_blue**2))*results[j,i].params['lam0_hb'].value/c))) * np.exp(-(x_hb-((velm[j,i]*results[j,i].params['lam0_hb'].value/c)+results[j,i].params['lam0_hb'].value))**2 / (2*(np.sqrt((sigm[j,i]**2)+(inst_sigma_blue**2))*results[j,i].params['lam0_hb'].value/c)**2))), \
                
                ((am_ha[j,i] + (bm_ha[j,i]*x_ha)) + \
                
                ((fluxm3[j,i]/2.94) / (np.sqrt(2*np.pi) * (np.sqrt((sigm[j,i]**2)+(inst_sigma_red**2))*results[j,i].params['lam01'].value/c))) * np.exp(-(x_ha-((velm[j,i]*results[j,i].params['lam01'].value/c)+results[j,i].params['lam01'].value))**2 / (2*(np.sqrt((sigm[j,i]**2) +(inst_sigma_red**2))*results[j,i].params['lam01'].value/c)**2)) + \
            
                (fluxm2[j,i] / (np.sqrt(2*np.pi) * (np.sqrt((sigm[j,i]**2)+(inst_sigma_red**2))*results[j,i].params['lam02'].value/c))) * np.exp(-(x_ha-((velm[j,i]*results[j,i].params['lam02'].value/c)+results[j,i].params['lam02'].value))**2 / (2*(np.sqrt((sigm[j,i]**2)+(inst_sigma_red**2))*results[j,i].params['lam02'].value/c)**2)) + \
                
                (fluxm3[j,i] / (np.sqrt(2*np.pi) * (np.sqrt((sigm[j,i]**2)+(inst_sigma_red**2))*results[j,i].params['lam03'].value/c))) * np.exp(-(x_ha-((velm[j,i]*results[j,i].params['lam03'].value/c)+results[j,i].params['lam03'].value))**2 / (2*(np.sqrt((sigm[j,i]**2)+(inst_sigma_red**2))*results[j,i].params['lam03'].value/c)**2))) ))
                
                model_2g_b[:,j,i] = np.concatenate(( ((am_hb[j,i] + (bm_hb[j,i]*x_hb)) + \
                
                (fluxm1_b[j,i] / (np.sqrt(2*np.pi) * (np.sqrt((sigm_b[j,i]**2)+(inst_sigma_blue**2))*results[j,i].params['lam0_hb'].value/c))) * np.exp(-(x_hb-((velm_b[j,i]*results[j,i].params['lam0_hb'].value/c)+results[j,i].params['lam0_hb'].value))**2 / (2*(np.sqrt((sigm_b[j,i]**2)+(inst_sigma_blue**2))*results[j,i].params['lam0_hb'].value/c)**2))), \
                
                ((am_ha[j,i] + (bm_ha[j,i]*x_ha)) + \
                
                ((fluxm3_b[j,i]/2.94) / (np.sqrt(2*np.pi) * (np.sqrt((sigm_b[j,i]**2)+(inst_sigma_red**2))*results[j,i].params['lam01'].value/c))) * np.exp(-(x_ha-((velm_b[j,i]*results[j,i].params['lam01'].value/c)+results[j,i].params['lam01'].value))**2 / (2*(np.sqrt((sigm_b[j,i]**2) +(inst_sigma_red**2))*results[j,i].params['lam01'].value/c)**2)) + \
            
                (fluxm2_b[j,i] / (np.sqrt(2*np.pi) * (np.sqrt((sigm_b[j,i]**2)+(inst_sigma_red**2))*results[j,i].params['lam02'].value/c))) * np.exp(-(x_ha-((velm_b[j,i]*results[j,i].params['lam02'].value/c)+results[j,i].params['lam02'].value))**2 / (2*(np.sqrt((sigm_b[j,i]**2)+(inst_sigma_red**2))*results[j,i].params['lam02'].value/c)**2)) + \
                
                (fluxm3_b[j,i] / (np.sqrt(2*np.pi) * (np.sqrt((sigm_b[j,i]**2)+(inst_sigma_red**2))*results[j,i].params['lam03'].value/c))) * np.exp(-(x_ha-((velm_b[j,i]*results[j,i].params['lam03'].value/c)+results[j,i].params['lam03'].value))**2 / (2*(np.sqrt((sigm_b[j,i]**2)+(inst_sigma_red**2))*results[j,i].params['lam03'].value/c)**2))) ))
            
            residualsm[j,i] = np.abs(np.sum(results[j,i].residual))
            redchi[j,i] = results[j,i].redchi
            redchi_w[j,i] = results[j,i].redchi*np.abs(residualsm[j,i])
            bic[j,i] = results[j,i].bic
                        
            residual[:,j,i] = results[j,i].residual
            data[:,j,i] = results[j,i].data
            model[:,j,i] = results[j,i].best_fit
            
            xm[j,i] = i
            ym[j,i] = j

    fits.writeto(results_dir+'/fluxm1.fits',fluxm1,overwrite=True)
    fits.writeto(results_dir+'/fluxm2.fits',fluxm2,overwrite=True)
    fits.writeto(results_dir+'/fluxm3.fits',fluxm3,overwrite=True)
    fits.writeto(results_dir+'/fluxm1_b.fits',fluxm1_b,overwrite=True)
    fits.writeto(results_dir+'/fluxm2_b.fits',fluxm2_b,overwrite=True)
    fits.writeto(results_dir+'/fluxm3_b.fits',fluxm3_b,overwrite=True)
    fits.writeto(results_dir+'/velm.fits',velm,overwrite=True)
    fits.writeto(results_dir+'/sigm.fits',sigm,overwrite=True)
    fits.writeto(results_dir+'/velm_b.fits',velm_b,overwrite=True)
    fits.writeto(results_dir+'/sigm_b.fits',sigm_b,overwrite=True)

    fits.writeto(results_dir+'/flux1_err.fits',fluxm1_err,overwrite=True)
    fits.writeto(results_dir+'/flux2_err.fits',fluxm2_err,overwrite=True)
    fits.writeto(results_dir+'/flux3_err.fits',fluxm3_err,overwrite=True)
    fits.writeto(results_dir+'/flux1_b_err.fits',fluxm1_b_err,overwrite=True)
    fits.writeto(results_dir+'/flux2_b_err.fits',fluxm2_b_err,overwrite=True)
    fits.writeto(results_dir+'/flux3_b_err.fits',fluxm3_b_err,overwrite=True)
    fits.writeto(results_dir+'/vel_err.fits',velm_err,overwrite=True)
    fits.writeto(results_dir+'/sig_err.fits',sigm_err,overwrite=True)
    fits.writeto(results_dir+'/vel_b_err.fits',velm_b_err,overwrite=True)
    fits.writeto(results_dir+'/sig_b_err.fits',sigm_b_err,overwrite=True)
    
    fits.writeto(results_dir+'/residuals.fits',residualsm,overwrite=True)
    fits.writeto(results_dir+'/redchi.fits',redchi,overwrite=True)
    fits.writeto(results_dir+'/redchi_w.fits',redchi_w,overwrite=True)
    fits.writeto(results_dir+'/bic.fits',bic,overwrite=True)
    
    fits.writeto(results_dir+'/ncomps_flag.fits',ncomps_flag,overwrite=True)
    fits.writeto(results_dir+'/bic_flag.fits',bic_flag,overwrite=True)

    xm = np.ravel(xm)
    ym = np.ravel(ym)

    am_hb = np.ravel(am_hb)
    am_ha = np.ravel(am_ha)
    bm_hb = np.ravel(bm_hb)
    bm_ha = np.ravel(bm_ha)
    fluxm1 = np.ravel(fluxm1)
    fluxm2 = np.ravel(fluxm2)
    fluxm3 = np.ravel(fluxm3)
    fluxm1_b = np.ravel(fluxm1_b)
    fluxm2_b = np.ravel(fluxm2_b)
    fluxm3_b = np.ravel(fluxm3_b)
    velm = np.ravel(velm)
    sigm = np.ravel(sigm)
    velm_b = np.ravel(velm_b)
    sigm_b = np.ravel(sigm_b)

    am_hb_err = np.ravel(am_hb_err)
    am_ha_err = np.ravel(am_ha_err)
    bm_hb_err = np.ravel(bm_hb_err)
    bm_ha_err = np.ravel(bm_ha_err)
    fluxm1_err = np.ravel(fluxm1_err)
    fluxm2_err = np.ravel(fluxm2_err)
    fluxm3_err = np.ravel(fluxm3_err)
    fluxm1_b_err = np.ravel(fluxm1_b_err)
    fluxm2_b_err = np.ravel(fluxm2_b_err)
    fluxm3_b_err = np.ravel(fluxm3_b_err)
    velm_err = np.ravel(velm_err)
    sigm_err = np.ravel(sigm_err)
    velm_b_err = np.ravel(velm_b_err)
    sigm_b_err = np.ravel(sigm_b_err)
    
    residualsm = np.ravel(residualsm)

    class t_res(tables.IsDescription):
        
        x = tables.IntCol(shape=(len(result_list)))
        y = tables.IntCol(shape=(len(result_list)))
        
        a_hb = tables.Float64Col(shape=(len(result_list)))
        a_ha = tables.Float64Col(shape=(len(result_list)))
        b_hb = tables.Float64Col(shape=(len(result_list)))
        b_ha = tables.Float64Col(shape=(len(result_list)))
        flux1 = tables.Float64Col(shape=(len(result_list)))
        flux2 = tables.Float64Col(shape=(len(result_list)))
        flux3 = tables.Float64Col(shape=(len(result_list)))
        flux1_b = tables.Float64Col(shape=(len(result_list)))
        flux2_b = tables.Float64Col(shape=(len(result_list)))
        flux3_b = tables.Float64Col(shape=(len(result_list)))
        vel = tables.Float64Col(shape=(len(result_list)))
        sig = tables.Float64Col(shape=(len(result_list)))
        vel_b = tables.Float64Col(shape=(len(result_list)))
        sig_b = tables.Float64Col(shape=(len(result_list)))
        
        a_hb_err = tables.Float64Col(shape=(len(result_list)))
        a_ha_err = tables.Float64Col(shape=(len(result_list)))
        b_hb_err = tables.Float64Col(shape=(len(result_list)))
        b_ha_err = tables.Float64Col(shape=(len(result_list)))
        flux1_err = tables.Float64Col(shape=(len(result_list)))
        flux2_err = tables.Float64Col(shape=(len(result_list)))
        flux3_err = tables.Float64Col(shape=(len(result_list)))
        flux1_b_err = tables.Float64Col(shape=(len(result_list)))
        flux2_b_err = tables.Float64Col(shape=(len(result_list)))
        flux3_b_err = tables.Float64Col(shape=(len(result_list)))
        vel_err = tables.Float64Col(shape=(len(result_list)))
        sig_err = tables.Float64Col(shape=(len(result_list)))
        vel_b_err = tables.Float64Col(shape=(len(result_list)))
        sig_b_err = tables.Float64Col(shape=(len(result_list)))
        
        residuals = tables.Float64Col(shape=(len(result_list)))
        
    hf = tables.open_file(results_dir+'/results.h5', 'w')

    th = hf.create_table('/','results', t_res)

    hr = th.row

    hr['x'] = xm
    hr['y'] = ym

    hr['a_hb'] = am_hb
    hr['a_ha'] = am_ha
    hr['b_hb'] = bm_hb
    hr['b_ha'] = bm_ha
    hr['flux1'] = fluxm1
    hr['flux2'] = fluxm2
    hr['flux3'] = fluxm3
    hr['flux1_b'] = fluxm1_b
    hr['flux2_b'] = fluxm2_b
    hr['flux3_b'] = fluxm3_b
    hr['vel'] = velm
    hr['sig'] = sigm
    hr['vel_b'] = velm_b
    hr['sig_b'] = sigm_b

    hr['a_hb_err'] = am_hb_err
    hr['a_ha_err'] = am_ha_err
    hr['b_hb_err'] = bm_hb_err
    hr['b_ha_err'] = bm_ha_err
    hr['flux1_err'] = fluxm1_err
    hr['flux2_err'] = fluxm2_err
    hr['flux3_err'] = fluxm3_err
    hr['flux1_b_err'] = fluxm1_b_err
    hr['flux2_b_err'] = fluxm2_b_err
    hr['flux3_b_err'] = fluxm3_b_err
    hr['vel_err'] = velm_err
    hr['sig_err'] = sigm_err
    hr['vel_b_err'] = velm_b_err
    hr['sig_b_err'] = sigm_b_err
    
    hr['residuals'] = residualsm

    hr.append()

    hf.flush()
    hf.close()
    
    ft = h5py.File(results_dir+'/fit.hdf5', 'w')
    ft.create_dataset('residual', data=residual)
    ft.create_dataset('model', data=model)
    ft.create_dataset('model_2g_n', data=model_2g_n)
    ft.create_dataset('model_2g_b', data=model_2g_b)
    ft.create_dataset('data', data=data)
    ft.create_dataset('lam', data=lam_r)
    ft.close()

    return