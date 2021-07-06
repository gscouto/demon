##### SYSTEM DEFINITIONS

n_proc = 48                 # number of processors to use in the run

cube_path = '/home/guilherme/oso/vales_sample2/starlight/'         # directory path to where the cubes to be fitted are located
results_path = '/home/guilherme/oso/vales_sample2/EL_measurements/'                          # directory path to where the results of the fit will be saved 

##### FITTING DEFINITIONS

lines_to_fit ='all_cons'   # which (set of) lines will be measured?
#options: 'ha_n2', 'ha_n2_cons', 'hb_ha_n2_cons', 'hb', 'o3', 'o3_cons', 's2', 's2_cons', 'o1', 'o1_cons', 'all' ['ha_n2'+'hb'+'o3'+'s2'+'o1'], 'all_cons' ['hb_ha_n2_cons'+'o3_cons'+'s2_cons'+'o1_cons']

morec_flag = 'no'          # should the code fit a second component if the residuals are considerable (1 gaussian fit does not represent the line profile)? / 'yes' or 'no'

refit_flag = 'no'           # should the code refit the worst fits? this may take much longer to finish the entire run. / 'yes' or 'no'

SL_flag = 'yes'              # is the datacube file a product of the STARLIGHT code (after stellar continuum fitting)?

aut_ini = 'no'              # initial parameters set by the code based on a preliminary fit in a integrated spectrum / 'yes' or 'no' [recommended to leave as 'yes' initially, and then 'no' plus tweaking the fitting parameters and limits below afterwards] / OBS: if there is a bright star in the FoV, leaving this as 'yes' may crash the fit, due to the estimated integrated spectrum being dominated by the star. in this case it is recommended to leave as 'no' and with flexible limits below.

hahb_flag = 'no'            # should the second component fit constrain the Ha/Hb ratio as the same as the first component? / 'yes' or 'no' [NOT IMPLEMENTED YET]

ids = ['HATLASJ083601']     # names of the galaxies to be fitted (they must match the name of the datacube file, ex: 'NGC1097'+.fits)

##### SPECTRA DEFINITIONS

inst_sigma_blue = 76.5        # instrumental sigma in the blue part, in km/s, to be corrected from the measured sigma
inst_sigma_red = 48.5         # instrumental sigma in the red part, in km/s, to be corrected from the measured sigma

lami_SN = 6500.0            # lower wavelenght to calculate signal to noise ratio
lamf_SN = 6630.0            # upper wavelenght to calculate signal to noise ratio

lami_cont1 = 6400.0         # lower wavelenght to first window of the local continuum / this is used only to calculate the S/N ratio
lamf_cont1 = 6450.0         # upper wavelenght to first window of the local continuum / this is used only to calculate the S/N ratio

lami_cont2 = 6650.0         # upper wavelenght to second window of the local continuum / this is used only to calculate the S/N ratio
lamf_cont2 = 6700.0         # upper wavelenght to second window of the local continuum / this is used only to calculate the S/N ratio

##### FITTING INITIAL PARAMETERS AND LIMITS

init_params = {

'a' : 15.,                  # first parameter of the local continuum fitted in the emission-line (a+b*x, x = wavelength vector)
'b' : 0.,                   # second parameter of the local continuum fitted in the emission-line (a+b*x, x = wavelength vector)

'flux' : 500.,              # initial guess (IG) for the integrated flux of the (main) emission line
'flux_min' : 0.0,           # minimum value

'vel' : 0.,
'vel_min' : -400.,
'vel_max' : 400.,

'sig' : 60.,
'sig_min' : 10.0,
'sig_max' : 120.,

'n2_ha_ratio' : 0.5,
'n2_ha_ratio_min' : 0.01,
'n2_ha_ratio_max' : 4.0,

'ha_hb_ratio' : 3.1,
'ha_hb_ratio_min' : 2.5,
'ha_hb_ratio_max' : 20.0,

's2_ratio' : 1.,
's2_ratio_min' : 0.2,
's2_ratio_max' : 1.6,

'flux_delta' : 5.,          # [USED IN MORE COMPS] relation between narrow and broad flux components: flux_delta = flux_narrow/flux_broad 
'flux_delta_min' : 1.,
'flux_delta_max' : 20.,

'sig_delta' : 1.5,          # [USED IN MORE COMPS] relation between narrow and broad sigma components: sig_delta = sig_broad/sig_narrow 
'sig_delta_min' : 1.2,
'sig_delta_max' : 3.,

'vel_delta' : 0.,           # [USED IN MORE COMPS] relation between narrow and broad velocity components: vel_delta = vel_narrow - vel_broad 
'vel_delta_min' : -200.,
'vel_delta_max' : 200.

}

# REFIT PARAMETERS

redchi_limit = 500      # reduced chisquate threshold: above this value, a refit will be done
refit_radius = 10       # spaxel radius to extract kick parameters for the refit

refit_vel_min = -100.   # lower velocity limit for the refit
refit_vel_max = 100.   # upper velocity limit for the refit
