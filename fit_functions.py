import numpy as np
from astropy.constants import c
from demon_config import *

c = c.value/1000.

def one_gaussian(x, a, b, flux, vel, sig, lam0):
    
    return (a + (b*x)) + \
        (flux / (np.sqrt(2*np.pi) * (np.sqrt((sig**2)+(inst_sigma_blue**2))*lam0/c))) * np.exp(-(x-((vel*lam0/c)+lam0))**2 / (2*(np.sqrt((sig**2)+(inst_sigma_blue**2))*lam0/c)**2))
    
def two_gaussians(x, a, b, flux1, flux2, vel1, vel2, sig1, sig2, lam01, lam02):
    
    return (a + (b*x)) + \
        (flux1 / (np.sqrt(2*np.pi) * (np.sqrt((sig1**2)+(inst_sigma_red**2))*lam01/c))) * np.exp(-(x-((vel1*lam01/c)+lam01))**2 / (2*(np.sqrt((sig1**2)+(inst_sigma_red**2))*lam01/c)**2)) + \
        (flux2 / (np.sqrt(2*np.pi) * (np.sqrt((sig2**2)+(inst_sigma_red**2))*lam02/c))) * np.exp(-(x-((vel2*lam02/c)+lam02))**2 / (2*(np.sqrt((sig2**2)+(inst_sigma_red**2))*lam02/c)**2))
    
def two_gaussians_cons_o3(x, a, b, flux, vel, sig, lam01, lam02):
    
    return (a + (b*x)) + \
        ((flux/2.94) / (np.sqrt(2*np.pi) * (np.sqrt((sig**2)+(inst_sigma_blue**2))*lam01/c))) * np.exp(-(x-((vel*lam01/c)+lam01))**2 / (2*(np.sqrt((sig**2)+(inst_sigma_blue**2))*lam01/c)**2)) + \
        (flux / (np.sqrt(2*np.pi) * (np.sqrt((sig**2)+(inst_sigma_blue**2))*lam02/c))) * np.exp(-(x-((vel*lam02/c)+lam02))**2 / (2*(np.sqrt((sig**2)+(inst_sigma_blue**2))*lam02/c)**2))
    
def two_gaussians_2g_cons_o3(x, a, b, flux, flux_b, vel, vel_b, sig, sig_b, lam01, lam02):
    
    return (a + (b*x)) + \
        ((flux/2.94) / (np.sqrt(2*np.pi) * (np.sqrt((sig**2)+(inst_sigma_blue**2))*lam01/c))) * np.exp(-(x-((vel*lam01/c)+lam01))**2 / (2*(np.sqrt((sig**2)+(inst_sigma_blue**2))*lam01/c)**2)) + \
        (flux / (np.sqrt(2*np.pi) * (np.sqrt((sig**2)+(inst_sigma_blue**2))*lam02/c))) * np.exp(-(x-((vel*lam02/c)+lam02))**2 / (2*(np.sqrt((sig**2)+(inst_sigma_blue**2))*lam02/c)**2)) + \
        ((flux_b/2.94) / (np.sqrt(2*np.pi) * (np.sqrt((sig_b**2)+(inst_sigma_blue**2))*lam01/c))) * np.exp(-(x-((vel_b*lam01/c)+lam01))**2 / (2*(np.sqrt((sig_b**2)+(inst_sigma_blue**2))*lam01/c)**2)) + \
        (flux_b / (np.sqrt(2*np.pi) * (np.sqrt((sig_b**2)+(inst_sigma_blue**2))*lam02/c))) * np.exp(-(x-((vel_b*lam02/c)+lam02))**2 / (2*(np.sqrt((sig_b**2)+(inst_sigma_blue**2))*lam02/c)**2))

def two_gaussians_cons_o1(x, a, b, flux, vel, sig, lam01, lam02):
    
    return (a + (b*x)) + \
        (flux / (np.sqrt(2*np.pi) * (np.sqrt((sig**2)+(inst_sigma_red**2))*lam01/c))) * np.exp(-(x-((vel*lam01/c)+lam01))**2 / (2*(np.sqrt((sig**2)+(inst_sigma_red**2))*lam01/c)**2)) + \
        ((flux/3.05) / (np.sqrt(2*np.pi) * (np.sqrt((sig**2)+(inst_sigma_red**2))*lam02/c))) * np.exp(-(x-((vel*lam02/c)+lam02))**2 / (2*(np.sqrt((sig**2)+(inst_sigma_red**2))*lam02/c)**2))
    
def two_gaussians_2g_cons_o1(x, a, b, flux, flux_b, vel, vel_b, sig, sig_b, lam01, lam02):
    
    return (a + (b*x)) + \
        (flux / (np.sqrt(2*np.pi) * (np.sqrt((sig**2)+(inst_sigma_red**2))*lam01/c))) * np.exp(-(x-((vel*lam01/c)+lam01))**2 / (2*(np.sqrt((sig**2)+(inst_sigma_red**2))*lam01/c)**2)) + \
        ((flux/3.05) / (np.sqrt(2*np.pi) * (np.sqrt((sig**2)+(inst_sigma_red**2))*lam02/c))) * np.exp(-(x-((vel*lam02/c)+lam02))**2 / (2*(np.sqrt((sig**2)+(inst_sigma_red**2))*lam02/c)**2)) + \
        (flux_b / (np.sqrt(2*np.pi) * (np.sqrt((sig_b**2)+(inst_sigma_red**2))*lam01/c))) * np.exp(-(x-((vel_b*lam01/c)+lam01))**2 / (2*(np.sqrt((sig_b**2)+(inst_sigma_red**2))*lam01/c)**2)) + \
        ((flux_b/3.05) / (np.sqrt(2*np.pi) * (np.sqrt((sig_b**2)+(inst_sigma_red**2))*lam02/c))) * np.exp(-(x-((vel_b*lam02/c)+lam02))**2 / (2*(np.sqrt((sig_b**2)+(inst_sigma_red**2))*lam02/c)**2))
    
def two_gaussians_cons_s2(x, a, b, flux, ratio, vel, sig, lam01, lam02):
    
    return (a + (b*x)) + \
        (flux / (np.sqrt(2*np.pi) * (np.sqrt((sig**2)+(inst_sigma_red**2))*lam01/c))) * np.exp(-(x-((vel*lam01/c)+lam01))**2 / (2*(np.sqrt((sig**2)+(inst_sigma_red**2))*lam01/c)**2)) + \
        ((flux/ratio) / (np.sqrt(2*np.pi) * (np.sqrt((sig**2)+(inst_sigma_red**2))*lam02/c))) * np.exp(-(x-((vel*lam02/c)+lam02))**2 / (2*(np.sqrt((sig**2)+(inst_sigma_red**2))*lam02/c)**2))

def two_gaussians_2g_cons_s2(x, a, b, flux, flux_b, ratio, ratio_b, vel, vel_b, sig, sig_b, lam01, lam02):
    
    return (a + (b*x)) + \
        (flux / (np.sqrt(2*np.pi) * (np.sqrt((sig**2)+(inst_sigma_red**2))*lam01/c))) * np.exp(-(x-((vel*lam01/c)+lam01))**2 / (2*(np.sqrt((sig**2)+(inst_sigma_red**2))*lam01/c)**2)) + \
        ((flux/ratio) / (np.sqrt(2*np.pi) * (np.sqrt((sig**2)+(inst_sigma_red**2))*lam02/c))) * np.exp(-(x-((vel*lam02/c)+lam02))**2 / (2*(np.sqrt((sig**2)+(inst_sigma_red**2))*lam02/c)**2)) + \
        (flux_b / (np.sqrt(2*np.pi) * (np.sqrt((sig_b**2)+(inst_sigma_red**2))*lam01/c))) * np.exp(-(x-((vel_b*lam01/c)+lam01))**2 / (2*(np.sqrt((sig_b**2)+(inst_sigma_red**2))*lam01/c)**2)) + \
        ((flux_b/ratio_b) / (np.sqrt(2*np.pi) * (np.sqrt((sig_b**2)+(inst_sigma_red**2))*lam02/c))) * np.exp(-(x-((vel_b*lam02/c)+lam02))**2 / (2*(np.sqrt((sig_b**2)+(inst_sigma_red**2))*lam02/c)**2))

def three_gaussians(x, a, b, flux1, flux2, flux3, vel1, vel2, vel3, sig1, sig2, sig3, lam01, lam02, lam03):
    
    return (a + (b*x)) + \
        (flux1 / (np.sqrt(2*np.pi) * (np.sqrt((sig1**2)+(inst_sigma_red**2))*lam01/c))) * np.exp(-(x-((vel1*lam01/c)+lam01))**2 / (2*(np.sqrt((sig1**2)+(inst_sigma_red**2))*lam01/c)**2)) + \
        (flux2 / (np.sqrt(2*np.pi) * (np.sqrt((sig2**2)+(inst_sigma_red**2))*lam02/c))) * np.exp(-(x-((vel2*lam02/c)+lam02))**2 / (2*(np.sqrt((sig2**2)+(inst_sigma_red**2))*lam02/c)**2)) + \
        (flux3 / (np.sqrt(2*np.pi) * (sig3*lam03/c))) * np.exp(-(x-((vel3*lam03/c)+lam03))**2 / (2*(sig3*lam03/c)**2))
    
def three_gaussians_cons(x, a, b, flux, ratio, vel, sig, lam01, lam02, lam03):
    
    return (a + (b*x)) + \
        ((ratio*flux/2.94) / (np.sqrt(2*np.pi) * (np.sqrt((sig**2)+(inst_sigma_red**2))*lam01/c))) * np.exp(-(x-((vel*lam01/c)+lam01))**2 / (2*(np.sqrt((sig**2)+(inst_sigma_red**2))*lam01/c)**2)) + \
        (flux / (np.sqrt(2*np.pi) * (np.sqrt((sig**2)+(inst_sigma_red**2))*lam02/c))) * np.exp(-(x-((vel*lam02/c)+lam02))**2 / (2*(np.sqrt((sig**2)+(inst_sigma_red**2))*lam02/c)**2)) + \
        ((ratio*flux) / (np.sqrt(2*np.pi) * (np.sqrt((sig**2)+(inst_sigma_red**2))*lam03/c))) * np.exp(-(x-((vel*lam03/c)+lam03))**2 / (2*(np.sqrt((sig**2)+(inst_sigma_red**2))*lam03/c)**2))
    
def three_gaussians_2g_cons(x, a, b, flux, flux_b, ratio, ratio_b, vel, vel_b, sig, sig_b, lam01, lam02, lam03):
    
    return (a + (b*x)) + \
        ((ratio*flux/2.94) / (np.sqrt(2*np.pi) * (np.sqrt((sig**2)+(inst_sigma_red**2))*lam01/c))) * np.exp(-(x-((vel*lam01/c)+lam01))**2 / (2*(np.sqrt((sig**2)+(inst_sigma_red**2))*lam01/c)**2)) + \
        (flux / (np.sqrt(2*np.pi) * (np.sqrt((sig**2)+(inst_sigma_red**2))*lam02/c))) * np.exp(-(x-((vel*lam02/c)+lam02))**2 / (2*(np.sqrt((sig**2)+(inst_sigma_red**2))*lam02/c)**2)) + \
        ((ratio*flux) / (np.sqrt(2*np.pi) * (np.sqrt((sig**2)+(inst_sigma_red**2))*lam03/c))) * np.exp(-(x-((vel*lam03/c)+lam03))**2 / (2*(np.sqrt((sig**2)+(inst_sigma_red**2))*lam03/c)**2)) + \
        ((ratio_b*flux_b/2.94) / (np.sqrt(2*np.pi) * (np.sqrt((sig_b**2)+(inst_sigma_red**2))*lam01/c))) * np.exp(-(x-((vel_b*lam01/c)+lam01))**2 / (2*(np.sqrt((sig_b**2)+(inst_sigma_red**2))*lam01/c)**2)) + \
        (flux_b / (np.sqrt(2*np.pi) * (np.sqrt((sig_b**2)+(inst_sigma_red**2))*lam02/c))) * np.exp(-(x-((vel_b*lam02/c)+lam02))**2 / (2*(np.sqrt((sig_b**2)+(inst_sigma_red**2))*lam02/c)**2)) + \
        ((ratio_b*flux_b) / (np.sqrt(2*np.pi) * (np.sqrt((sig_b**2)+(inst_sigma_red**2))*lam03/c))) * np.exp(-(x-((vel_b*lam03/c)+lam03))**2 / (2*(np.sqrt((sig_b**2)+(inst_sigma_red**2))*lam03/c)**2))
                
def hb_ha_n2_gaussians_cons(x, a_hb, a_ha, b_hb, b_ha, flux, ratio_hb, ratio_n2, vel, sig, lam0_hb, lam01, lam02, lam03):
    
    x_hb = x[0:int(len(x)/2)]
    x_ha = x[int(len(x)/2):]
    
    return np.concatenate(( ((a_hb + (b_hb*x_hb)) + \
        ((flux/ratio_hb) / (np.sqrt(2*np.pi) * (np.sqrt((sig**2)+(inst_sigma_blue**2))*lam0_hb/c))) * np.exp(-(x_hb-((vel*lam0_hb/c)+lam0_hb))**2 / (2*(np.sqrt((sig**2)+(inst_sigma_blue**2))*lam0_hb/c)**2))), \
        ((a_ha + (b_ha*x_ha)) + \
        ((ratio_n2*flux/2.94) / (np.sqrt(2*np.pi) * (np.sqrt((sig**2)+(inst_sigma_red**2))*lam01/c))) * np.exp(-(x_ha-((vel*lam01/c)+lam01))**2 / (2*(np.sqrt((sig**2)+(inst_sigma_red**2))*lam01/c)**2)) + \
        (flux / (np.sqrt(2*np.pi) * (np.sqrt((sig**2)+(inst_sigma_red**2))*lam02/c))) * np.exp(-(x_ha-((vel*lam02/c)+lam02))**2 / (2*(np.sqrt((sig**2)+(inst_sigma_red**2))*lam02/c)**2)) + \
        ((ratio_n2*flux) / (np.sqrt(2*np.pi) * (np.sqrt((sig**2)+(inst_sigma_red**2))*lam03/c))) * np.exp(-(x_ha-((vel*lam03/c)+lam03))**2 / (2*(np.sqrt((sig**2)+(inst_sigma_red**2))*lam03/c)**2))) ))

def hb_ha_n2_gaussians_2g_cons(x, a_hb, a_ha, b_hb, b_ha, flux, flux_b, ratio_hb, ratio_n2, ratio_hb_b, ratio_n2_b, vel, vel_b, sig, sig_b, lam0_hb, lam01, lam02, lam03):
    
    x_hb = x[0:int(len(x)/2)]
    x_ha = x[int(len(x)/2):]
    
    return np.concatenate(( ((a_hb + (b_hb*x_hb)) + \
        ((flux/ratio_hb) / (np.sqrt(2*np.pi) * (np.sqrt((sig**2)+(inst_sigma_blue**2))*lam0_hb/c))) * np.exp(-(x_hb-((vel*lam0_hb/c)+lam0_hb))**2 / (2*(np.sqrt((sig**2)+(inst_sigma_blue**2))*lam0_hb/c)**2)) + \
        ((flux_b/ratio_hb_b) / (np.sqrt(2*np.pi) * (np.sqrt((sig_b**2)+(inst_sigma_blue**2))*lam0_hb/c))) * np.exp(-(x_hb-((vel_b*lam0_hb/c)+lam0_hb))**2 / (2*(np.sqrt((sig_b**2)+(inst_sigma_blue**2))*lam0_hb/c)**2))), \
            
        ((a_ha + (b_ha*x_ha)) + \
        ((ratio_n2*flux/2.94) / (np.sqrt(2*np.pi) * (np.sqrt((sig**2)+(inst_sigma_red**2))*lam01/c))) * np.exp(-(x_ha-((vel*lam01/c)+lam01))**2 / (2*(np.sqrt((sig**2)+(inst_sigma_red**2))*lam01/c)**2)) + \
        (flux / (np.sqrt(2*np.pi) * (np.sqrt((sig**2)+(inst_sigma_red**2))*lam02/c))) * np.exp(-(x_ha-((vel*lam02/c)+lam02))**2 / (2*(np.sqrt((sig**2)+(inst_sigma_red**2))*lam02/c)**2)) + \
        ((ratio_n2*flux) / (np.sqrt(2*np.pi) * (np.sqrt((sig**2)+(inst_sigma_red**2))*lam03/c))) * np.exp(-(x_ha-((vel*lam03/c)+lam03))**2 / (2*(np.sqrt((sig**2)+(inst_sigma_red**2))*lam03/c)**2)) + \
        ((ratio_n2_b*flux_b/2.94) / (np.sqrt(2*np.pi) * (np.sqrt((sig_b**2)+(inst_sigma_red**2))*lam01/c))) * np.exp(-(x_ha-((vel_b*lam01/c)+lam01))**2 / (2*(np.sqrt((sig_b**2)+(inst_sigma_red**2))*lam01/c)**2)) + \
        (flux_b / (np.sqrt(2*np.pi) * (np.sqrt((sig_b**2)+(inst_sigma_red**2))*lam02/c))) * np.exp(-(x_ha-((vel_b*lam02/c)+lam02))**2 / (2*(np.sqrt((sig_b**2)+(inst_sigma_red**2))*lam02/c)**2)) + \
        ((ratio_n2_b*flux_b) / (np.sqrt(2*np.pi) * (np.sqrt((sig_b**2)+(inst_sigma_red**2))*lam03/c))) * np.exp(-(x_ha-((vel_b*lam03/c)+lam03))**2 / (2*(np.sqrt((sig_b**2)+(inst_sigma_red**2))*lam03/c)**2)))  ))
