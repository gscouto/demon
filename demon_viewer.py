from tkinter import *
from astropy.io import fits
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import h5py
import tables
import matplotlib.gridspec as gridspec
from matplotlib.ticker import NullFormatter,MultipleLocator, FormatStrFormatter
import numpy as np
from functools import partial
from matplotlib.colorbar import Colorbar
from matplotlib.colors import LogNorm
import matplotlib.patches as patches
import os

dir = 'HATLASJ085828/Hb_Ha_N2_vorbin10/'

fit_results = h5py.File(dir+'fit.hdf5', 'r')
t = tables.open_file(dir+'results.h5')

def update_plot():
    
    x = int(x1.get())
    y = int(y1.get())
    
    if 'Hb_Ha_N2' in dir:

        ax1.clear()
        ax1.set_title('x = '+str(x)+', y = '+str(y))
        ax1.plot(fit_results['lam'][0:int(len(fit_results['lam'])/2)],fit_results['data'][0:int(len(fit_results['lam'])/2),y,x],'.')
        ax1.plot(fit_results['lam'][0:int(len(fit_results['lam'])/2)],fit_results['data'][0:int(len(fit_results['lam'])/2),y,x],color='black')
        ax1.plot(fit_results['lam'][0:int(len(fit_results['lam'])/2)],fit_results['model'][0:int(len(fit_results['lam'])/2),y,x],color='red')
        if '2g' in dir:
            ax1.plot(fit_results['lam'][0:int(len(fit_results['lam'])/2)],fit_results['model_2g_n'][0:int(len(fit_results['lam'])/2),y,x],'--',color='green',linewidth=0.7)
            ax1.plot(fit_results['lam'][0:int(len(fit_results['lam'])/2)],fit_results['model_2g_b'][0:int(len(fit_results['lam'])/2),y,x],'--',color='blue',linewidth=0.7)
        ax1.xaxis.set_major_formatter( NullFormatter() )
        ax1.grid(True,alpha=0.3)

        ax2.clear()
        ax2.plot(fit_results['lam'][0:int(len(fit_results['lam'])/2)],fit_results['residual'][0:int(len(fit_results['lam'])/2),y,x])
        ax2.grid(True,alpha=0.3)
        
        ax3.clear()
        ax3.plot(fit_results['lam'][int(len(fit_results['lam'])/2):],fit_results['data'][int(len(fit_results['lam'])/2):,y,x],'.')
        ax3.plot(fit_results['lam'][int(len(fit_results['lam'])/2):],fit_results['data'][int(len(fit_results['lam'])/2):,y,x],color='black')
        ax3.plot(fit_results['lam'][int(len(fit_results['lam'])/2):],fit_results['model'][int(len(fit_results['lam'])/2):,y,x],color='red')
        if '2g' in dir:
            ax3.plot(fit_results['lam'][int(len(fit_results['lam'])/2):],fit_results['model_2g_n'][int(len(fit_results['lam'])/2):,y,x],'--',color='green',linewidth=0.7)
            ax3.plot(fit_results['lam'][int(len(fit_results['lam'])/2):],fit_results['model_2g_b'][int(len(fit_results['lam'])/2):,y,x],'--',color='blue',linewidth=0.7)
        ax3.xaxis.set_major_formatter( NullFormatter() )
        ax3.grid(True,alpha=0.3)
        ax3.yaxis.tick_right()

        ax4.clear()
        ax4.plot(fit_results['lam'][int(len(fit_results['lam'])/2):],fit_results['residual'][int(len(fit_results['lam'])/2):,y,x])
        ax4.grid(True,alpha=0.3)
        ax4.yaxis.tick_right()
        
    else:

        ax1.clear()
        ax1.set_title('x = '+str(x)+', y = '+str(y))
        ax1.plot(fit_results['lam'],fit_results['data'][:,y,x],'.')
        ax1.plot(fit_results['lam'],fit_results['data'][:,y,x],color='black')
        ax1.plot(fit_results['lam'],fit_results['model'][:,y,x],color='red')
        if '2g' in dir:
            ax1.plot(fit_results['lam'],fit_results['model_2g_n'][:,y,x],'--',color='green',linewidth=0.7)
            ax1.plot(fit_results['lam'],fit_results['model_2g_b'][:,y,x],'--',color='blue',linewidth=0.7)
        ax1.xaxis.set_major_formatter( NullFormatter() )
        ax1.grid(True,alpha=0.3)
        
        ax2.clear()
        ax2.plot(fit_results['lam'],fit_results['residual'][:,y,x])
        ax2.grid(True,alpha=0.3)
    
    
    for k in np.arange(len(maps_names)):
        if 'residuals' in maps_names[k]:
            pv[k].set(str(maps_names[k])+' = '+str('{:.3g}'.format(t.root.results.col(maps_names[k])[np.where((x_t == x)&(y_t == y))][0])))
        else:
            pv[k].set(str(maps_names[k])+' = '+str(round(t.root.results.col(maps_names[k])[np.where((x_t == x)&(y_t == y))][0],2)))        

    plot_canvas.draw()

    #fig_up, (ax_up) = plt.subplots(figsize=(8,6),ncols=1)
    #im_up = ax_up.imshow(map_z,origin='bottom',vmin=np.nanmin(map_z),vmax=np.nanmax(map_z))

    #rect = patches.Rectangle((y,x),1,1,linewidth=1,edgecolor='r',facecolor='none')

    #ax_up.add_patch(rect)

    #fig_up.colorbar(im,ax=ax_up)

    #map_canvas = FigureCanvasTkAgg(fig_up, master=window)
    #map_canvas.draw()
    
    #map_canvas.get_tk_widget().grid(row=0,column=0,rowspan=5,columnspan=3)

def change_map(map_name=None):
    
    mmap = t.root.results.col(map_name)

    for i in x_t:
        for j in y_t:
            map_z[j,i] = mmap[np.where((x_t == i)&(y_t == j))]
            
    im.set_data(map_z)
    
    if 'flux' in map_name:
        im.set_cmap('viridis')
        im.set_norm(LogNorm(vmin=0.01,vmax=np.nanmax(map_z)))
    
    if 'sig' in map_name:
        im.set_cmap('hot')
        im.set_clim(vmin=30.,vmax=100.)
        im.set_norm(None)
        
    if 'vel' in map_name:
        im.set_cmap('jet')
        im.set_clim(vmin=-np.nanmax(abs(map_z)),vmax=np.nanmax(abs(map_z)))
        im.set_norm(None)
        
    if 'residuals' in map_name:
        im.set_cmap('viridis')
        im.set_norm(LogNorm(vmin=1e-10,vmax=np.nanpercentile(map_z,90)))
        
    #print(x1,y1)
        
    map_canvas.draw()
    
    #map_canvas.get_tk_widget().grid(row=0,column=0,rowspan=5,columnspan=3)

def close_window():
    
    window.destroy()

colnames = t.root.results.colnames

maps_names = np.ravel([s for s in colnames if (("flux" in s) | ("vel" in s) | ("sig" in s) | ("residuals" in s)) & ("err" not in s)])

x_t = t.root.results.col('x').astype(int)
y_t = t.root.results.col('y').astype(int)

map_z = np.zeros([np.max(y_t)+1,np.max(x_t)+1])

window = Tk()
window.title('linefit plotter')
#window.geometry('1500x500')
window.configure(background='white')

x = int(fit_results['data'].shape[2]/2.)
y = int(fit_results['data'].shape[1]/2.)

x1 = x
y1 = y

fig, (ax1) = plt.subplots(figsize=(8,6),ncols=1)
im = ax1.imshow(map_z,origin='lower',vmin=np.nanmin(map_z),vmax=np.nanmax(map_z))

#rect = patches.Rectangle((y,x),1,1,linewidth=1,edgecolor='r',facecolor='none')

#ax1.add_patch(rect)

fig.colorbar(im,ax=ax1)

map_canvas = FigureCanvasTkAgg(fig, master=window)
map_canvas.draw()
#canvas.get_tk_widget().pack(side="left", fill="both")

x1 = Entry(window,width=10)
y1 = Entry(window,width=10)

#map_changer = Label(window, text="map changer")
map_changer = Menubutton(window, text='Change Map')
map_changer.menu = Menu(map_changer)
map_changer['menu'] = map_changer.menu

for mn in maps_names:
    map_changer.menu.add_command(label=mn,command=partial(change_map,mn))

xl = Label(window, text="x:",background='white')
yl = Label(window, text="y:",background='white')

b_update = Button(window,text='Update Spectra', command=update_plot)

b_quit = Button(window,text='Quit', command=close_window)

if 'Hb_Ha_N2' in dir:

    fig = plt.figure(figsize=(15,3))

    gs = gridspec.GridSpec(2,2, height_ratios=[1,0.5], width_ratios=[1,1])
    gs.update(left=0.07, right=0.92, bottom=0.1, top=0.90, wspace=0.0, hspace=0.0)
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

    ax1 = plt.subplot(gs[0,0])
    ax1.set_title('x = '+str(x)+', y = '+str(y))
    ax1.plot(fit_results['lam'][0:int(len(fit_results['lam'])/2)],fit_results['data'][0:int(len(fit_results['lam'])/2),y,x],'.')
    ax1.plot(fit_results['lam'][0:int(len(fit_results['lam'])/2)],fit_results['data'][0:int(len(fit_results['lam'])/2),y,x],color='black')
    ax1.plot(fit_results['lam'][0:int(len(fit_results['lam'])/2)],fit_results['model'][0:int(len(fit_results['lam'])/2),y,x],color='red')
    if '2g' in dir:
        ax1.plot(fit_results['lam'][0:int(len(fit_results['lam'])/2)],fit_results['model_2g_n'][0:int(len(fit_results['lam'])/2),y,x],'--',color='green',linewidth=0.7)
        ax1.plot(fit_results['lam'][0:int(len(fit_results['lam'])/2)],fit_results['model_2g_b'][0:int(len(fit_results['lam'])/2),y,x],'--',color='blue',linewidth=0.7)
    ax1.xaxis.set_major_formatter( NullFormatter() )
    ax1.grid(True,alpha=0.3)

    ax2 = plt.subplot(gs[1,0])
    ax2.plot(fit_results['lam'][0:int(len(fit_results['lam'])/2)],fit_results['residual'][0:int(len(fit_results['lam'])/2),y,x])
    ax2.grid(True,alpha=0.3)
    
    ax3 = plt.subplot(gs[0,1])
    ax3.plot(fit_results['lam'][int(len(fit_results['lam'])/2):],fit_results['data'][int(len(fit_results['lam'])/2):,y,x],'.')
    ax3.plot(fit_results['lam'][int(len(fit_results['lam'])/2):],fit_results['data'][int(len(fit_results['lam'])/2):,y,x],color='black')
    ax3.plot(fit_results['lam'][int(len(fit_results['lam'])/2):],fit_results['model'][int(len(fit_results['lam'])/2):,y,x],color='red')
    if '2g' in dir:
        ax3.plot(fit_results['lam'][int(len(fit_results['lam'])/2):],fit_results['model_2g_n'][int(len(fit_results['lam'])/2):,y,x],'--',color='green',linewidth=0.7)
        ax3.plot(fit_results['lam'][int(len(fit_results['lam'])/2):],fit_results['model_2g_b'][int(len(fit_results['lam'])/2):,y,x],'--',color='blue',linewidth=0.7)
    ax3.xaxis.set_major_formatter( NullFormatter() )
    ax3.grid(True,alpha=0.3)
    ax3.yaxis.tick_right()

    ax4 = plt.subplot(gs[1,1])
    ax4.plot(fit_results['lam'][int(len(fit_results['lam'])/2):],fit_results['residual'][int(len(fit_results['lam'])/2):,y,x])
    ax4.grid(True,alpha=0.3)
    ax4.yaxis.tick_right()

    plot_canvas = FigureCanvasTkAgg(fig, master=window)
    plot_canvas.draw()
    
else:
    
    fig = plt.figure(figsize=(15,3))

    gs = gridspec.GridSpec(2,1, height_ratios=[1,0.5], width_ratios=[1])
    gs.update(left=0.07, right=0.92, bottom=0.1, top=0.90, wspace=0.2, hspace=0.0)
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

    ax1 = plt.subplot(gs[0,0])
    ax1.set_title('x = '+str(x)+', y = '+str(y))
    ax1.plot(fit_results['lam'],fit_results['data'][:,y,x],'.')
    ax1.plot(fit_results['lam'],fit_results['data'][:,y,x],color='black')
    ax1.plot(fit_results['lam'],fit_results['model'][:,y,x],color='red')
    if '2g' in dir:
        ax1.plot(fit_results['lam'],fit_results['model_2g_n'][:,y,x],'--',color='green',linewidth=0.7)
        ax1.plot(fit_results['lam'],fit_results['model_2g_b'][:,y,x],'--',color='blue',linewidth=0.7)
    ax1.xaxis.set_major_formatter( NullFormatter() )
    ax1.grid(True,alpha=0.3)

    ax2 = plt.subplot(gs[1,0])
    ax2.plot(fit_results['lam'],fit_results['residual'][:,y,x])
    ax2.grid(True,alpha=0.3)

    plot_canvas = FigureCanvasTkAgg(fig, master=window)
    plot_canvas.draw()

pv = []
p = []

for k in np.arange(len(maps_names)):
    svar = StringVar()
    if 'residuals' in maps_names[k]:
        svar.set(str(maps_names[k])+' = '+str('{:.3g}'.format(t.root.results.col(maps_names[k])[np.where((x_t == x)&(y_t == y))][0])))
    else:
        svar.set(str(maps_names[k])+' = '+str(round(t.root.results.col(maps_names[k])[np.where((x_t == x)&(y_t == y))][0],2)))
    pv.append(svar)
    lp = Label(window, textvariable=pv[k],background='white')
    p.append(lp)

#pp1 = StringVar()
#pp1.set(str(maps_names[0])+' = '+str(round(t.root.results.col(maps_names[0])[np.where((x_t == x)&(y_t == y))][0],2)))
#p1 = Label(window, textvariable=pp1)
#p2 = Label(window, text="p2",background='red')
#p3 = Label(window, text="p3",background='red')
#p4 = Label(window, text="p4",background='red')
#p5 = Label(window, text="p5",background='red')
#p6 = Label(window, text="p6",background='red')
#p7 = Label(window, text="p7",background='red')
#p8 = Label(window, text="p8",background='red')
#p9 = Label(window, text="p9",background='red')
#p10 = Label(window, text="p10",background='red')
#p11 = Label(window, text="p11",background='red')
#p12 = Label(window, text="p12",background='red')

map_canvas.get_tk_widget().grid(row=0,column=0,rowspan=5,columnspan=3)
plot_canvas.get_tk_widget().grid(row=6,column=0,rowspan=2,columnspan=6,sticky=S)
map_changer.grid(row=5,column=1)
xl.grid(row=0,column=2,sticky=E)
yl.grid(row=1,column=2,sticky=E)
b_update.grid(row=0,column=4)
b_quit.grid(row=1,column=4)
x1.grid(row=0,column=3,sticky=N+S+E+W)
y1.grid(row=1,column=3,sticky=N+S+E+W)
#window.grid_columnconfigure(0, weight=1)
#window.grid_rowconfigure(8, weight=1)

rows = np.array([2,3,4,5])
columns = np.array([3,4,5])

c = 0

for j in columns:
    for i in rows:
        p[c].grid(row=i,column=j) 
        c = c+1
        if c == len(p):
            break
    if c == len(p):
        break

#p1.grid(row=2,column=3)
#p2.grid(row=3,column=3)
#p3.grid(row=4,column=3)
#p4.grid(row=5,column=3)

#p5.grid(row=2,column=4)
#p6.grid(row=3,column=4)
#p7.grid(row=4,column=4)
#p8.grid(row=5,column=4)

#p9.grid(row=2,column=5)
#p10.grid(row=3,column=5)
#p11.grid(row=4,column=5)
#p12.grid(row=5,column=5)

window.mainloop()
