"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Author: nsaini
Created: 2019-08-14
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""

from cycler import cycler
import math
import numpy as np
#import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pyplot as plt
import os
import time
from scipy.interpolate import griddata
from matplotlib import cm

def plot_contour(x,y,z,resolution=1.e-4,contour_method='linear'):
    X,Y = np.mgrid[min(x):max(x):resolution,min(y):max(y):resolution]

    points = [[a,b] for a,b in zip(x,y)]
    Z = griddata(points, z, (X, Y), method=contour_method, fill_value=np.nan)
    
    rrod = 4.57e-3
    corner = 0.0063
    dist = np.sqrt(np.power(corner-np.abs(X),2)+\
                   np.power(corner-np.abs(Y),2))-rrod
    
    mask = np.where(dist < 0., True, False)
    Z[mask] = np.nan
    
    return X,Y,Z

def plot_data(yplane,zplane,data,fname,pln,xplns,vmin,vmax,mapfactor=1.5):
    X,Y,Z = plot_contour(yplane,zplane,data,\
                         resolution=1.e-5,contour_method='linear')

    fig = plt.figure(figsize=(7,5))
    ax = fig.add_subplot(111)
    levels = np.linspace(vmin,vmax,100)
    CS = ax.contourf(X,Y,Z, levels=levels, origin='lower',cmap=cm.coolwarm,vmin=vmin/mapfactor,vmax=vmax/mapfactor)
    cbar = fig.colorbar(CS).ax.tick_params(labelsize=15)
    
    
    label = 'x/$D_h$ = %.2f'%(xplns[pln]/12.976e-3)
    plt.tick_params(axis='x',which='both',bottom=False, top=False,labelbottom=False,labeltop=False) 
    plt.tick_params(axis='y',which='both',left=False, labelleft=False)
    
    fig.suptitle(fname+' @ '+label,fontsize=20)
    fig.savefig('contourPlots/'+fname+'_'+str(pln),quality=100,bbox_inches='tight',dpi=500)
    plt.show()
    plt.close()

    return


def main():


    plns = np.array([15,18,21,24])
    xp = plns*(0.04/30.)

    try:
        os.mkdir('contourPlots')
        print('dir created')
    except FileExistsError:
        print('dir exists')

    fname = 'output/probeCoord.csv'
    data = np.loadtxt(fname, comments='#', delimiter=',')
    x = data[:,0]
    y = data[:,1]
    z = data[:,2]

    fname = 'output/RStress.csv'
    data = np.loadtxt(fname, comments='#', delimiter=',')
    R11 = data[:,0]
    R22 = data[:,1]
    R33 = data[:,2]
    R12 = data[:,3]
    R13 = data[:,4]
    R23 = data[:,5]

    fname = 'output/velMean.csv'
    data = np.loadtxt(fname, comments='#', delimiter=',')
    u = data[:,0]
    v = data[:,1]
    w = data[:,2]

    xplns, plnindices, plncount = np.unique(x,return_inverse=True,return_counts=True)
    print('Number of identified x-planes: ',xplns.shape[0])
    print(xplns)

    ulim = [1000,-1000]
    vlim = [1000,-1000]
    wlim = [1000,-1000]
    
    rxxlim = [1000,-1000]
    ryylim = [1000,-1000]
    rzzlim = [1000,-1000]
    rxylim = [1000,-1000]
    rxzlim = [1000,-1000]
    ryzlim = [1000,-1000]
    for pln in plns:
        planemask = np.where(plnindices == pln,True,False)
        yplane = y[planemask]
        zplane = z[planemask]
        uplane = u[planemask]
        vplane = v[planemask]
        wplane = w[planemask]

        R11plane = R11[planemask]
        R22plane = R22[planemask]
        R33plane = R33[planemask]
        R12plane = R12[planemask]
        R13plane = R13[planemask]
        R23plane = R23[planemask]
        
        ulim[0] = min(ulim[0],np.amin(uplane))  
        ulim[1] = max(ulim[1],np.amax(uplane))
        
        vlim[0] = min(vlim[0],np.amin(vplane))  
        vlim[1] = max(vlim[1],np.amax(vplane))
        
        wlim[0] = min(wlim[0],np.amin(wplane))  
        wlim[1] = max(wlim[1],np.amax(wplane))
        
        rxxlim[0] = min(rxxlim[0],np.amin(R11plane))  
        rxxlim[1] = max(rxxlim[1],np.amax(R11plane))
        
        ryylim[0] = min(ryylim[0],np.amin(R22plane))  
        ryylim[1] = max(ryylim[1],np.amax(R22plane))
        
        rzzlim[0] = min(rzzlim[0],np.amin(R33plane))  
        rzzlim[1] = max(rzzlim[1],np.amax(R33plane))
        
        rxylim[0] = min(rxylim[0],np.amin(R12plane))  
        rxylim[1] = max(rxylim[1],np.amax(R12plane))
        
        rxzlim[0] = min(rxzlim[0],np.amin(R13plane))  
        rxzlim[1] = max(rxzlim[1],np.amax(R13plane))
        
        ryzlim[0] = min(ryzlim[0],np.amin(R23plane))  
        ryzlim[1] = max(ryzlim[1],np.amax(R23plane))
        
    for pln in plns:
        planemask = np.where(plnindices == pln,True,False)
        yplane = y[planemask]
        zplane = z[planemask]
        uplane = u[planemask]
        vplane = v[planemask]
        wplane = w[planemask]

        R11plane = R11[planemask]
        R22plane = R22[planemask]
        R33plane = R33[planemask]
        R12plane = R12[planemask]
        R13plane = R13[planemask]
        R23plane = R23[planemask]
        
        # Contour plot
        plot_data(yplane,zplane,uplane,'U',pln,xplns,vmin=ulim[0],vmax=ulim[1],mapfactor=1)
        plot_data(yplane,zplane,vplane,'V',pln,xplns,vmin=vlim[0],vmax=vlim[1])
        plot_data(yplane,zplane,wplane,'W',pln,xplns,vmin=wlim[0],vmax=wlim[1])
        
        plot_data(yplane,zplane,R11plane,'Rxx',pln,xplns,vmin=rxxlim[0],vmax=rxxlim[1])
        plot_data(yplane,zplane,R22plane,'Ryy',pln,xplns,vmin=ryylim[0],vmax=ryylim[1],mapfactor=2)
        plot_data(yplane,zplane,R33plane,'Rzz',pln,xplns,vmin=rzzlim[0],vmax=rzzlim[1])
        plot_data(yplane,zplane,R12plane,'Rxy',pln,xplns,vmin=rxylim[0],vmax=rxylim[1],mapfactor=2)
        plot_data(yplane,zplane,R13plane,'Rxz',pln,xplns,vmin=rxzlim[0],vmax=rxzlim[1])
        plot_data(yplane,zplane,R23plane,'Ryz',pln,xplns,vmin=ryzlim[0],vmax=ryzlim[1],mapfactor=2)
        
    return


if __name__ == "__main__":
    starttime = time.time()
    main()
    print("--- Code ran in %s seconds ---"%(time.time()-starttime))
