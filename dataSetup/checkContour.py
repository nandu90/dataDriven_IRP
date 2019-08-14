"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Author: nsaini
Created: 2019-08-14
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""

from cycler import cycler
import math
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import os
import time
from scipy.interpolate import griddata


def plot_contour(x,y,z,resolution = 50,contour_method='linear'):
    resolution = str(resolution)+'j'
    X,Y = np.mgrid[min(x):max(x):complex(resolution),min(y):max(y):complex(resolution)]
    points = [[a,b] for a,b in zip(x,y)]
    Z = griddata(points, z, (X, Y), method=contour_method)
    return X,Y,Z

def plot_data(yplane,zplane,data,fname,pln,xplns):
    X,Y,Z = plot_contour(yplane,zplane,data,\
                         resolution=2000,contour_method='linear')

    fig = plt.figure(figsize=(7,5))
    ax = fig.add_subplot(111)
    CS = ax.contourf(X,Y,Z, 100, origin='lower')
    cbar = fig.colorbar(CS)
    label = 'x = %.2f mm'%(xplns[pln]*1000.0)
    fig.suptitle(fname+' @ '+label,fontsize=20)
    fig.savefig('contourPlots/'+fname+'_'+str(pln),quality=100,bbox_inches='tight',dpi=200)
    plt.close()

    return
    

def main():

    pln = 23

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
    plot_data(yplane,zplane,uplane,'U',pln,xplns)
    plot_data(yplane,zplane,vplane,'V',pln,xplns)
    plot_data(yplane,zplane,wplane,'W',pln,xplns)
    
    plot_data(yplane,zplane,R11plane,'R11',pln,xplns)
    plot_data(yplane,zplane,R22plane,'R22',pln,xplns)
    plot_data(yplane,zplane,R33plane,'R33',pln,xplns)
    plot_data(yplane,zplane,R12plane,'R12',pln,xplns)
    plot_data(yplane,zplane,R13plane,'R13',pln,xplns)
    plot_data(yplane,zplane,R23plane,'R23',pln,xplns)
    return


if __name__ == "__main__":
    starttime = time.time()
    main()
    print("--- Code ran in %s seconds ---"%(time.time()-starttime))
