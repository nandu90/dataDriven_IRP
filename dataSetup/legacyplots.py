"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Author: nsaini
Created: 2019-07-08
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""

from cycler import cycler
import math
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import inp


    

def extractPlane(plnindex,plnindices,data,y,z,fname,ylabel):

    dw, dwdata = extractPlaneData(plnindex, plnindices,\
                    data, y, z)

    yplus = dw*inp.rho*inp.utau/inp.mu

    default_cycler = (cycler(color=['b', 'r', 'k', 'm','g']) * \
                  cycler(linestyle=['-']) * cycler(marker=['.']))
    plt.rc('lines', linewidth=1)
    plt.rc('axes', prop_cycle=default_cycler)
    

    fig = plt.figure(figsize=(7,5))
    ax = fig.add_subplot(111)
    ax.set_xlabel('$y^+$',fontsize=20)
    ax.set_ylabel(ylabel,fontsize=20)
    ax.semilogx(yplus,dwdata)
    ax.grid()
    if(fname == 'Umean'):
        kappa = 0.41
        B = 5.0
        uPlusSublayer = []
        yPlusSublayer = []
        uPlusLog = []
        yPlusLog = []
        for i in yplus:
            if(i<=15.0):
                uPlusSublayer.append(i)
                yPlusSublayer.append(i)
            else:
                uPlusLog.append(math.log(i)/kappa + B)
                yPlusLog.append(i)
        ax.semilogx(yPlusSublayer, uPlusSublayer,'--r')
        ax.semilogx(yPlusLog, uPlusLog,'--k')
    fig.savefig(fname+'_yplus.png',quality=100,bbox_inches='tight',dpi=500)

    # fig2 = plt.figure(figsize=(7,5))
    # ax2 = fig2.add_subplot(111)
    # ax2.set_xlabel('dwall (m)',fontsize=20)
    # ax2.set_ylabel(ylabel,fontsize=20)
    # ax2.plot(dw,dwdata)
    # ax2.grid()
    # fig2.savefig(fname+'.png',quality=100,bbox_inches='tight',dpi=500)
    return


def plotMultiPlane(fname,ylabel,xplns,plns,dw,dwdata):
        
    default_cycler = (cycler(marker=['.','+','v','^','<','>'])*\
                      cycler(color=['b', 'r', 'k', 'm','g']) * \
                      cycler(linestyle=['-'])
                      )
    plt.rc('lines', linewidth=1)
    plt.rc('axes', prop_cycle=default_cycler)
    

    if(inp.plttype == '3d'):
        fname = fname+'3d'
        fig = plt.figure(figsize=(10,5))
        ax = fig.add_subplot(111,projection='3d')
        ax.set_xlabel('x (mm)',fontsize=20)
        ax.set_ylabel('$y^+$',fontsize=20)
        ax.set_zlabel(ylabel,fontsize=20)
    else:
        fig = plt.figure(figsize=(7,5))
        ax = fig.add_subplot(111)
        ax.set_xlabel('$y^+$',fontsize=20)
        ax.set_ylabel(ylabel,fontsize=20)

    for ipln in range(len(plns)):
        yplus = dw[ipln]*inp.rho*inp.utau/inp.mu        
        strlabel = 'x = %.2f mm'%(xplns[plns[ipln]]*1000.0)
        xdummy = np.full(dwdata.shape[1],xplns[plns[ipln]]*1000.0)
        if(inp.plttype == '3d'):
            ax.plot(xdummy,yplus,dwdata[ipln],label=strlabel)
        else:
            ax.semilogx(yplus,dwdata[ipln],label = strlabel)
        
    if(inp.plttype == '1d'):     
        ax.grid()
        ax.legend(loc='best',fontsize=15)
    fig.savefig('multi'+fname+'_yplus.png',quality=100,\
                bbox_inches='tight',dpi=500)

    return


def extractMean(dwclass,plnmask,data):
    dataplane = data[plnmask]
    dwclassarr = np.asarray(dwclass)
    nclass = max(dwclass)
    dwdata = np.zeros(nclass)
    
    for i in range(nclass):
        dwmask = np.where(dwclassarr == i+1, True, False)
        datatemp = dataplane[dwmask]
        dwdata[i] = np.mean(datatemp)
    
    return dwdata

def getdw(yplane,zplane):    
    
    dwclass = getdwclass(yplane, zplane)   
    nclass = np.amax(dwclass)
    dw = np.zeros(nclass)

    for iclass in range(nclass):
        dwmask = np.where(dwclass == iclass+1, True, False)
        ytemp = yplane[dwmask]
        ztemp = zplane[dwmask]        
        dwtemp = np.sqrt(np.power(np.abs(ytemp)-inp.pitch,2)\
                +np.power(np.abs(ztemp)-inp.pitch,2))-inp.rrod    
        dw[iclass] = np.mean(dwtemp)

    return dwclass, dw

def getdwclass(yplane, zplane):
    dwclass = np.zeros(yplane.shape[0],dtype=int)
    coord = np.loadtxt("xyzts.dat",skiprows=1)
    yx = coord[:,1]
    zx = coord[:,2]
    
    for ihom in range(inp.nhom):
        ysub = yx[ihom::inp.nhom]
        zsub = zx[ihom::inp.nhom]
        
        for i in range(ysub.shape[0]):
            for j in range(yplane.shape[0]):
                if(dwclass[j] == 0):
                    if(ysub[i]*yplane[j] > 0.0 and \
                       zsub[i]*zplane[j] > 0.0):
                        dist = math.pow(ysub[i]-yplane[j],2)
                        dist += math.pow(zsub[i]-zplane[j],2)
                        dist = math.sqrt(dist)
                        if(dist < 1.0e-10):
                            dwclass[j] = i+1
    return dwclass

def extractfluct(probedata,plnmask,dwclass,udw,vdw,wdw): 
    
    nclass = np.amax(dwclass)
    
    udw = np.array(udw)
    vdw = np.array(vdw)
    wdw = np.array(wdw)

    um = np.zeros(dwclass.shape[0])
    vm = np.zeros(dwclass.shape[0])
    wm = np.zeros(dwclass.shape[0])

    for iclass in range(nclass):
        um = np.where(dwclass == iclass+1,udw[iclass],um)
        vm = np.where(dwclass == iclass+1,vdw[iclass],vm)
        wm = np.where(dwclass == iclass+1,wdw[iclass],wm)

    uprime = np.zeros((inp.totalsteps,dwclass.shape[0]))
    vprime = np.zeros((inp.totalsteps,dwclass.shape[0]))
    wprime = np.zeros((inp.totalsteps,dwclass.shape[0]))
    dt = np.zeros((inp.totalsteps,dwclass.shape[0]))
    
    for istep in range(inp.totalsteps):
        uplane = probedata[istep,:,1][plnmask]
        vplane = probedata[istep,:,2][plnmask]
        wplane = probedata[istep,:,3][plnmask]
        tplane = probedata[istep,:,0][plnmask]
        
        uprime[istep,:] = np.subtract(uplane,um)
        vprime[istep,:] = np.subtract(vplane,vm)
        wprime[istep,:] = np.subtract(wplane,wm)
        dt[istep,:] = tplane

    return uprime, vprime, wprime, dt

def extractTKE(uprime,vprime,wprime,dt,dwclass):
    nclass = np.amax(dwclass)
        
    uut = np.sum(uprime*uprime*dt,axis=0)
    vvt = np.sum(vprime*vprime*dt,axis=0)
    wwt = np.sum(wprime*wprime*dt,axis=0)
    Dt = np.sum(dt,axis=0)
    TKEt = 0.5*(uut+vvt+wwt)
    
    Rstresst = np.zeros((6,uut.shape[0]))
    Rstresst[0,:] = uut
    Rstresst[1,:] = vvt
    Rstresst[2,:] = wwt
    Rstresst[3,:] = np.sum(uprime*vprime*dt,axis=0)
    Rstresst[4,:] = np.sum(uprime*wprime*dt,axis=0)
    Rstresst[5,:] = np.sum(vprime*wprime*dt,axis=0)

    TKE = np.zeros(nclass)
    Rstress = np.zeros((6,nclass))
    for iclass in range(nclass):
        dwmask = np.where(dwclass == iclass+1,True,False)
        Deltat = np.sum(Dt[dwmask])
        TKE[iclass] = np.sum(TKEt[dwmask])/Deltat
        for i in range(6):
            Rstress[i,iclass] = np.sum(Rstresst[i,:][dwmask])/Deltat

    return TKE, Rstress


def extractPlots(plnindices,xplns,y,z,probedata,umean,vmean,wmean):
    
    plns = list(range(0,xplns.shape[0],int(xplns.shape[0]/inp.nplot)))
    dw = [[] for i in range(len(plns))]
    udw = [[] for i in range(len(plns))]
    vdw = [[] for i in range(len(plns))]
    wdw = [[] for i in range(len(plns))]
    TKE = [[] for i in range(len(plns))]
    Rstress = [[] for i in range(len(plns))]
    
    print("Planes being processed: ",plns)

    for ipln in range(len(plns)):
        plnmask = np.where(plnindices == plns[ipln], True, False)
        yplane = y[plnmask]
        zplane = z[plnmask]
        
        dwclass, dw[ipln] = getdw(yplane,zplane)
        print("Syncronized Plane %i with legacy xyzts"%(plns[ipln]))

        udw[ipln] = extractMean(dwclass,plnmask,umean)
        vdw[ipln] = extractMean(dwclass,plnmask,vmean)
        wdw[ipln] = extractMean(dwclass,plnmask,wmean)
        
        uprime, vprime, wprime, dt = extractfluct(probedata,\
                    plnmask,dwclass,\
                    udw[ipln],vdw[ipln],wdw[ipln])
        TKE[ipln], Rstress[ipln] = extractTKE(uprime,vprime,wprime,dt,dwclass)

    dw = np.array(dw)
    udw = np.array(udw)
    vdw = np.array(vdw)
    wdw = np.array(wdw)
    TKE = np.array(TKE)
    Rstress = np.array(Rstress)
    
    print("Creating (Legacy) Plots Now")
    plotMultiPlane('Umean','$U^+$',xplns,plns,dw,udw/inp.utau)
    plotMultiPlane('Vmean','$V^+$',xplns,plns,dw,vdw/inp.utau)
    plotMultiPlane('Wmean','$W^+$',xplns,plns,dw,wdw/inp.utau)
    plotMultiPlane('TKE','$TKE (m^2/s^2)$',xplns,plns,dw,TKE)

    plotMultiPlane('Rxx','$R_{xx}$',xplns,plns,dw,Rstress[:,0,:]/math.pow(inp.utau,2))
    plotMultiPlane('Rnn','$R_{nn}$',xplns,plns,dw,Rstress[:,1,:]/math.pow(inp.utau,2))
    plotMultiPlane('Rtt','$R_{tt}$',xplns,plns,dw,Rstress[:,2,:]/math.pow(inp.utau,2))
    plotMultiPlane('Rxn','$R_{xn}$',xplns,plns,dw,Rstress[:,3,:]/math.pow(inp.utau,2))
    plotMultiPlane('Rxt','$R_{xt}$',xplns,plns,dw,Rstress[:,4,:]/math.pow(inp.utau,2))
    plotMultiPlane('Rnt','$R_{nt}$',xplns,plns,dw,Rstress[:,5,:]/math.pow(inp.utau,2))
    
    # with open('compare.txt','w') as f:
    #     for i in range(dw.shape[1]):
    #         f.write('%.6e %.6e %.6e %.6e %.6e\n'%(dw[0][i],udw[0][i],vdw[0][i],wdw[0][i],TKE[0][i]))

    
    return
