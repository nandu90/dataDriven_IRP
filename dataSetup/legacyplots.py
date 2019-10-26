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
import matplotlib.ticker as ticker
from mpl_toolkits.mplot3d import Axes3D
import inp
import funcs


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


def plotMultiPlane(fname,ylabel,xplns,plns,dw,dwdata,axistype=0):
        
    default_cycler = (cycler(marker=['.','+','v','^','<','>'])*\
                      cycler(color=['b', 'r', 'k', 'm','g']) * \
                      cycler(linestyle=['-'])
                      )
    plt.rc('lines', linewidth=1)
    plt.rc('axes', prop_cycle=default_cycler)
    
    name = fname
    fname = inp.cwd+'/legacyData/plots/'+fname

    if(inp.plttype == '3d' and name != 'anisotropy'):
        fname = fname+'3d'
        fig = plt.figure(figsize=(10,5))
        ax = fig.add_subplot(111,projection='3d')
        ax.set_xlabel('x (mm)',fontsize=20)
        ax.set_ylabel('$y^+$',fontsize=20)
        ax.set_zlabel(ylabel,fontsize=20)
    else:
        fig = plt.figure(figsize=(7,5))
        ax = fig.add_subplot(111)
        if(name != 'anisotropy'):
            ax.set_xlabel('$y^+$',fontsize=20)
        else:
            ax.set_xlabel('$\\xi$',fontsize=20)
        ax.set_ylabel(ylabel,fontsize=20)

    for ipln in range(len(plns)):
        if(name != 'anisotropy'):
            yplus = dw[ipln]*inp.rho*inp.utau/inp.mu  
        else:
            yplus = dw[ipln]
        strlabel = 'x = %.2f mm'%(xplns[plns[ipln]]*1000.0)
        xdummy = np.full(dwdata.shape[1],xplns[plns[ipln]]*1000.0)
        if(inp.plttype == '3d'):
            ax.plot(xdummy,yplus,dwdata[ipln],label=strlabel)
        else:
            if(axistype == 0):
                ax.semilogx(yplus,dwdata[ipln],label = strlabel)
            elif(axistype == 1):
                ax.plot(yplus,dwdata[ipln],label=strlabel)

    if(name == 'anisotropy'):
        x = np.linspace(0,1./3.,num=100)
        y = x
        ax.plot(x,y,linestyle='-',marker='',color='k')

        x = np.linspace(-1./6.,0.0,num=100)
        y = -x
        ax.plot(x,y,linestyle='-',marker='',color='k')

        x = np.linspace(-1./6.,1./3.,num=100)
        y = np.sqrt(1./27. + 2.*x**3.)
        ax.plot(x,y,linestyle='-',marker='',color='k')
        ax.xaxis.set_ticks([-1./6.,0.,1./6.,1./3.])
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.3f'))

        ax.yaxis.set_ticks([0.,1./6.,1./3.])
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.3f'))

    if(inp.plttype == '1d'):     
        ax.grid()
        ax.legend(loc='best',fontsize=15)
    # elif(inp.plttype == '3d'):
    #     extraticks = [3,13,18.17]
    #     plt.xticks(list(plt.xticks()[0]) + extraticks)
    #     a = ax.get_xticks().tolist()
    #     a[-3]='spacer'
    #     a[-2]='vane'
    #     a[-1]='end vane'
    #     ax.set_xticklabels(a)
    
    if(axistype == 0):
        fig.savefig(fname+'_yplus.png',quality=100,\
                    bbox_inches='tight',dpi=500)
    elif(axistype == 1):
        fig.savefig(fname+'.png',quality=100,\
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
        uplane = probedata[istep,:,4][plnmask]
        vplane = probedata[istep,:,5][plnmask]
        wplane = probedata[istep,:,6][plnmask]
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
    bij = np.zeros((6,nclass))
    xi =np.zeros(nclass)
    eta =np.zeros(nclass)

    for iclass in range(nclass):
        dwmask = np.where(dwclass == iclass+1,True,False)
        Deltat = np.sum(Dt[dwmask])
        TKE[iclass] = np.sum(TKEt[dwmask])/Deltat
        for i in range(6):
            Rstress[i,iclass] = np.sum(Rstresst[i,:][dwmask])/Deltat
            bij[i,iclass] = (Rstress[i,iclass]/TKE[iclass])*0.5
            if(i < 3):
                bij[i,iclass] -= (1.0/3.0)
        eta[iclass] = math.pow(bij[0,iclass],2.)+\
                      math.pow(bij[1,iclass],2.)+\
                      math.pow(bij[2,iclass],2.)
        eta[iclass] = math.sqrt(eta[iclass]/6.0)

        xi[iclass] = math.pow(bij[0,iclass],3.)+\
                     math.pow(bij[1,iclass],3.)+\
                     math.pow(bij[2,iclass],3.)
        new = funcs.cubic_root(xi[iclass]/6.0)
        xi[iclass] = new
        
        mat = np.matrix([[bij[0,iclass],bij[3,iclass],bij[4,iclass]],\
                        [bij[3,iclass],bij[1,iclass],bij[5,iclass]],\
                         [bij[4,iclass],bij[5,iclass],bij[2,iclass]]])
        eig = np.linalg.eigvals(mat)
        eta[iclass] = math.sqrt((eig[0]**2.+eig[0]*eig[1]+eig[1]**2.)/3.)
        xi[iclass] = funcs.cubic_root(-(eig[0]*eig[1]*(eig[0]+eig[1]))/2.)
        print(eta[iclass],xi[iclass])
    return TKE, Rstress, bij, xi, eta


def extractPlots(plnindices,xplns,y,z,probedata,umean,vmean,wmean):
    
    plns = list(range(0,xplns.shape[0],int(xplns.shape[0]/inp.nplot)))
    dw = [[] for i in range(len(plns))]
    udw = [[] for i in range(len(plns))]
    vdw = [[] for i in range(len(plns))]
    wdw = [[] for i in range(len(plns))]
    TKE = [[] for i in range(len(plns))]
    Rstress = [[] for i in range(len(plns))]
    bij = [[] for i in range(len(plns))]
    xi = [[] for i in range(len(plns))]
    eta = [[] for i in range(len(plns))]
    
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
        TKE[ipln],Rstress[ipln],bij[ipln],xi[ipln],eta[ipln] =\
                    extractTKE(uprime,vprime,wprime,dt,dwclass)

    dw = np.array(dw)
    udw = np.array(udw)
    vdw = np.array(vdw)
    wdw = np.array(wdw)
    TKE = np.array(TKE)
    Rstress = np.array(Rstress)
    bij = np.array(bij)
    xi = np.array(xi)
    eta = np.array(eta)
    
    print("Creating (Legacy) Plots Now")
    plotMultiPlane('Umean','$U^+$',xplns,plns,dw,udw/inp.utau,1)
    plotMultiPlane('Vmean','$V^+$',xplns,plns,dw,vdw/inp.utau,1)
    plotMultiPlane('Wmean','$W^+$',xplns,plns,dw,wdw/inp.utau,1)
    plotMultiPlane('TKE','$TKE$',xplns,plns,dw,TKE/math.pow(inp.utau,2),1)

    plotMultiPlane('Rxx','$R_{xx}$',xplns,plns,dw,Rstress[:,0,:]/math.pow(inp.utau,2),1)
    plotMultiPlane('Rnn','$R_{nn}$',xplns,plns,dw,Rstress[:,1,:]/math.pow(inp.utau,2),1)
    plotMultiPlane('Rtt','$R_{tt}$',xplns,plns,dw,Rstress[:,2,:]/math.pow(inp.utau,2),1)
    plotMultiPlane('Rxn','$R_{xn}$',xplns,plns,dw,Rstress[:,3,:]/math.pow(inp.utau,2),1)
    plotMultiPlane('Rxt','$R_{xt}$',xplns,plns,dw,Rstress[:,4,:]/math.pow(inp.utau,2),1)
    plotMultiPlane('Rnt','$R_{nt}$',xplns,plns,dw,Rstress[:,5,:]/math.pow(inp.utau,2),1)

    plotMultiPlane('anisotropy','$\\eta$',xplns,plns,xi,eta,1)
    
    # with open('compare.txt','w') as f:
    #     for i in range(dw.shape[1]):
    #         f.write('%.6e %.6e %.6e %.6e %.6e\n'%(dw[0][i],udw[0][i],vdw[0][i],wdw[0][i],TKE[0][i]))

    for ipln in range(dw.shape[0]):
        fname = inp.cwd+'/legacyData/velExtracts_plane_'+str(plns[ipln])+'.csv'
        stack = np.column_stack((dw[ipln],udw[ipln],vdw[ipln],wdw[ipln],\
                Rstress[ipln,0,:],Rstress[ipln,1,:],Rstress[ipln,2,:],\
                Rstress[ipln,3,:],Rstress[ipln,4,:],Rstress[ipln,5,:],\
                TKE[ipln]))
        head = 'dwall,<u>,<v>,<w>,<u\'u\'>,<v\'v\'>,<w\'w\'>,'
        head = head + '<u\'v\'>,<u\'w\'>,<v\'w\'>,TKE'
        np.savetxt(fname,stack,header=head,delimiter=',')
    
    return


def extractgradfluct(probedata,plnmask,dwclass,press,\
                     dudx,dvdx,dwdx,\
                     dudy,dvdy,dwdy,\
                     dudz,dvdz,dwdz,\
                     dpdx,dpdy,dpdz):
    nclass = np.amax(dwclass)
    
    gradm = np.zeros((dwclass.shape[0],9))
    gradmp = np.zeros((dwclass.shape[0],3))
    pressm = np.zeros((dwclass.shape[0]))

    for iclass in range(nclass):
        gradm[:,0]=np.where(dwclass==iclass+1,dudx[iclass],gradm[:,0])
        gradm[:,1]=np.where(dwclass==iclass+1,dvdx[iclass],gradm[:,1])
        gradm[:,2]=np.where(dwclass==iclass+1,dwdx[iclass],gradm[:,2])

        gradm[:,3]=np.where(dwclass==iclass+1,dudy[iclass],gradm[:,3])
        gradm[:,4]=np.where(dwclass==iclass+1,dvdy[iclass],gradm[:,4])
        gradm[:,5]=np.where(dwclass==iclass+1,dwdy[iclass],gradm[:,5])
        
        gradm[:,6]=np.where(dwclass==iclass+1,dudz[iclass],gradm[:,6])
        gradm[:,7]=np.where(dwclass==iclass+1,dvdz[iclass],gradm[:,7])
        gradm[:,8]=np.where(dwclass==iclass+1,dwdz[iclass],gradm[:,8])

        gradmp[:,0]=np.where(dwclass==iclass+1,dpdx[iclass],gradmp[:,0])
        gradmp[:,1]=np.where(dwclass==iclass+1,dpdy[iclass],gradmp[:,1])
        gradmp[:,2]=np.where(dwclass==iclass+1,dpdz[iclass],gradmp[:,2])
        pressm=np.where(dwclass==iclass+1,press[iclass],pressm)

    gradprime = np.zeros((inp.totalsteps,dwclass.shape[0],9))
    pgradprime = np.zeros((inp.totalsteps,dwclass.shape[0],3))
    dt = np.zeros((inp.totalsteps,dwclass.shape[0]))
    pressprime = np.zeros((inp.totalsteps,dwclass.shape[0]))

    for istep in range(inp.totalsteps):
        for i in range(9):
            gradplane = probedata[istep,:,i+4][plnmask]
            gradprime[istep,:,i] = np.subtract(gradplane,gradm[:,i])
        for i in range(3):
            pgradplane = probedata[istep,:,i+13][plnmask]
            pgradprime[istep,:,i]=np.subtract(pgradplane,gradmp[:,i])

        presspln = probedata[istep,:,16][plnmask]
        pressprime[istep,:] = np.subtract(presspln,pressm)

        tplane = probedata[istep,:,0][plnmask]
        dt[istep,:] = tplane

    return gradprime, pgradprime, pressprime, dt

def extractstrain(gradprime, dt, dwclass):

    nclass = np.amax(dwclass)

    s11 = 0.5*(gradprime[:,:,0]+gradprime[:,:,0])
    s22 = 0.5*(gradprime[:,:,4]+gradprime[:,:,4])
    s33 = 0.5*(gradprime[:,:,8]+gradprime[:,:,8])
    s12 = 0.5*(gradprime[:,:,3]+gradprime[:,:,1])
    s13 = 0.5*(gradprime[:,:,6]+gradprime[:,:,2])
    s23 = 0.5*(gradprime[:,:,7]+gradprime[:,:,5])

    s11t = np.sum(s11*dt,axis=0)
    s22t = np.sum(s22*dt,axis=0)
    s33t = np.sum(s33*dt,axis=0)
    s12t = np.sum(s12*dt,axis=0)
    s13t = np.sum(s13*dt,axis=0)
    s23t = np.sum(s23*dt,axis=0)

    nu = inp.mu/inp.rho
    epst = np.sum((s11*s11+s12*s12+s13*s13+\
                   s12*s12+s22*s22+s23*s23+\
                   s13*s13+s23*s23+s33*s33)*dt,axis=0)*2.0*nu

    # epst = (s11t*s11t+s22t*s22t+s33t*s33t+\
    #         s12t*s12t+s13t*s13t+s23t*s23t)*2.0*nu

    Dt = np.sum(dt,axis=0)

    dissipation = np.zeros(nclass)
    strainprime = np.zeros((6,nclass))
    for iclass in range(nclass):
        dwmask = np.where(dwclass == iclass+1,True,False)
        Deltat = np.sum(Dt[dwmask])
        dissipation[iclass] = np.sum(epst[dwmask])/Deltat
        
        strainprime[0,iclass] = np.sum(s11t[dwmask])/Deltat
        strainprime[1,iclass] = np.sum(s22t[dwmask])/Deltat
        strainprime[2,iclass] = np.sum(s33t[dwmask])/Deltat
        strainprime[3,iclass] = np.sum(s12t[dwmask])/Deltat
        strainprime[4,iclass] = np.sum(s13t[dwmask])/Deltat
        strainprime[5,iclass] = np.sum(s23t[dwmask])/Deltat
    
    return strainprime, dissipation

def extractDissipationTensor(gradprime, dt, dwclass):

    nclass = np.amax(dwclass)
    
    d11 = gradprime[:,:,0]*gradprime[:,:,0]+\
          gradprime[:,:,3]*gradprime[:,:,3]+\
          gradprime[:,:,6]*gradprime[:,:,6]
    d22 = gradprime[:,:,1]*gradprime[:,:,1]+\
          gradprime[:,:,4]*gradprime[:,:,4]+\
          gradprime[:,:,7]*gradprime[:,:,7]
    d33 = gradprime[:,:,2]*gradprime[:,:,2]+\
          gradprime[:,:,5]*gradprime[:,:,5]+\
          gradprime[:,:,8]*gradprime[:,:,8]
    d12 = gradprime[:,:,0]*gradprime[:,:,1]+\
          gradprime[:,:,3]*gradprime[:,:,4]+\
          gradprime[:,:,6]*gradprime[:,:,7]
    d13 = gradprime[:,:,0]*gradprime[:,:,2]+\
          gradprime[:,:,3]*gradprime[:,:,5]+\
          gradprime[:,:,6]*gradprime[:,:,8]
    d23 = gradprime[:,:,1]*gradprime[:,:,2]+\
          gradprime[:,:,4]*gradprime[:,:,5]+\
          gradprime[:,:,7]*gradprime[:,:,8]

    d11t = np.sum(d11*dt,axis=0)
    d22t = np.sum(d22*dt,axis=0)
    d33t = np.sum(d33*dt,axis=0)
    d12t = np.sum(d12*dt,axis=0)
    d13t = np.sum(d13*dt,axis=0)
    d23t = np.sum(d23*dt,axis=0)

    nu = inp.mu/inp.rho

    Dt = np.sum(dt,axis=0)

    dissipation = np.zeros((6,nclass))
    for iclass in range(nclass):
        dwmask = np.where(dwclass == iclass+1,True,False)
        Deltat = np.sum(Dt[dwmask])

        dissipation[0,iclass] = np.sum(d11t[dwmask])/Deltat
        dissipation[1,iclass] = np.sum(d22t[dwmask])/Deltat
        dissipation[2,iclass] = np.sum(d33t[dwmask])/Deltat
        dissipation[3,iclass] = np.sum(d12t[dwmask])/Deltat
        dissipation[4,iclass] = np.sum(d13t[dwmask])/Deltat
        dissipation[5,iclass] = np.sum(d23t[dwmask])/Deltat

    dissipation = 2.0*nu*dissipation

    return dissipation


def extractProductionTensor(dwclass, Rstress,gradmean,plnmask):

    nclass = np.amax(dwclass)

    dudx = gradmean[:,0][plnmask]
    dvdx = gradmean[:,1][plnmask]
    dwdx = gradmean[:,2][plnmask]

    dudy = gradmean[:,3][plnmask]
    dvdy = gradmean[:,4][plnmask]
    dwdy = gradmean[:,5][plnmask]

    dudz = gradmean[:,6][plnmask]
    dvdz = gradmean[:,7][plnmask]
    dwdz = gradmean[:,8][plnmask]

    p11 = -(Rstress[:,0]*dudx+Rstress[:,0]*dudx+\
            Rstress[:,3]*dudy+Rstress[:,3]*dudy+\
            Rstress[:,4]*dudz+Rstress[:,4]*dudz)

    p22 = -(Rstress[:,3]*dvdx+Rstress[:,3]*dvdx+\
            Rstress[:,1]*dvdy+Rstress[:,1]*dvdy+\
            Rstress[:,5]*dvdz+Rstress[:,5]*dvdz)

    p33 = -(Rstress[:,4]*dwdx+Rstress[:,4]*dwdx+\
            Rstress[:,5]*dwdy+Rstress[:,5]*dwdy+\
            Rstress[:,2]*dwdz+Rstress[:,2]*dwdz)

    p12 = -(Rstress[:,0]*dvdx+Rstress[:,3]*dudx+\
            Rstress[:,3]*dvdy+Rstress[:,2]*dudy+\
            Rstress[:,4]*dvdz+Rstress[:,5]*dudz)

    p13 = -(Rstress[:,0]*dwdx+Rstress[:,4]*dudx+\
            Rstress[:,3]*dwdy+Rstress[:,5]*dudy+\
            Rstress[:,4]*dwdz+Rstress[:,2]*dudz)

    p23 = -(Rstress[:,3]*dwdx+Rstress[:,4]*dvdx+\
            Rstress[:,2]*dwdy+Rstress[:,5]*dvdy+\
            Rstress[:,5]*dwdz+Rstress[:,2]*dvdz)

    production = np.zeros((6,nclass))
    for iclass in range(nclass):
        dwmask = np.where(dwclass == iclass+1,True,False)
        
        production[0,iclass] = np.mean(p11[dwmask])
        production[1,iclass] = np.mean(p22[dwmask])
        production[2,iclass] = np.mean(p33[dwmask])
        production[3,iclass] = np.mean(p12[dwmask])
        production[4,iclass] = np.mean(p13[dwmask])
        production[5,iclass] = np.mean(p23[dwmask])

        
    return production

def extractPressTensor(dwclass, pressprime, gradprime, dt):
    
    nclass = np.amax(dwclass)
    
    R11 = (gradprime[:,:,0]+gradprime[:,:,0])*pressprime/inp.rho
    R22 = (gradprime[:,:,4]+gradprime[:,:,4])*pressprime/inp.rho
    R33 = (gradprime[:,:,8]+gradprime[:,:,8])*pressprime/inp.rho
    R12 = (gradprime[:,:,3]+gradprime[:,:,1])*pressprime/inp.rho
    R13 = (gradprime[:,:,6]+gradprime[:,:,2])*pressprime/inp.rho
    R23 = (gradprime[:,:,7]+gradprime[:,:,5])*pressprime/inp.rho

    R11t = np.sum(R11*dt,axis=0)
    R22t = np.sum(R22*dt,axis=0)
    R33t = np.sum(R33*dt,axis=0)
    R12t = np.sum(R12*dt,axis=0)
    R13t = np.sum(R13*dt,axis=0)
    R23t = np.sum(R23*dt,axis=0)

    Dt = np.sum(dt,axis=0)

    

    pressStrainTensor = np.zeros((6,nclass))
    for iclass in range(nclass):
        dwmask = np.where(dwclass == iclass+1,True,False)
        Deltat = np.sum(Dt[dwmask])
        
        pressStrainTensor[0,iclass] = np.sum(R11t[dwmask])/Deltat
        pressStrainTensor[1,iclass] = np.sum(R22t[dwmask])/Deltat
        pressStrainTensor[2,iclass] = np.sum(R33t[dwmask])/Deltat
        pressStrainTensor[3,iclass] = np.sum(R12t[dwmask])/Deltat
        pressStrainTensor[4,iclass] = np.sum(R13t[dwmask])/Deltat
        pressStrainTensor[5,iclass] = np.sum(R23t[dwmask])/Deltat

    return pressStrainTensor

def extractGrads(plnindices,xplns,y,z,probedata,gradmean,\
                 gradpmean,pmean,Rstress):

    plns = list(range(0,xplns.shape[0],int(xplns.shape[0]/inp.nplot)))
    dw = [[] for i in range(len(plns))]
    
    dudx = [[] for i in range(len(plns))]
    dvdx = [[] for i in range(len(plns))]
    dwdx = [[] for i in range(len(plns))]

    dudy = [[] for i in range(len(plns))]
    dvdy = [[] for i in range(len(plns))]
    dwdy = [[] for i in range(len(plns))]

    dudz = [[] for i in range(len(plns))]
    dvdz = [[] for i in range(len(plns))]
    dwdz = [[] for i in range(len(plns))]

    dpdx = [[] for i in range(len(plns))]
    dpdy = [[] for i in range(len(plns))]
    dpdz = [[] for i in range(len(plns))]

    strainprime = [[] for i in range(len(plns))]
    dissipation = [[] for i in range(len(plns))]
    production = [[] for i in range(len(plns))]

    disstensor = [[] for i in range(len(plns))]
    prodtensor = [[] for i in range(len(plns))]
    pressStrainTensor = [[] for i in range(len(plns))]

    print("Planes being processed: ",plns)
    for ipln in range(len(plns)):
        plnmask = np.where(plnindices == plns[ipln],True, False)
        yplane = y[plnmask]
        zplane = z[plnmask]
        Rstresspln = Rstress[plnmask]
        
        dwclass, dw[ipln] = getdw(yplane,zplane)
        print("Synchronized Plane %i with legacy xyzts"%(plns[ipln]))

        dudx[ipln] = extractMean(dwclass,plnmask,gradmean[:,0])
        dvdx[ipln] = extractMean(dwclass,plnmask,gradmean[:,1])
        dwdx[ipln] = extractMean(dwclass,plnmask,gradmean[:,2])

        dudy[ipln] = extractMean(dwclass,plnmask,gradmean[:,3])
        dvdy[ipln] = extractMean(dwclass,plnmask,gradmean[:,4])
        dwdy[ipln] = extractMean(dwclass,plnmask,gradmean[:,5])

        dudz[ipln] = extractMean(dwclass,plnmask,gradmean[:,6])
        dvdz[ipln] = extractMean(dwclass,plnmask,gradmean[:,7])
        dwdz[ipln] = extractMean(dwclass,plnmask,gradmean[:,8])

        dpdx[ipln] = extractMean(dwclass,plnmask,gradpmean[:,0])
        dpdy[ipln] = extractMean(dwclass,plnmask,gradpmean[:,1])
        dpdz[ipln] = extractMean(dwclass,plnmask,gradpmean[:,2])

        pdw = extractMean(dwclass,plnmask,pmean)
        
        gradprime, pgradprime, pressprime, dt = extractgradfluct(\
                            probedata,plnmask,dwclass,pdw,\
                            dudx[ipln],dvdx[ipln],dwdx[ipln],\
                            dudy[ipln],dvdy[ipln],dwdy[ipln],\
                            dudz[ipln],dvdz[ipln],dwdz[ipln],\
                            dpdx[ipln],dpdy[ipln],dpdz[ipln])

        strainprime[ipln], dissipation[ipln] = extractstrain(\
                                     gradprime, dt, dwclass)
        disstensor[ipln] = extractDissipationTensor(gradprime,\
                        dt, dwclass)
        prodtensor[ipln] = extractProductionTensor(dwclass,\
                        Rstresspln,gradmean,plnmask)
        pressStrainTensor[ipln] = extractPressTensor(dwclass,\
                        pressprime, gradprime, dt)
        
    dw = np.array(dw)
    dissipation = np.array(dissipation)
    strainprime = np.array(strainprime)
    Sxx = np.array(dudx)
    Snn = np.array(dvdy)
    Stt = np.array(dwdz)
    Sxn = 0.5*(np.array(dudy)+np.array(dvdx))
    Sxt = 0.5*(np.array(dudz)+np.array(dwdx))
    Snt = 0.5*(np.array(dvdz)+np.array(dwdy))
    disstensor = np.array(disstensor)
    prodtensor = np.array(prodtensor)
    pressStrainTensor = np.array(pressStrainTensor)

    print('Creating Plots Now')
    # plotMultiPlane('dissipation','${\epsilon}(m^2/s^3)$',\
    #                xplns,plns,dw,dissipation,1)

    plotMultiPlane('Sxx','${Sxx}(s^{-1})$',\
                   xplns,plns,dw,Sxx)
    plotMultiPlane('Snn','${Snn}(s^{-1})$',\
                   xplns,plns,dw,Snn)
    plotMultiPlane('Stt','${Stt}(s^{-1})$',\
                   xplns,plns,dw,Stt)
    plotMultiPlane('Sxn','${Sxn}(s^{-1})$',\
                   xplns,plns,dw,Sxn)
    plotMultiPlane('Sxt','${Sxt}(s^{-1})$',\
                   xplns,plns,dw,Sxt)
    plotMultiPlane('Snt','${Snt}(s^{-1})$',\
                   xplns,plns,dw,Snt)
   
    plotMultiPlane('diss','${\epsilon}$',\
                   xplns,plns,dw,\
                   0.5*(disstensor[:,0,:]+disstensor[:,1,:]\
                        +disstensor[:,2,:])/(math.pow(inp.utau,4)/\
                        (inp.mu/inp.rho)),1)

    plotMultiPlane('prod','${P}$',\
                   xplns,plns,dw,\
                   0.5*(prodtensor[:,0,:]+prodtensor[:,1,:]\
                        +prodtensor[:,2,:])/(math.pow(inp.utau,4)/\
                        (inp.mu/inp.rho)),1)

    plotMultiPlane('pressStrain','${P}$',\
                   xplns,plns,dw,\
                   0.5*(pressStrainTensor[:,0,:]+pressStrainTensor[:,1,:]\
                        +pressStrainTensor[:,2,:])/(math.pow(inp.utau,4)/\
                        (inp.mu/inp.rho)),1)

    # # Read the Reynolds stressed to get the production term
    # production = np.zeros((dw.shape[0],dw.shape[1]))
    # Rstress = np.zeros((dw.shape[0],dw.shape[1],6))
    # for ipln in range(dw.shape[0]):
    #     fname = inp.cwd+'/legacyData/velExtracts_plane_'+str(plns[ipln])+'.csv'
    #     temp = np.loadtxt(fname,delimiter=',',skiprows=1)
    #     Rstress[ipln,:,:] = temp[:,4:10]
        
    # production += Rstress[:,:,0]*np.array(dudx)
    # production += Rstress[:,:,1]*np.array(dvdy)
    # production += Rstress[:,:,2]*np.array(dwdz)
    # production += Rstress[:,:,3]*np.array(dudy)
    # production += Rstress[:,:,3]*np.array(dvdx)
    # production += Rstress[:,:,4]*np.array(dudz)
    # production += Rstress[:,:,4]*np.array(dwdx)
    # production += Rstress[:,:,5]*np.array(dvdz)
    # production += Rstress[:,:,5]*np.array(dwdy)

    # production = -production

    # plotMultiPlane('production','${P}(m^2/s^3)$',\
    #                xplns,plns,dw,production,1)
    # plotMultiPlane('prod_eps','${P/\epsilon}$',\
    #                xplns,plns,dw,production/dissipation,1)

    plotMultiPlane('new_prod_eps','${P/\epsilon}$',\
                   xplns,plns,dw,\
    (0.5*(prodtensor[:,0,:]+prodtensor[:,1,:]+prodtensor[:,2,:]))/\
    (0.5*(disstensor[:,0,:]+disstensor[:,1,:]+disstensor[:,2,:])),1)


    for ipln in range(dw.shape[0]):
        fname = inp.cwd+'/legacyData/gradExtracts_plane_'+str(plns[ipln])+'.csv'
        stack = np.column_stack((dw[ipln],Sxx[ipln],Snn[ipln],Stt[ipln],\
                Sxn[ipln],Sxt[ipln],Snt[ipln],\
                disstensor[ipln,0,:],disstensor[ipln,1,:],\
                disstensor[ipln,2,:],disstensor[ipln,3,:],\
                disstensor[ipln,4,:],disstensor[ipln,5,:],\
                prodtensor[ipln,0,:],prodtensor[ipln,1,:],\
                prodtensor[ipln,2,:],prodtensor[ipln,3,:],\
                prodtensor[ipln,4,:],prodtensor[ipln,5,:],\
                pressStrainTensor[ipln,0,:],pressStrainTensor[ipln,1,:],\
                pressStrainTensor[ipln,2,:],pressStrainTensor[ipln,3,:],\
                pressStrainTensor[ipln,4,:],pressStrainTensor[ipln,5,:]))

        head = 'dwall,Sxx,Snn,Stt,Sxn,Sxt,Snt'
        head = head+',eps_xx,eps_nn,eps_tt,eps_xn,eps_xt,eps_nt'
        head = head+',P_xx,P_nn,P_tt,P_xn,P_xt,P_nt'
        head = head+',R_xx,R_nn,R_tt,R_xn,R_xt,R_nt'
        np.savetxt(fname,stack,header=head,delimiter=',')

    return

def extractgradpressfluct(probedata, plnmask, dwclass,\
                          gradpdwx,gradpdwy,gradpdwz):
    nclass = np.amax(dwclass)

    gradmp = np.zeros((dwclass.shape[0],3))
    
    for iclass in range(nclass):
        gradmp[:,0]=np.where(dwclass==iclass+1,gradpdwx[iclass],gradmp[:,0])
        gradmp[:,1]=np.where(dwclass==iclass+1,gradpdwy[iclass],gradmp[:,1])
        gradmp[:,2]=np.where(dwclass==iclass+1,gradpdwz[iclass],gradmp[:,2])

    gradpprime = np.zeros((inp.totalsteps,dwclass.shape[0],3))
    
    for istep in range(inp.totalsteps):
        for i in range(3):
            gradpplane = probedata[istep,:,i+7][plnmask]
            gradpprime[istep,:,i]=np.subtract(gradpplane,gradmp[:,i])

    return gradpprime

def extractVelPressTensor(dwclass, uprime, vprime, wprime,\
                         gradpprime, dt):

    nclass = np.amax(dwclass)

    Pi11=-(uprime*gradpprime[:,:,0]+uprime*gradpprime[:,:,0])/inp.rho
    Pi22=-(vprime*gradpprime[:,:,1]+vprime*gradpprime[:,:,1])/inp.rho
    Pi33=-(wprime*gradpprime[:,:,2]+wprime*gradpprime[:,:,2])/inp.rho
    Pi12=-(uprime*gradpprime[:,:,1]+vprime*gradpprime[:,:,0])/inp.rho
    Pi13=-(uprime*gradpprime[:,:,2]+wprime*gradpprime[:,:,0])/inp.rho
    Pi23=-(vprime*gradpprime[:,:,2]+wprime*gradpprime[:,:,1])/inp.rho

    Pi11t = np.sum(Pi11*dt,axis=0)
    Pi22t = np.sum(Pi22*dt,axis=0)
    Pi33t = np.sum(Pi33*dt,axis=0)
    Pi12t = np.sum(Pi12*dt,axis=0)
    Pi13t = np.sum(Pi13*dt,axis=0)
    Pi23t = np.sum(Pi23*dt,axis=0)

    Dt = np.sum(dt,axis=0)

    pressStrainTensor = np.zeros((6,nclass))
    for iclass in range(nclass):
        dwmask = np.where(dwclass == iclass+1,True,False)
        Deltat = np.sum(Dt[dwmask])
        
        pressStrainTensor[0,iclass] = np.sum(Pi11t[dwmask])/Deltat
        pressStrainTensor[1,iclass] = np.sum(Pi22t[dwmask])/Deltat
        pressStrainTensor[2,iclass] = np.sum(Pi33t[dwmask])/Deltat
        pressStrainTensor[3,iclass] = np.sum(Pi12t[dwmask])/Deltat
        pressStrainTensor[4,iclass] = np.sum(Pi13t[dwmask])/Deltat
        pressStrainTensor[5,iclass] = np.sum(Pi23t[dwmask])/Deltat

    return pressStrainTensor

def extractpressgradient(plnindices,xplns,y,z,\
                             umean,vmean,wmean,gradpmean,probedata):

    plns = list(range(0,xplns.shape[0],int(xplns.shape[0]/inp.nplot)))
    dw = [[] for i in range(len(plns))]
    velPressStrainTensor = [[] for i in range(len(plns))]
    
    print("Planes being processed: ",plns)
    for ipln in range(len(plns)):
        plnmask = np.where(plnindices == plns[ipln],True,False)
        yplane = y[plnmask]
        zplane = z[plnmask]
        
        dwclass, dw[ipln] = getdw(yplane, zplane)
        print("Synchronized Plane %i with legacy xyzts"%(plns[ipln]))

        udw = extractMean(dwclass,plnmask,umean)
        vdw = extractMean(dwclass,plnmask,vmean)
        wdw = extractMean(dwclass,plnmask,wmean)

        gradpdwx = extractMean(dwclass,plnmask,gradpmean[:,0])
        gradpdwy = extractMean(dwclass,plnmask,gradpmean[:,1])
        gradpdwz = extractMean(dwclass,plnmask,gradpmean[:,2])

        uprime, vprime, wprime, dt = extractfluct(probedata,\
                    plnmask,dwclass,\
                    udw,vdw,wdw)
        
        gradpprime = extractgradpressfluct(probedata,\
                    plnmask,dwclass,\
                    gradpdwx,gradpdwy,gradpdwz)

        velPressStrainTensor[ipln] = extractVelPressTensor(dwclass,\
                                    uprime, vprime, wprime,\
                                    gradpprime, dt)
    
    dw = np.array(dw)
    velPressStrainTensor = np.array(velPressStrainTensor)

    print('Creating Plots Now')

    plotMultiPlane('PI','$\Pi$',\
                   xplns,plns,dw,\
                   0.5*(velPressStrainTensor[:,0,:]+\
                        velPressStrainTensor[:,1,:]+\
                        velPressStrainTensor[:,2,:])\
                   /(math.pow(inp.utau,4)/(inp.mu/inp.rho)),1)

    for ipln in range(dw.shape[0]):
        fname=inp.cwd+'/legacyData/gradPressExtracts_plane_'+\
               str(plns[ipln])+'.csv'
        stack = np.column_stack((dw[ipln],\
                velPressStrainTensor[ipln,0,:],velPressStrainTensor[ipln,1,:],\
                velPressStrainTensor[ipln,2,:],velPressStrainTensor[ipln,3,:],\
                velPressStrainTensor[ipln,4,:],velPressStrainTensor[ipln,5,:]))

        head='dwall,Pixx,Pinn,Pitt,Pixn,Pixt,Pint'
        np.savetxt(fname,stack,header=head,delimiter=',')
    
    return
