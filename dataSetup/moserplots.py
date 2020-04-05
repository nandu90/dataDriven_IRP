"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Author: nsaini
Created: 2019-08-01
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""

from cycler import cycler
import math
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
import time

def cubic_root(x):
    return math.copysign(math.pow(abs(x),1./3.),x)

def pow_with_nan(x,y):
    try:
        return math.pow(x,y)
    except ValueError:
        #return float('nan')
        return 0.0

def readJunData():
    fname = 'Jun2015/JunTKE.csv'
    data = np.loadtxt(fname,delimiter=',')
    
    utau = 0.015
    yplus = data[:,0]
    TKE = data[:,1]
    TKE /= math.pow(utau,2)

    TKE = TKE[yplus<300]
    yplus = yplus[yplus<300]
    
    return yplus, TKE

def readJunDataU():
    fname = 'Jun2015/U.csv'
    data = np.loadtxt(fname,delimiter=',')
    
    utau = 0.015
    yplus = data[:,0]
    umean = data[:,1]

    umean = umean[yplus<300]
    yplus = yplus[yplus<300]
    
    return yplus, umean    

def plotnow(fname,ylabel,data,yplus,axistype=0):
    default_cycler = (cycler(color=['b','r','k','m']))

    name = fname
    plt.rc('lines',linewidth=1)
    plt.rc('axes',prop_cycle=default_cycler)
    
    if(fname != 'anisotropy'):
        fig = plt.figure(figsize=(7,5))
    else:
        fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(111)
    if(name != 'anisotropy'):
        ax.set_xlabel('$y^+$',fontsize=20)
    else:
        ax.set_xlabel('$\\xi$',fontsize=20)
    ax.set_ylabel(ylabel,fontsize=20)

    markers = ['','','.','']
    labels = ['$Re_{\\tau}180$ (Moser)',\
              '$Re_{\\tau}550$ (Moser)',\
              '$Re_{\\tau}230$ (PHASTA)',\
              '$Re_{\\tau}400$ (Fang)']

    
    for i in range(len(data)):
        if(axistype == 0):
            ax.semilogx(yplus[i],data[i],label=labels[i],marker=markers[i])     
        else:
            ax.plot(yplus[i],data[i],label=labels[i],marker=markers[i])
    '''
    if(fname == 'TKE'):
        juny, junTKE = readJunData()
        if(axistype == 0):
            ax.semilogx(juny,junTKE,label=labels[-1],\
                        marker=markers[-1])
        else:
            ax.plot(juny,junTKE,label=labels[-1],\
                    marker=markers[-1])
    elif(fname == 'U'):
        juny, junU = readJunDataU()
        if(axistype == 0):
            ax.semilogx(juny,junU,label=labels[-1],\
                        marker=markers[-1])
        else:
            ax.plot(juny,junU,label=labels[-1],\
                    marker=markers[-1])
    '''
    if(name == 'anisotropy'):
        x = np.linspace(0,1./3.,num=100)
        y = x
        ax.plot(x,y,linestyle='--',marker='',color='k',linewidth=0.5)

        # x = np.linspace(-1./6.,0.0,num=100)
        # y = -x
        # ax.plot(x,y,linestyle='--',marker='',color='k',linewidth=0.5)

        x = np.linspace(0.0,1./3.,num=100)
        y = np.sqrt(1./27. + 2.*x**3.)
        ax.plot(x,y,linestyle='--',marker='',color='k',linewidth=0.5)
        ax.xaxis.set_ticks([0.,1./6.,1./3.])
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.3f'))

        ax.yaxis.set_ticks([0.,1./6.,1./3.])
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.3f'))

    ax.grid()
    ax.legend(loc='best',fontsize=15)

    if(axistype == 0):
        fig.savefig(fname+'_log.png',quality=100,\
                    bbox_inches='tight',dpi=500)
    else:
        fig.savefig(fname+'.png',quality=100,\
                    bbox_inches='tight',dpi=500)

    plt.close()
    return



def readMoservel(fname):
    data = np.loadtxt(fname,comments='%')
    ybydelta = data[:,0]
    yplus = data[:,1]
    U = data[:,2]
    dUdy = data[:,3]
    W = data[:,4]
    P = data[:,5]

    U = U[yplus<300]
    W = W[yplus<300]
    yplus = yplus[yplus<300]

    U = U[yplus>0.1]
    W = W[yplus>0.1]
    yplus = yplus[yplus>0.1]
    return yplus.tolist(), U.tolist(), W.tolist()

def readMoserFluct(fname):
    data =np.loadtxt(fname,comments='%')
    yplus = data[:,1]
    Rxx = data[:,2]
    Rnn = data[:,3]
    Rtt = data[:,4]
    Rxn = data[:,5]
    Rxt = data[:,6]
    Rnt = data[:,7]
    TKE = data[:,8]

    Rxx = Rxx[yplus<300]
    Rnn = Rnn[yplus<300]
    Rtt = Rtt[yplus<300]
    Rxn = Rxn[yplus<300]
    Rxt = Rxt[yplus<300]
    Rnt = Rnt[yplus<300]
    TKE = TKE[yplus<300]
    yplus = yplus[yplus<300]

    Rxx = Rxx[yplus>0.1]
    Rnn = Rnn[yplus>0.1]
    Rtt = Rtt[yplus>0.1]
    Rxn = Rxn[yplus>0.1]
    Rxt = Rxt[yplus>0.1]
    Rnt = Rnt[yplus>0.1]
    TKE = TKE[yplus>0.1]
    yplus = yplus[yplus>0.1]
    
    b11 = Rxx/(TKE*2.0) -1./3.
    b22 = Rnn/(TKE*2.0) -1./3.
    b33 = Rtt/(TKE*2.0) -1./3.
    b12 = Rxn/(TKE*2.0)
    b13 = Rxt/(TKE*2.0)
    b23 = Rnt/(TKE*2.0)

    
    eta = np.zeros(yplus.shape)
    xi = np.zeros(yplus.shape)
    for i in range(yplus.shape[0]):
        mat = np.matrix([[b11[i],b12[i],b13[i]],\
                         [b12[i],b22[i],b23[i]],\
                         [b13[i],b23[i],b33[i]]])
        eig = np.linalg.eigvals(mat)
        eta[i] = math.sqrt((eig[0]**2.+eig[0]*eig[1]+eig[1]**2.)/3.)
        xi[i] = cubic_root(-(eig[0]*eig[1]*(eig[0]+eig[1]))/2.)

    R = np.zeros((6,Rxx.shape[0]))
    R[0,:] = Rxx
    R[1,:] = Rnn
    R[2,:] = Rtt
    R[3,:] = Rxn
    R[4,:] = Rxt
    R[5,:] = Rnt
    return yplus.tolist(), TKE.tolist(), eta.tolist(), xi.tolist(), R


def readMoserbudget(fname):
    data =np.loadtxt(fname,comments='%')
    yplus = data[:,1]
    production = data[:,2]
    dissipation = data[:,7]
    pstrain = data[:,5]

    production = production[yplus<300]
    dissipation = dissipation[yplus<300]
    pstrain = pstrain[yplus<300]
    yplus = yplus[yplus<300]

    production = production[yplus>0.1]
    dissipation = dissipation[yplus>0.1]
    pstrain = pstrain[yplus>0.1]
    yplus = yplus[yplus>0.1]

    return yplus.tolist(), production.tolist(), dissipation.tolist(), pstrain.tolist()

def myLegacyData():
    fname = '/home/nsaini3/CASES/summer2019/singlephasethermal/Re11000/legacyData/velExtracts_plane_0.csv'
    data = np.loadtxt(fname,comments='#',delimiter=',')

    Re = 11000.0
    Um = 15.772
    rho = 0.914482
    mu = 17.0141E-6
    
    f = 0.316*math.pow(Re,-0.25)
    tauw = rho*Um*Um*f/8
    utau = math.sqrt(tauw/rho)

    data[:,0] = data[:,0]*rho*utau/mu
    data[:,1:4] = data[:,1:4]/utau
    data[:,4:11] = data[:,4:11]/math.pow(utau,2)
    
    b11 = data[:,4]/(data[:,10]*2.0) -1./3.
    b22 = data[:,5]/(data[:,10]*2.0) -1./3.
    b33 = data[:,6]/(data[:,10]*2.0) -1./3.
    b12 = data[:,7]/(data[:,10]*2.0)
    b13 = data[:,8]/(data[:,10]*2.0)
    b23 = data[:,9]/(data[:,10]*2.0)

    
    eta = np.zeros(data[:,0].shape[0])
    xi = np.zeros(data[:,0].shape[0])
    for i in range(data[:,0].shape[0]):
        mat = np.matrix([[b11[i],b12[i],b13[i]],\
                         [b12[i],b22[i],b23[i]],\
                         [b13[i],b23[i],b33[i]]])
        eig = np.linalg.eigvals(mat)
        eta[i] = math.sqrt((eig[0]**2.+eig[0]*eig[1]+eig[1]**2.)/3.)
        xi[i] = cubic_root(-(eig[0]*eig[1]*(eig[0]+eig[1]))/2.)

    zeromask = np.where(xi != 0.0, True, False)
    eta = eta[zeromask]
    xi = xi[zeromask]
    return data, eta, xi

def myLegacyGradData():
    fname = '/home/nsaini3/CASES/summer2019/singlephasethermal/Re11000/legacyData/gradExtracts_plane_0.csv'
    data = np.loadtxt(fname,comments='#',delimiter=',')

    Re = 11000.0
    Um = 15.772
    rho = 0.914482
    mu = 17.0141E-6
    nu = mu/rho

    f = 0.316*math.pow(Re,-0.25)
    tauw = rho*Um*Um*f/8
    utau = math.sqrt(tauw/rho)

    data[:,0] = data[:,0]*rho*utau/mu
    data[:,7:26] = data[:,7:26]/(math.pow(utau,4)/nu)

    return data

def myLegacyPiData():
    fname = '/home/nsaini3/CASES/summer2019/singlephasethermal/Re11000/legacyData/gradPressExtracts_plane_0.csv'
    data = np.loadtxt(fname,comments='#',delimiter=',')

    Re = 11000.0
    Um = 15.772
    rho = 0.914482
    mu = 17.0141E-6
    nu = mu/rho

    f = 0.316*math.pow(Re,-0.25)
    tauw = rho*Um*Um*f/8
    utau = math.sqrt(tauw/rho)

    data[:,0] = data[:,0]*rho*utau/mu
    data[:,1:7] = data[:,1:7]/(math.pow(utau,4)/nu)
    
    return data
    
def main():
    
    # Get your extracted data
    simData, myeta, myxi = myLegacyData()
    #gradData = myLegacyGradData()

    # Get mean Velocities Moser
    moser = 2+1
    yplus = [[]]*moser
    U = [[]]*moser
    W = [[]]*moser
    V = [[]]
    print('Reading mean velocities')
    fname = 'Re180/LM_Channel_0180_mean_prof.dat'
    yplus[0], U[0], W[0] = readMoservel(fname)
    fname = 'Re550/LM_Channel_0550_mean_prof.dat'
    yplus[1], U[1], W[1] = readMoservel(fname)
    
    yplus[-1] = simData[:,0].tolist()
    U[-1] = simData[:,1].tolist()
    W[-1] = simData[:,3].tolist()
    V[-1] = simData[:,2].tolist()

    
    plotnow('U','$U^+$',U,yplus)
    plotnow('U','$U^+$',U,yplus,1)
    plotnow('W','$W^+$',W,yplus)
    plotnow('W','$W^+$',W,yplus,1)

    yp = [yplus[-1]]
    plotnow('V','$V^+$',V,yp)
    plotnow('V','$V^+$',V,yp,1)
    

    # Get Fluctuations Moser 
    yplus = [[]]*moser
    Rxx = [[]]*moser
    Rnn = [[]]*moser
    Rtt = [[]]*moser
    TKE = [[]]*moser
    eta = [[]]*moser
    xi = [[]]*moser
    Rxn = [[]]*moser
    Rxt = [[]]*moser
    Rnt = [[]]*moser

    print('Reading Reynolds Stresses')
    fname = 'Re180/LM_Channel_0180_vel_fluc_prof.dat'
    yplus[0], TKE[0], eta[0], xi[0], R = readMoserFluct(fname)
    Rxx[0] = R[0,:].tolist()
    Rnn[0] = R[1,:].tolist()
    Rtt[0] = R[2,:].tolist()
    Rxn[0] = R[3,:].tolist()
    Rxt[0] = R[4,:].tolist()
    Rnt[0] = R[5,:].tolist()

    fname = 'Re550/LM_Channel_0550_vel_fluc_prof.dat'
    yplus[1], TKE[1], eta[1], xi[1], R = readMoserFluct(fname)
    Rxx[1] = R[0,:].tolist()
    Rnn[1] = R[1,:].tolist()
    Rtt[1] = R[2,:].tolist()
    Rxn[1] = R[3,:].tolist()
    Rxt[1] = R[4,:].tolist()
    Rnt[1] = R[5,:].tolist()
    
    yplus[-1] = simData[:,0].tolist()
    Rxx[-1] = simData[:,4].tolist()
    Rnn[-1] = simData[:,5].tolist()
    Rtt[-1] = simData[:,6].tolist()
    TKE[-1] = simData[:,10].tolist()
    eta[-1] = myeta.tolist()
    xi[-1] = myxi.tolist()

    Rxn[-1] = simData[:,7].tolist()
    Rxt[-1] = simData[:,8].tolist()
    Rnt[-1] = simData[:,9].tolist()

    plotnow('Rxx','$R_{xx}$',Rxx,yplus)
    plotnow('Rnn','$R_{nn}$',Rnn,yplus)
    plotnow('Rtt','$R_{tt}$',Rtt,yplus)
    plotnow('Rxn','$R_{xn}$',Rxn,yplus)
    plotnow('Rxt','$R_{xt}$',Rxt,yplus)
    plotnow('Rnt','$R_{nt}$',Rnt,yplus)
    plotnow('TKE','$TKE$',TKE,yplus)
    plotnow('Rxx','$R_{xx}$',Rxx,yplus,1)
    plotnow('Rnn','$R_{nn}$',Rnn,yplus,1)
    plotnow('Rtt','$R_{tt}$',Rtt,yplus,1)
    plotnow('Rxn','$R_{xn}$',Rxn,yplus,1)
    plotnow('Rxt','$R_{xt}$',Rxt,yplus,1)
    plotnow('Rnt','$R_{nt}$',Rnt,yplus,1)
    plotnow('TKE','$TKE$',TKE,yplus,1)

    plotnow('anisotropy','$\eta$',eta,xi,1)
    
    # Get TKE Budget Moser
    '''
    yplus = [[]]*3
    production = [[]]*3
    dissipation = [[]]*3
    pstrain = [[]]*3    
    print('Reading TKE budget')
    fname = 'Re180/LM_Channel_0180_RSTE_k_prof.dat'
    yplus[0], production[0], dissipation[0], pstrain[0] = readMoserbudget(fname)
    fname = 'Re550/LM_Channel_0550_RSTE_k_prof.dat'
    yplus[1], production[1], dissipation[1], pstrain[1] = readMoserbudget(fname)

    yplus[2] = gradData[:,0].tolist()
    dissipation[2] = (0.5*(gradData[:,7]+gradData[:,8]+gradData[:,9])).tolist()
    production[2] = (0.5*(gradData[:,13]+gradData[:,14]+gradData[:,15])).tolist()
    pstrain[2] = (0.5*(gradData[:,19]+gradData[:,20]+gradData[:,21])).tolist()

    plotnow('production_TKE','$P_{TKE}$',production,yplus)
    plotnow('dissipation_TKE','$\epsilon_{TKE}$',dissipation,yplus)
    plotnow('pressStrain_TKE','$R_{TKE}$',pstrain,yplus)
    plotnow('production_TKE','$P_{TKE}$',production,yplus,1)
    plotnow('dissipation_TKE','$\epsilon_{TKE}$',dissipation,yplus,1)
    plotnow('pressStrain_TKE','$R_{TKE}$',pstrain,yplus,1)

    # Get xx Budget Moser
    yplus = [[]]*3
    production = [[]]*3
    dissipation = [[]]*3
    pstrain = [[]]*3 
    print('Reading xx budget')
    fname = 'Re180/LM_Channel_0180_RSTE_uu_prof.dat'
    yplus[0], production[0], dissipation[0], pstrain[0] = readMoserbudget(fname)
    fname = 'Re550/LM_Channel_0550_RSTE_uu_prof.dat'
    yplus[1], production[1], dissipation[1], pstrain[1] = readMoserbudget(fname)
    
    yplus[2] = gradData[:,0].tolist()
    dissipation[2] = (gradData[:,7]).tolist()
    production[2] = (gradData[:,13]).tolist()
    pstrain[2] = (gradData[:,19]).tolist()

    plotnow('production_xx','$P_{xx}$',production,yplus)
    plotnow('dissipation_xx','$\epsilon_{xx}$',dissipation,yplus)
    plotnow('pressStrain_xx','$R_{xx}$',pstrain,yplus)
    plotnow('production_xx','$P_{xx}$',production,yplus,1)
    plotnow('dissipation_xx','$\epsilon_{xx}$',dissipation,yplus,1)
    plotnow('pressStrain_xx','$R_{xx}$',pstrain,yplus,1)

    # Get nn Budget Moser
    yplus = [[]]*3
    production = [[]]*3
    dissipation = [[]]*3
    pstrain = [[]]*3 
    print('Reading nn budget')
    fname = 'Re180/LM_Channel_0180_RSTE_vv_prof.dat'
    yplus[0], production[0], dissipation[0], pstrain[0] = readMoserbudget(fname)
    fname = 'Re550/LM_Channel_0550_RSTE_vv_prof.dat'
    yplus[1], production[1], dissipation[1], pstrain[1] = readMoserbudget(fname)
    
    yplus[2] = gradData[:,0].tolist()
    dissipation[2] = (gradData[:,8]).tolist()
    production[2] = (gradData[:,14]).tolist()
    pstrain[2] = (gradData[:,20]).tolist()

    plotnow('production_nn','$P_{nn}$',production,yplus)
    plotnow('dissipation_nn','$\epsilon_{nn}$',dissipation,yplus)
    plotnow('pressStrain_nn','$R_{nn}$',pstrain,yplus)
    plotnow('production_nn','$P_{nn}$',production,yplus,1)
    plotnow('dissipation_nn','$\epsilon_{nn}$',dissipation,yplus,1)
    plotnow('pressStrain_nn','$R_{nn}$',pstrain,yplus,1)

    # Get tt Budget Moser
    yplus = [[]]*3
    production = [[]]*3
    dissipation = [[]]*3
    pstrain = [[]]*3 
    print('Reading tt budget')
    fname = 'Re180/LM_Channel_0180_RSTE_ww_prof.dat'
    yplus[0], production[0], dissipation[0], pstrain[0] = readMoserbudget(fname)
    fname = 'Re550/LM_Channel_0550_RSTE_ww_prof.dat'
    yplus[1], production[1], dissipation[1], pstrain[1] = readMoserbudget(fname)
    
    yplus[2] = gradData[:,0].tolist()
    dissipation[2] = (gradData[:,9]).tolist()
    production[2] = (gradData[:,15]).tolist()
    pstrain[2] = (gradData[:,21]).tolist()

    plotnow('production_tt','$P_{tt}$',production,yplus)
    plotnow('dissipation_tt','$\epsilon_{tt}$',dissipation,yplus)
    plotnow('pressStrain_tt','$R_{tt}$',pstrain,yplus)
    plotnow('production_tt','$P_{tt}$',production,yplus,1)
    plotnow('dissipation_tt','$\epsilon_{tt}$',dissipation,yplus,1)
    plotnow('pressStrain_tt','$R_{tt}$',pstrain,yplus,1)

    # Get un Budget Moser
    yplus = [[]]*3
    production = [[]]*3
    dissipation = [[]]*3
    pstrain = [[]]*3 
    print('Reading un budget')
    fname = 'Re180/LM_Channel_0180_RSTE_uv_prof.dat'
    yplus[0], production[0], dissipation[0], pstrain[0] = readMoserbudget(fname)
    fname = 'Re550/LM_Channel_0550_RSTE_uv_prof.dat'
    yplus[1], production[1], dissipation[1], pstrain[1] = readMoserbudget(fname)
    
    yplus[2] = gradData[:,0].tolist()
    dissipation[2] = (gradData[:,10]).tolist()
    production[2] = (gradData[:,16]).tolist()
    pstrain[2] = (gradData[:,22]).tolist()

    plotnow('production_uv','$P_{un}$',production,yplus)
    plotnow('dissipation_uv','$\epsilon_{un}$',dissipation,yplus)
    plotnow('pressStrain_uv','$R_{un}$',pstrain,yplus)
    plotnow('production_uv','$P_{un}$',production,yplus,1)
    plotnow('dissipation_uv','$\epsilon_{un}$',dissipation,yplus,1)
    plotnow('pressStrain_uv','$R_{un}$',pstrain,yplus,1)
    '''

    '''
    # Get the velocity-pressure-gradient tensor
    Pidata = myLegacyPiData()
    yplus = [[]]*1
    Pitrace = [[]]*1
    print('Plotting Velocity pressure gradient tensor')
    
    yplus[0] = Pidata[:,0].tolist()
    Pitrace[0] = (0.5*(Pidata[:,1]+Pidata[:,2]+Pidata[:,3])).tolist()
    plotnow('Pitrace','$\Pi$',Pitrace,yplus)
    plotnow('Pitrace','$\Pi$',Pitrace,yplus,1)

    yplus = [[]]*1
    Pixx = [[]]*1    
    yplus[0] = Pidata[:,0].tolist()
    Pixx[0] = (Pidata[:,1]).tolist()
    plotnow('Pitrace_xx','$\Pi_{xx}$',Pixx,yplus)
    plotnow('Pitrace_xx','$\Pi_{xx}$',Pixx,yplus,1)

    yplus = [[]]*1
    Pinn = [[]]*1    
    yplus[0] = Pidata[:,0].tolist()
    Pinn[0] = (Pidata[:,2]).tolist()
    plotnow('Pitrace_nn','$\Pi_{nn}$',Pinn,yplus)
    plotnow('Pitrace_nn','$\Pi_{nn}$',Pinn,yplus,1)

    yplus = [[]]*1
    Pitt = [[]]*1    
    yplus[0] = Pidata[:,0].tolist()
    Pitt[0] = (Pidata[:,3]).tolist()
    plotnow('Pitrace_tt','$\Pi_{tt}$',Pitt,yplus)
    plotnow('Pitrace_tt','$\Pi_{tt}$',Pitt,yplus,1)
    
    yplus = [[]]*1
    Pixn = [[]]*1    
    yplus[0] = Pidata[:,0].tolist()
    Pixn[0] = (Pidata[:,4]).tolist()
    plotnow('Pitrace_xn','$\Pi_{xn}$',Pixn,yplus)
    plotnow('Pitrace_xn','$\Pi_{xn}$',Pixn,yplus,1)
    '''
    return

if __name__ == "__main__":
    starttime = time.time()
    main()
    print("--- Code ran in %s seconds ---"%(time.time()-starttime))





