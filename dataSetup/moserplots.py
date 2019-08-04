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
import os
import time

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
    default_cycler = (cycler(color=['b','r','k','m','g'])*\
                      cycler(linestyle=['-'])*cycler(marker=['.']))

    plt.rc('lines',linewidth=1)
    plt.rc('axes',prop_cycle=default_cycler)
    
    fig = plt.figure(figsize=(7,5))
    ax = fig.add_subplot(111)
    ax.set_xlabel('$y^+$',fontsize=20)
    ax.set_ylabel(ylabel,fontsize=20)

    markers = ['','','.','']
    labels = ['$Re_{\\tau}550 (Moser)$',\
              '${Re_\\tau}1000 (Moser)$','$Re_{\\tau}683 (PHASTA)$',\
              '$Re_{\\tau}1600$ (Jun et al, 2015)']
    for i in range(len(data)):
        if(axistype == 0):
            ax.semilogx(yplus[i],data[i],label=labels[i],marker=markers[i])     
        else:
            ax.plot(yplus[i],data[i],label=labels[i],marker=markers[i])
    
    if(fname == 'TKE'):
        juny, junTKE = readJunData()
        if(axistype == 0):
            ax.semilogx(juny,junTKE,label=labels[3],\
                        marker=markers[3])
        else:
            ax.plot(juny,junTKE,label=labels[3],\
                    marker=markers[3])
    elif(fname == 'U'):
        juny, junU = readJunDataU()
        if(axistype == 0):
            ax.semilogx(juny,junU,label=labels[3],\
                        marker=markers[3])
        else:
            ax.plot(juny,junU,label=labels[3],\
                    marker=markers[3])

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
    yplus = yplus[yplus<300]
    return yplus.tolist(), U.tolist()

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
    
    return yplus.tolist(), Rxx.tolist(), Rnn.tolist(), Rtt.tolist(), TKE.tolist()

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

    return data

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
    
def main():

    # Get your extracted data
    simData = myLegacyData()
    gradData = myLegacyGradData()

    # Get mean Velocities Moser
    yplus = [[]]*3
    U = [[]]*3
    print('Reading mean velocities')
    fname = 'Re550/LM_Channel_0550_mean_prof.dat'
    yplus[0], U[0] = readMoservel(fname)
    fname = 'Re1000/LM_Channel_1000_mean_prof.dat'
    yplus[1], U[1] = readMoservel(fname)
    
    yplus[2] = simData[:,0].tolist()
    U[2] = simData[:,1].tolist()

    plotnow('U','$U^+$',U,yplus)
    plotnow('U','$U^+$',U,yplus,1)


    # Get Fluctuations Moser 
    yplus = [[]]*3
    Rxx = [[]]*3
    Rnn = [[]]*3
    Rtt = [[]]*3
    TKE = [[]]*3
    print('Reading Reynolds Stresses')
    fname = 'Re550/LM_Channel_0550_vel_fluc_prof.dat'
    yplus[0], Rxx[0], Rnn[0], Rtt[0], TKE[0] = readMoserFluct(fname)
    fname = 'Re1000/LM_Channel_1000_vel_fluc_prof.dat'
    yplus[1], Rxx[1], Rnn[1], Rtt[1], TKE[1] = readMoserFluct(fname)
    
    yplus[2] = simData[:,0].tolist()
    Rxx[2] = simData[:,4].tolist()
    Rnn[2] = simData[:,5].tolist()
    Rtt[2] = simData[:,6].tolist()
    TKE[2] = simData[:,10].tolist()
    
    plotnow('Rxx','$R_{xx}$',Rxx,yplus)
    plotnow('Rnn','$R_{nn}$',Rnn,yplus)
    plotnow('Rtt','$R_{tt}$',Rtt,yplus)
    plotnow('TKE','$TKE$',TKE,yplus)
    plotnow('Rxx','$R_{xx}$',Rxx,yplus,1)
    plotnow('Rnn','$R_{nn}$',Rnn,yplus,1)
    plotnow('Rtt','$R_{tt}$',Rtt,yplus,1)
    plotnow('TKE','$TKE$',TKE,yplus,1)

    # Get TKE Budget Moser
    yplus = [[]]*3
    production = [[]]*3
    dissipation = [[]]*3
    pstrain = [[]]*3    
    print('Reading TKE budget')
    fname = 'Re550/LM_Channel_0550_RSTE_k_prof.dat'
    yplus[0], production[0], dissipation[0], pstrain[0] = readMoserbudget(fname)
    fname = 'Re1000/LM_Channel_1000_RSTE_k_prof.dat'
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
    fname = 'Re550/LM_Channel_0550_RSTE_uu_prof.dat'
    yplus[0], production[0], dissipation[0], pstrain[0] = readMoserbudget(fname)
    fname = 'Re1000/LM_Channel_1000_RSTE_uu_prof.dat'
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
    fname = 'Re550/LM_Channel_0550_RSTE_vv_prof.dat'
    yplus[0], production[0], dissipation[0], pstrain[0] = readMoserbudget(fname)
    fname = 'Re1000/LM_Channel_1000_RSTE_vv_prof.dat'
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
    fname = 'Re550/LM_Channel_0550_RSTE_ww_prof.dat'
    yplus[0], production[0], dissipation[0], pstrain[0] = readMoserbudget(fname)
    fname = 'Re1000/LM_Channel_1000_RSTE_ww_prof.dat'
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
    fname = 'Re550/LM_Channel_0550_RSTE_uv_prof.dat'
    yplus[0], production[0], dissipation[0], pstrain[0] = readMoserbudget(fname)
    fname = 'Re1000/LM_Channel_1000_RSTE_uv_prof.dat'
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

    return

if __name__ == "__main__":
    starttime = time.time()
    main()
    print("--- Code ran in %s seconds ---"%(time.time()-starttime))





