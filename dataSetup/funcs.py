"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Author: nsaini
Created: 2019-07-13
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""

import math
import numpy as np
import os
import inp
import numpy.testing as npt

def pow_with_nan(x,y):
    try:
        return math.pow(x,y)
    except ValueError:
        #return float('nan')
        return 0.0

def cubic_root(x):
    return math.copysign(math.pow(abs(x),1./3.),x)

def createplnmask(data):
    x = data[:,25]
    xplns, plnindices, plncount = np.unique(x,return_inverse=True,\
                                            return_counts=True)
    print('Number of identified x-planes: ',xplns.shape[0])

    inp.plnindices.append(plnindices)
    
    inp.npl = plncount[inp.nplot]
    print('Number of probes on plane: ',inp.npl)

    return

def readvarts(fname):
    
    readata = np.zeros([inp.nsteps,inp.nprobes,28])

    with open(fname,'r') as fobj:
        for istep in range(inp.nsteps):
            pos = istep*inp.nprobes*28*8
            readata[istep] = np.memmap(fobj,dtype='>f8',\
                            mode='r',shape=(inp.nprobes,28),offset=pos)
    if(inp.npl ==0):
        createplnmask(readata[0,:,:])
    probedata = np.zeros([inp.nsteps,inp.npl,20])

    for istep in range(inp.nsteps):
        ip=0
        for iprobe in range(inp.nprobes):
            if(inp.plnindices[0][iprobe] == inp.nplot):
                probedata[istep,ip,0] = readata[istep,iprobe,4]
                probedata[istep,ip,1:4] = readata[istep,iprobe,25:28]
                probedata[istep,ip,4:7] = readata[istep,iprobe,1:4]
                
                probedata[istep,ip,7:16]=readata[istep,iprobe,6:15]
                probedata[istep,ip,16:19]=readata[istep,iprobe,15:18]
                probedata[istep,ip,19] = readata[istep,iprobe,0]
                ip += 1
                # elif(inp.extract == 3):
                #     probedata[istep,ip,0]=readata[istep,iprobe,4]
                #     probedata[istep,ip,1:4] = readata[istep,iprobe,25:28]
                #     probedata[istep,ip,4:7] = readata[istep,iprobe,1:4]
                #     probedata[istep,ip,7:10]=readata[istep,iprobe,16:19]
    return probedata



def coordRotation(y, z, v, w):
    
    tan = np.zeros(v.shape[1])
    norm = np.zeros(v.shape[1])

    """
    magorig = np.zeros(v.shape)
    magnew = np.zeros(v.shape)

    
    for i in range(y.shape[1]):
        if(y[0,i] > 0 and z[0,i] > 0):
            delt = inp.pitch - abs(z[0,i])
            quad = 1
        elif(y[0,i] > 0 and z[0,i] < 0):
            delt = inp.pitch - abs(y[0,i])
            quad = 2
        elif(y[0,i] < 0 and z[0,i] > 0):
            delt = inp.pitch - abs(y[0,i])
            quad = 3
        elif(y[0,i] < 0 and z[0,i] < 0):
            delt = inp.pitch - abs(z[0,i])
            quad = 4

        mag = math.sqrt(math.pow(inp.pitch-abs(y[0,i]),2)+\
                        math.pow(inp.pitch-abs(z[0,i]),2))
        theta = math.acos(delt/mag)

        if(quad == 1):
            norm[:,i] = -v[:,i]*math.sin(theta)-w[:,i]*math.cos(theta)
            tan[:,i] = v[:,i]*math.cos(theta)-w[:,i]*math.sin(theta)
        elif(quad == 2):
            norm[:,i] = -v[:,i]*math.sin(theta)+w[:,i]*math.cos(theta)
            tan[:,i] = -v[:,i]*math.cos(theta)-w[:,i]*math.sin(theta)
        elif(quad == 3):
            norm[:,i] = v[:,i]*math.sin(theta)-w[:,i]*math.cos(theta)
            tan[:,i] = v[:,i]*math.cos(theta)+w[:,i]*math.sin(theta)
        elif(quad == 4):
            norm[:,i] = v[:,i]*math.sin(theta)+w[:,i]*math.cos(theta)
            tan[:,i] = -v[:,i]*math.cos(theta)+w[:,i]*math.sin(theta)
        
        magorig[:,i] = np.sqrt(np.power(v[:,i],2)+np.power(w[:,i],2))
        magnew[:,i] = np.sqrt(np.power(norm[:,i],2)+np.power(tan[:,i],2))
    """
    #magorig = np.sqrt(np.power(v,2)+np.power(w,2))
    # Linear transformation
    ynew = y[0,:] - np.sign(y[0,:])*(inp.pitch/2.0)
    znew = z[0,:] - np.sign(z[0,:])*(inp.pitch/2.0)

    # Angle
    theta = np.arctan2(znew,ynew)

    print('Vector shape',v.shape[0])
    # Rotation
    for i in range(v.shape[0]):
        norm = v[i,:]*np.cos(theta) - w[i,:]*np.sin(theta)
        tan = v[i,:]*np.sin(theta) + w[i,:]*np.cos(theta)
        v[i,:] = norm[:]
        w[i,:] = tan[:]
        if(i%100 == 0):
            print('Rotated',i)

    #magnew = np.sqrt(np.power(norm,2)+np.power(tan,2))
    
    #npt.assert_almost_equal(magorig,magnew,decimal=9)

    return


def rotateVectors(probedata):
    y = probedata[:,:,2]
    z = probedata[:,:,3]

    #    Velocities
    if(inp.extract == 1):
        coordRotation(y, z,\
                      probedata[:,:,5], probedata[:,:,6])
    elif(inp.extract == 2):
        probedata[:,:,7], probedata[:,:,10] = coordRotation(y, z,\
                                probedata[:,:,7], probedata[:,:,10])
        probedata[:,:,8], probedata[:,:,11] = coordRotation(y, z,\
                                probedata[:,:,8], probedata[:,:,11])
        probedata[:,:,9], probedata[:,:,12] = coordRotation(y, z,\
                                probedata[:,:,9], probedata[:,:,12])
        
        probedata[:,:,14], probedata[:,:,15] = coordRotation(y, z,\
                                probedata[:,:,14], probedata[:,:,15])

    elif(inp.extract == 3):
        probedata[:,:,5], probedata[:,:,6] = coordRotation(y, z,\
                                probedata[:,:,5], probedata[:,:,6])

        probedata[:,:,8], probedata[:,:,9] = coordRotation(y, z,\
                                probedata[:,:,8], probedata[:,:,9])


    print('Exiting Rotate Vectors')
    return


def getmean(data,tarray,mean):
    totaltime = np.sum(tarray[:,0], axis = 0)
    mean[:] = np.divide(np.sum(np.multiply(data, tarray), axis=0),totaltime)
    return

