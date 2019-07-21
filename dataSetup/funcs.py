"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Author: nsaini
Created: 2019-07-13
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""

import math
import numpy as np
import os
import inp

def readvarts(fname):
    probedata = np.zeros([inp.nsteps,inp.nprobes,inp.nvar])
    readata = np.zeros([inp.nsteps,inp.nprobes,28])

    with open(fname,'r') as fobj:
        for istep in range(inp.nsteps):
            pos = istep*inp.nprobes*28*8
            readata[istep] = np.memmap(fobj,dtype='>f8',\
                            mode='r',shape=(inp.nprobes,28),offset=pos)

    for istep in range(inp.nsteps):
        for iprobe in range(inp.nprobes):
            if(inp.extract == 1):
                probedata[istep,iprobe,0] = readata[istep,iprobe,4]
                probedata[istep,iprobe,1:4] = readata[istep,iprobe,1:4]
                probedata[istep,iprobe,4:7] = readata[istep,iprobe,25:28]
    return probedata



def coordRotation(y, z, v, w):
    
    tan = np.zeros(v.shape)
    norm = np.zeros(v.shape)

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

    return norm, tan


def rotateVectors(probedata):
    y = probedata[:,:,inp.nvar-2]
    z = probedata[:,:,inp.nvar-1]

    #    Velocities
    if(inp.extract == 1):
        probedata[:,:,2], probedata[:,:,3] = coordRotation(y, z,\
                    probedata[:,:,2], probedata[:,:,3])

    return probedata


def getmean(data,tarray):
    totaltime = np.sum(tarray[:,0], axis = 0)
    arrmean = np.divide(np.sum(np.multiply(data, tarray), axis=0),totaltime)
    return arrmean

