"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Author: nsaini
Created: 2019-07-20
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""

import math
import numpy as np
import os
import inp
import time
import struct

def readvarts(tstep):
    fname =inp.cwd+'/varts/varts.'+str(tstep)+'.run.'+str(inp.nrun)+'.dat'
    probedata = np.zeros([inp.nsteps,inp.nPHASTAprobes,9])

    with open(fname,'r') as fobj:
        probedata = np.memmap(fobj,dtype='>f8',mode='r',\
                        shape=(inp.nsteps,inp.nPHASTAprobes,9),offset=0)
    return probedata



def writemodvarts(probedata,dflag,tstep):
    fname = inp.cwd+'/newvarts/varts.'+str(tstep)+'.run.1.dat'
    with open(fname,'wb') as f:
        probedata[:,dflag==0,:].astype('>f8').tofile(f)
    return
    

def main():
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    try:
        os.mkdir(inp.cwd+'/newvarts')
        print('newvarts dir created')
    except FileExistsError:
        print('newvarts dir exists')

    probedata = np.zeros((inp.nsteps,inp.nPHASTAprobes,9))
    
    print('\n<-------- Reading the first file to determine duplicate probes -------->\n')
    tstep = inp.inistep    
    probedata = readvarts(tstep)

    #        if(tstep == inp.inistep):
    print("File: ",tstep," First step First probe:")
    print(probedata[0,0,:])
    print("File: ",tstep," Last step Last probe:")
    print(probedata[inp.nsteps-1,-1,:])

    x = np.average(probedata[:,:,inp.nvar-3], axis=0)
    y = np.average(probedata[:,:,inp.nvar-2], axis=0)
    z = np.average(probedata[:,:,inp.nvar-1], axis=0)
    
    dlist = [[]]*x.shape[0]
    dupindex = [[]]*x.shape[0]
    for i in range(x.shape[0]):
        ix = x[i]
        iy = y[i]
        iz = z[i]
        dist = np.sqrt(np.power(ix-x,2)\
                       +np.power(iy-y,2)+np.power(iz-z,2))
        dist = dist.tolist()
        distindex = [j for j,x in enumerate(dist) if j!=i]
        dist = [x for j,x in enumerate(dist) if j!=i]
        dist = np.array(dist)
        distindex = np.array(distindex)
        
        dupindex[i] = np.where(dist < 1e-10,distindex,-1)
        dupindex[i] = dupindex[i][dupindex[i] >= 0]
        
        dmask = np.where(dist < 1e-10, True, False)
        dist = dist[dmask]
        dlist[i] = dist.shape[0]
        print("Processed probe ",i+1)
    dlist = np.array(dlist)
    dups = np.where(dlist != 0)

    # Ensure that there is only 1 duplicate
    for dup in dups[0]:
        if(len(dupindex[dup]) > 1):
            print('Probe %d has %d duplicates'%(dup,len(dupindex[dup])))
        #print(dup,dupindex[dup])

    # Mark probes to be deleted
    dflag = np.zeros(x.shape[0],dtype=int)
    for i in range(x.shape[0]):
        if(np.any(dups[0] == i)):
            dflag[i] += 1
    for dup in dups[0]:
        if(dflag[dup] == 1):
            dflag[dupindex[dup][0]] = 0
        
    temp, dcount = np.unique(dflag,return_counts=True)
    print('Number of new probes: ',dcount[0])
    print('Number of deleted probes: ',dcount[1])
    print('Number of total probes was: ',x.shape[0])
    
    print('Writing modified varts file, timestep: ',tstep)
    writemodvarts(probedata,dflag,tstep)

    # Now write the remaining time steps
    for tstep in range(inp.inistep+inp.nsteps,\
                       inp.laststep+inp.nsteps,inp.nsteps):
        probedata = readvarts(tstep)
        
        print('Writing modified varts file, timestep: ',tstep)
        writemodvarts(probedata,dflag,tstep)

    print('\n<-------- All files written -------->\n')

    return

if __name__ == "__main__":
    start_time = time.time()
    main()
    print("--- Code ran in %s seconds ---" % (time.time() - start_time))
