"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
        Author: nsaini
        Created: 2019-07-04
Notes:
- np.memmap only partially opens the file. Use it for reading large
        files. Very efficient
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""

import math
import numpy as np
import pandas as pd
import os
import time
import legacyplots
import inp
import funcs

def main():
    os.environ["CUDA_VISIBLE_DEVICES"]="-1"
    
    probedata = np.zeros((inp.totalsteps,inp.nprobes,inp.nvar))

    print('\n<-------- Check the file contents -------->\n')
    for tstep in range(inp.inistep, inp.laststep+inp.nsteps, inp.nsteps):
        istepindex = tstep-inp.inistep
        lstepindex = istepindex+inp.nsteps
        fname =inp.cwd+'/varts/varts.'+str(tstep)+'.run.8.dat'
        probedata[istepindex:lstepindex,:,:] = funcs.readvarts(fname)

        #        if(tstep == inp.inistep):
        print("File: ",tstep," First step First probe:")
        print(probedata[istepindex,0,:])
        print("File: ",tstep," Last step Last probe:")
        print(probedata[istepindex+inp.nsteps-1,-1,:])
    print('\n<-------- All files read -------->\n')

    #    Get the coordinates of the probes
    x = np.average(probedata[:,:,inp.nvar-3], axis=0)
    y = np.average(probedata[:,:,inp.nvar-2], axis=0)
    z = np.average(probedata[:,:,inp.nvar-1], axis=0)
    
    #    Identify the number of planes
    xplns, plnindices, plncount = np.unique(x,return_inverse=True,return_counts=True)
    print('Number of identified x-planes: ',xplns.shape[0])
        
    #    Rotate the vectors to norm-tan system
    probedata = funcs.rotateVectors(probedata)
    print('Rotated All Vectors')

    #    Get the averages of variables
    #    pmean = getmean(probedata[:,:,0], probedata[:,:,4], tsum)
    
    if(inp.extract == 1):
        umean = funcs.getmean(probedata[:,:,1], probedata[:,:,0])
        vmean = funcs.getmean(probedata[:,:,2], probedata[:,:,0])
        wmean = funcs.getmean(probedata[:,:,3], probedata[:,:,0])
        print('Extracted Mean Velocities')
    
        # # Extract instantaneous velocities
        # uprime = np.subtract(probedata[:,:,1],umean)
        # vprime = np.subtract(probedata[:,:,2],vmean)
        # wprime = np.subtract(probedata[:,:,3],wmean)
        # print('Extracted Instantaneous velocities')

    if(inp.legacyPlot == 1):
        print('Legacy Plots Requested')
        legacyplots.extractPlots(plnindices,xplns,y,z,probedata,\
                                 umean,vmean,wmean)
        

if __name__ == "__main__":
    start_time = time.time()
    main()
    print("--- Code ran in %s seconds ---" % (time.time() - start_time))

