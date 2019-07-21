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
        fname =inp.cwd+'/newvarts/varts.'+str(tstep)+'.run.1.dat'
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
    dwall = np.sqrt(np.power(np.abs(y)-inp.pitch,2) + \
                    np.power(np.abs(z)-inp.pitch,2))-inp.rrod
    fname = inp.cwd+'/output/probeCoord.csv'
    np.savetxt(fname,np.column_stack((x,y,z,dwall)),\
               delimiter=',',header='x,y,z,dwall')
    print('Saved probe coordinates')
    
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
        fname = inp.cwd+'/output/velMean.csv'
        np.savetxt(fname,np.column_stack((umean,vmean,wmean,)),\
                   delimiter=',',header='<u>,<v>,<w>')
        print('Saved Mean Velocities')

        if(inp.legacyPlot == 1):
            print('Legacy Plots Requested')
            legacyplots.extractPlots(plnindices,xplns,y,z,probedata,\
                                     umean,vmean,wmean)

        uprime = probedata[:,:,1] - umean
        vprime = probedata[:,:,2] - vmean
        wprime = probedata[:,:,3] - wmean
        print('Calculated fluctuating component of velocity')
        
        Rstress = np.zeros((inp.nprobes,6))
        Rstress[:,0] = funcs.getmean(uprime*uprime,probedata[:,:,0])
        Rstress[:,1] = funcs.getmean(vprime*vprime,probedata[:,:,0])
        Rstress[:,2] = funcs.getmean(wprime*wprime,probedata[:,:,0])
        Rstress[:,3] = funcs.getmean(uprime*vprime,probedata[:,:,0])
        Rstress[:,4] = funcs.getmean(uprime*wprime,probedata[:,:,0])
        Rstress[:,5] = funcs.getmean(vprime*wprime,probedata[:,:,0])
        print('Extracted Reynolds Stresses')

        fname = inp.cwd+'/output/RStress.csv'
        np.savetxt(fname,np.column_stack((Rstress[:,0],Rstress[:,1],\
        Rstress[:,2],Rstress[:,3],\
        Rstress[:,4],Rstress[:,5])),delimiter=',',\
        header='<u\'u\'>,<v\'v\'>,<w\'w\'>,<u\'v\'>,<u\'w\'>,<v\'w\'>')
        print('Saved Reynolds Stresses')

    return
        
        
if __name__ == "__main__":
    start_time = time.time()
    main()
    print("--- Code ran in %s seconds ---" % (time.time() - start_time))

