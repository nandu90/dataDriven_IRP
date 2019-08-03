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
    x = np.average(probedata[:,:,1], axis=0)
    y = np.average(probedata[:,:,2], axis=0)
    z = np.average(probedata[:,:,3], axis=0)
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
    if(inp.legacyPlot == 1):
        probedata = funcs.rotateVectors(probedata)
        print('Rotated All Vectors')
    else:
        print('Vectors not being rotated. Only done for legacy plots')
        
    
    if(inp.extract == 1):
        umean = funcs.getmean(probedata[:,:,4], probedata[:,:,0])
        vmean = funcs.getmean(probedata[:,:,5], probedata[:,:,0])
        wmean = funcs.getmean(probedata[:,:,6], probedata[:,:,0])
        print('Extracted Mean Velocities')
        fname = inp.cwd+'/output/velMean.csv'
        np.savetxt(fname,np.column_stack((umean,vmean,wmean,)),\
                   delimiter=',',header='<u>,<v>,<w>')
        print('Saved Mean Velocities')

        if(inp.legacyPlot == 1):
            print('Legacy Plots Requested')
            legacyplots.extractPlots(plnindices,xplns,y,z,probedata,\
                                     umean,vmean,wmean)
        
        uprime = probedata[:,:,4] - umean
        vprime = probedata[:,:,5] - vmean
        wprime = probedata[:,:,6] - wmean
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
        head = '<u\'u\'>,<v\'v\'>,<w\'w\'>,<u\'v\'>,<u\'w\'>,<v\'w\'>'
        np.savetxt(fname,np.column_stack((Rstress[:,0],Rstress[:,1],\
                        Rstress[:,2],Rstress[:,3],\
                        Rstress[:,4],Rstress[:,5])),delimiter=',',\
                header=head)
        print('Saved Reynolds Stresses')

    elif(inp.extract == 2):
        # Read the already extracted Reynolds Stresses
        Rstress = np.zeros((inp.nprobes,6))
        fname = inp.cwd+'/output/RStress.csv'
        Rstress = np.loadtxt(fname,delimiter=',',skiprows=1)
                
        pmean = funcs.getmean(probedata[:,:,16], probedata[:,:,0])

        gradmean = np.zeros((inp.nprobes,9))
        for i in range(9):
            gradmean[:,i] = funcs.getmean(probedata[:,:,i+4],\
                                          probedata[:,:,0])
        print('Extracted Mean of Velocity Gradients')
        gradpmean = np.zeros((inp.nprobes,3))
        for i in range(3):
            gradpmean[:,i] = funcs.getmean(probedata[:,:,i+13],\
                                           probedata[:,:,0])
        print('Extracted Mean of Pressure Gradients')

        if(inp.legacyPlot == 1):
            print('Extracting Data in Legacy Format')
            legacyplots.extractGrads(plnindices,xplns,y,z,probedata,\
                                     gradmean,gradpmean,pmean,Rstress)
    return
        
        
if __name__ == "__main__":
    start_time = time.time()
    main()
    print("--- Code ran in %s seconds ---" % (time.time() - start_time))

