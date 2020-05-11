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
    
    probedata = []

    print('\n<-------- Check the file contents -------->\n')
    for tstep in range(inp.inistep, inp.laststep+inp.nsteps, inp.nsteps):
        istepindex = tstep-inp.inistep
        lstepindex = istepindex+inp.nsteps
        if(inp.rotate == 0):
            fname =inp.cwd+'/newvarts/varts.'+str(tstep)\
                    +'.run.'+str(inp.nrun)+'.dat'
        elif(inp.rotate == 1):
            fname =inp.cwd+'/rotatedvarts/varts.'+str(tstep)\
                    +'.run.'+str(inp.nrun)+'.dat'

        probedata.append(funcs.readvarts(fname))

        #        if(tstep == inp.inistep):
        print("File: ",tstep," First step First probe:")
        #print(probedata[istepindex,0,:])
        print("File: ",tstep," Last step Last probe:")
        #print(probedata[istepindex+inp.nsteps-1,-1,:])
    print('\n<-------- All files read -------->\n')

    probedata = np.asarray(probedata)
    probedata = np.reshape(probedata,(len(probedata)*inp.nsteps,inp.npl,20))
    
    dir2 = 'pln_'+str(inp.nplot)
    try:
        os.mkdir(inp.cwd+'/output/'+dir2)
        print('pln dir created')
    except FileExistsError:
        print('pln dir exists')

    #    Get the coordinates of the probes
    x = probedata[0,:,1]
    y = probedata[0,:,2]
    z = probedata[0,:,3]
    dwall = np.sqrt(np.power(np.abs(y)-inp.pitch,2) + \
                    np.power(np.abs(z)-inp.pitch,2))-inp.rrod
    fname = inp.cwd+'/output/'+dir2+'/probeCoord.csv'
    np.savetxt(fname,np.column_stack((x,y,z,dwall)),\
               delimiter=',',header='x,y,z,dwall')
    print('Saved probe coordinates')
    
    #    Rotate the vectors to norm-tan system
    if(inp.rotate == 1):
        #funcs.rotateVectors(probedata)
        print('Vectors were rotated earlier')
    else:
        print('Vectors not being rotated. Only done for legacy plots')
        
    
    
    umean = np.zeros((inp.npl))
    vmean = np.zeros((inp.npl))
    wmean = np.zeros((inp.npl))

    funcs.getmean(probedata[:,:,4], probedata[:,:,0],umean)
    funcs.getmean(probedata[:,:,5], probedata[:,:,0],vmean)
    funcs.getmean(probedata[:,:,6], probedata[:,:,0],wmean)
    print('Extracted Mean Velocities')
        
    if(inp.rotate == 1):
        fname = inp.cwd+'/output/'+dir2+'/velMean_tn.csv'
        np.savetxt(fname,np.column_stack((umean,vmean,wmean,)),\
                   delimiter=',',header='<u>,<ut>,<un>')
        print('Saved Mean Velocities')
            
        
        uprime = probedata[:,:,4] - umean
        vprime = probedata[:,:,5] - vmean
        wprime = probedata[:,:,6] - wmean
        print('Calculated fluctuating component of velocity')
        
        Rstress = np.zeros((inp.npl,6))
        funcs.getmean(uprime*uprime,probedata[:,:,0],Rstress[:,0])
        print('Calculated R0')
        funcs.getmean(vprime*vprime,probedata[:,:,0],Rstress[:,1])
        print('Calculated R1')
        funcs.getmean(wprime*wprime,probedata[:,:,0],Rstress[:,2])
        print('Calculated R2')
        funcs.getmean(uprime*vprime,probedata[:,:,0],Rstress[:,3])
        print('Calculated R3')
        funcs.getmean(uprime*wprime,probedata[:,:,0],Rstress[:,4])
        print('Calculated R4')
        funcs.getmean(vprime*wprime,probedata[:,:,0],Rstress[:,5])
        print('Extracted Reynolds Stresses')
        
        fname = inp.cwd+'/output/RStress_tn.csv'
        head = '<u\'u\'>,<ut\'ut\'>,<un\'un\'>,'
        head = head+'<u\'ut\'>,<u\'un\'>,<ut\'un\'>'
        np.savetxt(fname,np.column_stack((Rstress[:,0],Rstress[:,1],\
                                          Rstress[:,2],Rstress[:,3],\
                                          Rstress[:,4],Rstress[:,5])),delimiter=',',\
                   header=head)
        print('Saved Reynolds Stresses')

        print('Legacy Plots Requested')
        #legacyplots.extractPlots(plnindices,xplns,y,z,probedata,\
            #                         umean,vmean,wmean)
    else:
        fname = inp.cwd+'/output/'+dir2+'/velMean.csv'
        np.savetxt(fname,np.column_stack((umean,vmean,wmean,)),\
                delimiter=',',header='<u>,<v>,<w>')
        print('Saved Mean Velocities')
    
        
        uprime = probedata[:,:,4] - umean
        vprime = probedata[:,:,5] - vmean
        wprime = probedata[:,:,6] - wmean
        print('Calculated fluctuating component of velocity')
        
        Rstress = np.zeros((inp.npl,6))
        funcs.getmean(uprime*uprime,probedata[:,:,0],Rstress[:,0])
        print('Calculated R0')
        funcs.getmean(vprime*vprime,probedata[:,:,0],Rstress[:,1])
        print('Calculated R1')
        funcs.getmean(wprime*wprime,probedata[:,:,0],Rstress[:,2])
        print('Calculated R2')
        funcs.getmean(uprime*vprime,probedata[:,:,0],Rstress[:,3])
        print('Calculated R3')
        funcs.getmean(uprime*wprime,probedata[:,:,0],Rstress[:,4])
        print('Calculated R4')
        funcs.getmean(vprime*wprime,probedata[:,:,0],Rstress[:,5])
        print('Extracted Reynolds Stresses')
        
        fname = inp.cwd+'/output/'+dir2+'/RStress.csv'
        head = '<u\'u\'>,<v\'v\'>,<w\'w\'>,<u\'v\'>,<u\'w\'>,<v\'w\'>'
        np.savetxt(fname,np.column_stack((Rstress[:,0],Rstress[:,1],\
                                          Rstress[:,2],Rstress[:,3],\
                                          Rstress[:,4],Rstress[:,5])),delimiter=',',\
                   header=head)
        print('Saved Reynolds Stresses')
            
        '''
        bij = np.zeros((inp.npl,6))
        TKE = funcs.getmean(0.5*(uprime*uprime+vprime*vprime+\
        wprime*wprime),probedata[:,:,0])
        for i in range(6):
        bij[:,i] = Rstress[:,i]/(2.0*TKE)
        
        bij[:,0:3] = bij[:,0:3] - 1./3.
        
        print('Extracted Normalized Anisotropy Tensor')
        
        fname = inp.cwd+'/output/anisotropy.csv'
        head = 'b11,b22,b33,b12,b13,b23'
        np.savetxt(fname,np.column_stack((bij[:,0],bij[:,1],\
        bij[:,2],bij[:,3],\
        bij[:,4],bij[:,5])),delimiter=',',\
        header=head)
        print('Saved Anisotropy Tensor')
        
        eta = np.zeros(inp.npl)
        xi = np.zeros(inp.npl)
        
        for i in range(inp.npl):
        mat = np.matrix([[bij[i,0],bij[i,3],bij[i,4]],\
        [bij[i,3],bij[i,1],bij[i,5]],\
        [bij[i,4],bij[i,5],bij[i,2]]])
        mat2 = np.matmul(mat,mat)
        trmat = mat[0,0]+mat[1,1]+mat[2,2]
        trmat2 = mat2[0,0]+mat2[1,1]+mat2[2,2]
        II = 0.5*(trmat**2. - trmat2)
        III = np.linalg.det(mat)
        eta[i] = math.sqrt((-1./3.)*II)
        xi[i] = funcs.cubic_root((1./2.)*III)
        
        fname = inp.cwd+'/output/lumley.csv'
        head = 'eta,xi'
        np.savetxt(fname,np.column_stack((eta,xi)),delimiter=',',header=head)
        print('Saved Invariants')
            '''
        
        
        # Read the already extracted Reynolds Stresses
    if (inp.rotate == 1):
        Rstress = np.zeros((inp.npl,6))
        fname = inp.cwd+'/output/RStress_tn.csv'
        Rstress = np.loadtxt(fname,delimiter=',',skiprows=1)
        
        pmean = funcs.getmean(probedata[:,:,16], probedata[:,:,0])
        pprime = probedata[:,:,16] - pmean
        
        gradmean = np.zeros((inp.npl,9))
        for i in range(9):
            gradmean[:,i] = funcs.getmean(probedata[:,:,i+4],\
                                          probedata[:,:,0])
        print('Extracted Mean of Velocity Gradients')
        gradpmean = np.zeros((inp.npl,3))
        for i in range(3):
            gradpmean[:,i] = funcs.getmean(probedata[:,:,i+13],\
                                                   probedata[:,:,0])
        print('Extracted Mean of Pressure Gradients')
            
        print('Extracting Data in Legacy Format')
        legacyplots.extractGrads(plnindices,xplns,y,z,probedata,\
                                     gradmean,gradpmean,pmean,Rstress)
    elif(inp.rotate == 0):        
        pmean = np.zeros((inp.npl))
        funcs.getmean(probedata[:,:,19], probedata[:,:,0],pmean)
        pprime = probedata[:,:,19] - pmean
        
        gradmean = np.zeros((inp.npl,9))
        for i in range(9):
            funcs.getmean(probedata[:,:,i+7],\
                          probedata[:,:,0],gradmean[:,i])
        print('Extracted Mean of Velocity Gradients')
        gradpmean = np.zeros((inp.npl,3))
        for i in range(3):
            funcs.getmean(probedata[:,:,i+16],\
                          probedata[:,:,0],gradpmean[:,i])
        print('Extracted Mean of Pressure Gradients')
        
        gradprime = np.zeros((inp.totalsteps,inp.npl,9))
        for i in range(9):
            gradprime[:,:,i] = probedata[:,:,i+7] - gradmean[:,i]
        print('Calculated fluctuating velocity gradient components')
            
        gradpprime = np.zeros((inp.totalsteps,inp.npl,3))
        for i in range(3):
            gradpprime[:,:,i] = probedata[:,:,i+16] - gradpmean[:,i]
        print('Calculated fluctuating pressure gradient components')
            

        #Extract Strain rate (fluctuating) Pg. 125 Eq. 5.130
        strainrate = np.zeros((inp.npl,6))
        funcs.getmean(0.5*(gradprime[:,:,0]+gradprime[:,:,0]),\
                      probedata[:,:,0],strainrate[:,0])
        funcs.getmean(0.5*(gradprime[:,:,4]+gradprime[:,:,4]),\
                      probedata[:,:,0],strainrate[:,1])
        funcs.getmean(0.5*(gradprime[:,:,8]+gradprime[:,:,8]),\
                      probedata[:,:,0],strainrate[:,2])
        funcs.getmean(0.5*(gradprime[:,:,3]+gradprime[:,:,1]),\
                      probedata[:,:,0],strainrate[:,3])
        funcs.getmean(0.5*(gradprime[:,:,6]+gradprime[:,:,2]),\
                      probedata[:,:,0],strainrate[:,4])
        funcs.getmean(0.5*(gradprime[:,:,7]+gradprime[:,:,5]),\
                      probedata[:,:,0],strainrate[:,5])
        print("Extracted fluctuating strain rate tensor")

            
        #Extract dissipation tensor Pg. 315
        #Missing 2nu factor
        dissipation = np.zeros((inp.npl,6))
        funcs.getmean(gradprime[:,:,0]*gradprime[:,:,0]+\
                      gradprime[:,:,3]*gradprime[:,:,3]+\
                      gradprime[:,:,6]*gradprime[:,:,6],\
                      probedata[:,:,0],dissipation[:,0])
        funcs.getmean(gradprime[:,:,1]*gradprime[:,:,1]+\
                      gradprime[:,:,4]*gradprime[:,:,4]+\
                      gradprime[:,:,7]*gradprime[:,:,7],\
                      probedata[:,:,0],dissipation[:,1])
        funcs.getmean(gradprime[:,:,2]*gradprime[:,:,2]+\
                      gradprime[:,:,5]*gradprime[:,:,5]+\
                      gradprime[:,:,8]*gradprime[:,:,8],\
                      probedata[:,:,0],dissipation[:,2])
        funcs.getmean(gradprime[:,:,0]*gradprime[:,:,1]+\
                      gradprime[:,:,3]*gradprime[:,:,4]+\
                      gradprime[:,:,6]*gradprime[:,:,7],\
                      probedata[:,:,0],dissipation[:,3])
        funcs.getmean(gradprime[:,:,0]*gradprime[:,:,2]+\
                      gradprime[:,:,3]*gradprime[:,:,5]+\
                      gradprime[:,:,6]*gradprime[:,:,8],\
                      probedata[:,:,0],dissipation[:,4])
        funcs.getmean(gradprime[:,:,1]*gradprime[:,:,2]+\
                      gradprime[:,:,4]*gradprime[:,:,5]+\
                      gradprime[:,:,7]*gradprime[:,:,8],\
                      probedata[:,:,0],dissipation[:,5])
        print("Extracted dissipation tensor")

        #Extract Production tensor
        production = np.zeros((inp.npl,6))
        production[:,0] = -(Rstress[:,0]*gradmean[:,0]+\
                            Rstress[:,0]*gradmean[:,0]+\
                            Rstress[:,3]*gradmean[:,3]+\
                            Rstress[:,3]*gradmean[:,3]+\
                            Rstress[:,4]*gradmean[:,6]+\
                            Rstress[:,4]*gradmean[:,6])
        production[:,1] = -(Rstress[:,3]*gradmean[:,1]+\
                            Rstress[:,3]*gradmean[:,1]+\
                            Rstress[:,1]*gradmean[:,4]+\
                            Rstress[:,1]*gradmean[:,4]+\
                            Rstress[:,5]*gradmean[:,7]+\
                            Rstress[:,5]*gradmean[:,7])
        production[:,2] = -(Rstress[:,4]*gradmean[:,2]+\
                            Rstress[:,4]*gradmean[:,2]+\
                            Rstress[:,5]*gradmean[:,5]+\
                            Rstress[:,5]*gradmean[:,5]+\
                            Rstress[:,2]*gradmean[:,8]+\
                            Rstress[:,2]*gradmean[:,8])
        production[:,3] = -(Rstress[:,0]*gradmean[:,1]+\
                            Rstress[:,3]*gradmean[:,0]+\
                            Rstress[:,3]*gradmean[:,4]+\
                            Rstress[:,2]*gradmean[:,3]+\
                            Rstress[:,4]*gradmean[:,7]+\
                            Rstress[:,5]*gradmean[:,6])
        production[:,4] = -(Rstress[:,0]*gradmean[:,2]+\
                            Rstress[:,4]*gradmean[:,0]+\
                            Rstress[:,3]*gradmean[:,5]+\
                            Rstress[:,5]*gradmean[:,3]+\
                            Rstress[:,4]*gradmean[:,8]+\
                            Rstress[:,2]*gradmean[:,6])
        production[:,5] = -(Rstress[:,3]*gradmean[:,2]+\
                            Rstress[:,4]*gradmean[:,1]+\
                            Rstress[:,2]*gradmean[:,5]+\
                            Rstress[:,5]*gradmean[:,4]+\
                            Rstress[:,5]*gradmean[:,8]+\
                            Rstress[:,2]*gradmean[:,7])
        
        #Extract pressure strain rate tensor Pg 317
        #missing 1/rho
        R = np.zeros((inp.npl,6))
        funcs.getmean((gradprime[:,:,0]+gradprime[:,:,0])*pprime,\
                      probedata[:,:,0],R[:,0])
        funcs.getmean((gradprime[:,:,4]+gradprime[:,:,4])*pprime,\
                      probedata[:,:,0],R[:,1])
        funcs.getmean((gradprime[:,:,8]+gradprime[:,:,8])*pprime,\
                      probedata[:,:,0],R[:,2])
        funcs.getmean((gradprime[:,:,3]+gradprime[:,:,1])*pprime,\
                      probedata[:,:,0],R[:,3])
        funcs.getmean((gradprime[:,:,6]+gradprime[:,:,2])*pprime,\
                      probedata[:,:,0],R[:,4])
        funcs.getmean((gradprime[:,:,7]+gradprime[:,:,5])*pprime,\
                          probedata[:,:,0],R[:,5])

        
        fname = inp.cwd+'/output/'+dir2+'/strainrate.csv'
        head ='s11,s22,s33,s12,s13,s23'
        np.savetxt(fname,np.column_stack((strainrate[:,0],strainrate[:,1],\
                                          strainrate[:,2],strainrate[:,3],\
                                          strainrate[:,4],strainrate[:,5])),\
                   delimiter=',',header=head)
        print('Saved Strain Rate Tensor')
    
        fname = inp.cwd+'/output/'+dir2+'/dissipation.csv'
        head ='d11,d22,d33,d12,d13,d23'
        np.savetxt(fname,np.column_stack((dissipation[:,0],dissipation[:,1],\
                                          dissipation[:,2],dissipation[:,3],\
                                        dissipation[:,4],dissipation[:,5])),\
                   delimiter=',',header=head)
        print('Saved Dissipation Tensor')
            
        fname = inp.cwd+'/output/'+dir2+'/production.csv'
        head ='p11,p22,p33,p12,p13,p23'
        np.savetxt(fname,np.column_stack((production[:,0],production[:,1],\
                                          production[:,2],production[:,3],\
                                          production[:,4],production[:,5])),\
                   delimiter=',',header=head)
        print('Saved Production Tensor')
        
        fname = inp.cwd+'/output/'+dir2+'/presstrainrate.csv'
        head ='R11,R22,R33,R12,R13,R23'
        np.savetxt(fname,np.column_stack((R[:,0],R[:,1],\
                                          R[:,2],R[:,3],\
                                          R[:,4],R[:,5])),\
                   delimiter=',',header=head)
        print('Saved Pressure Rate of Strain Tensor')

    return
        
        
if __name__ == "__main__":
    start_time = time.time()
    main()
    print("--- Code ran in %s seconds ---" % (time.time() - start_time))

