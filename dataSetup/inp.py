"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Author: nsaini
Created: 2019-07-13
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""

import os
import math

def checkline(f):
    for l in f:
        line = l.rstrip()
        if line:
            li = line.strip()
            if not li.startswith('#'):
                yield line

inp = {}
with open("inputs.txt") as f:
    for line in checkline(f):
        (key, val) = line.split()
        inp[str(key)] = val


cwd = os.getcwd()
print(cwd)

inistep = int(inp.get('inistep',0))
laststep = int(inp.get('laststep',100))
nprobes = int(inp.get('nprobes',1))
nPHASTAprobes = int(inp.get('nPHASTAprobes',1))
nrun = int(inp.get('nrun',1))
nsteps = int(inp.get('nsteps',100))

pitch = float(inp.get('pitch',0.0063))
rrod = float(inp.get('rrod',0.00457))
Dh = float(inp.get('Dh',0.013))

Re = float(inp.get('Re',1000))
Um = float(inp.get('Um',1))
rho = float(inp.get('rho',1))
mu = float(inp.get('mu',1.0E-6))

plttype = inp.get('plttype','1d')
extract = int(inp.get('extract',1))
nhom = int(inp.get('nhom',80))
legacyPlot = int(inp.get('legacyPlot',0))
nplot = int(inp.get('nplot',1))
rotate = int(inp.get('Rotate',1))

npl=0
plnindices = []

print(rotate)

tempstr = "--------> "

print(tempstr+"Initial File = ",inistep)
print(tempstr+"Last File = ",laststep)
print(tempstr+"Number of probes = ",nprobes)
print(tempstr+"Number of time steps per file = ",nsteps)
print("")
print(tempstr+"Pitch of rods = ",pitch)
print(tempstr+"Radius of rods = ",rrod)
print("")
print(tempstr+"Reynolds number = ",Re)
print(tempstr+"Mean Streamwise Velocity = ",Um)
print(tempstr+"Density = ",rho)
print(tempstr+"Viscosity = ",mu)
print("")
if(legacyPlot == 1):
    if(extract == 1):
        print(tempstr+"Extracting Velocities, TKE and Stresses")
    if(plttype == '1d'):
        print(tempstr+'Making 1d plane plots')
    elif(plttype == '3d'):
        print(tempstr+'Making 3d plane plots')
    print(tempstr+"Number of homogeneous probes = ",nhom)
    print(tempstr+"Index of plane to plot = ",nplot)


# Some preliminary calculations
f = 0.316*math.pow(Re,-0.25)
tauw = rho*Um*Um*f/8
utau = math.sqrt(tauw/rho)

#    Velocities (TKE, Mean stresses, Reynolds Stresses)
if(extract == 1):
    nvar = 3
elif(extract == 2):
    nvar = 13
elif(extract == 3):
    nvar = 6
#    Coordinates and time
nvar += 4
print(tempstr+"Number of variables = ",nvar)

totalsteps = laststep-inistep+nsteps

# Create output dir
try:
    os.mkdir(cwd+'/output')
    print('out dir created')
except FileExistsError:
    print('out dir exists')

if(legacyPlot == 1):
    try:
        os.mkdir(cwd+'/legacyData')
        os.mkdir(cwd+'/legacyData/plots')
        print('legacy dir created')
    except FileExistsError:
        print('legacy dir exists')
