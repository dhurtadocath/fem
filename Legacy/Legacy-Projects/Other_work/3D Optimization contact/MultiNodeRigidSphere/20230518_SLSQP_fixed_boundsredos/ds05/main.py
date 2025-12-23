# This script is devoted to the generation of an ABAQUS study
# to simulate the frictionless contact of a truss structure
# with a rigid sphere.

# Importing basic IO libraries
import numpy as np

# Import Abaqus modules
from abaqus import *
from abaqusConstants import *
from caeModules import *
from driverUtils import executeOnCaeStartup
from mesh import *

# import Diego's model
nodes = np.genfromtxt('data/pos.csv', delimiter=',')
connect = np.genfromtxt('data/connect.csv', delimiter=',', dtype=int)
slave_nds = np.genfromtxt('data/ID_slaves.csv', delimiter=',', dtype=int)
bottom_nds = np.genfromtxt('data/ID_base.csv', delimiter=',', dtype=int)

# Diego's Model variables

# Beams' cross-section
A = 1.0
# Mechanical properties of beams
E = 1.0  # Young modulus
nu = 0.3  # Poisson modulus
mu = 0.5  # static friction coefficient

# Rigid sphere center and Radius
O3 = (1.0, 0.0, 3.0)
Ox, Oy, Oz = O3
Rad = 3.1

# Abaqus Instructions for model generation
executeOnCaeStartup()
myModel = mdb.Model(name='TrussModel')

# Define the material properties
myModel.Material(name='Mat1')
myModel.materials['Mat1'].Elastic(table=((E, nu),))

# Define the section properties
myModel.TrussSection(name='TrussSection', material='Mat1', area=A)

# Create the part
myModel.Part(name='Truss', dimensionality=THREE_D, type=DEFORMABLE_BODY)
myTruss = myModel.parts['Truss']

# Define the points and the lines connecting them
for i in nodes:
    myTruss.DatumPointByCoordinate(coords=(i[0], i[1], i[2]))

Connex = []
for i in connect:
    Connex.append((myTruss.datums[int(i[0] + 1)], myTruss.datums[int(i[1] + 1)]))

for pair in Connex:
    myTruss.WirePolyLine(points=(pair,))

# Assign the section properties to the part
region = (myTruss.edges,)
myTruss.SectionAssignment(region=region, sectionName='TrussSection', offset=0.0,
                         offsetType=MIDDLE_SURFACE, offsetField='', thicknessAssignment=FROM_SECTION)

# Create a Discrete Rigid Sphere
# First a sketch to revovle
s = myModel.ConstrainedSketch(name='__profile__', sheetSize=5.0)
g, v, d, c = s.geometry, s.vertices, s.dimensions, s.constraints
s.setPrimaryObject(option=STANDALONE)
s.ConstructionLine(point1=(0.0, -2.5), point2=(0.0, 2.5))
s.FixedConstraint(entity=g[2])
s.ArcByCenterEnds(center=(0.0, 0.0), point1=(Rad, 0.0), point2=(0.0, -Rad), direction=CLOCKWISE)

# now generate the sphere
myModel.Part(name='RigidSphere', dimensionality=THREE_D, type=ANALYTIC_RIGID_SURFACE)
mySphere = myModel.parts['RigidSphere']
mySphere.AnalyticRigidSurfRevolve(sketch=s)
s.unsetPrimaryObject()
del myModel.sketches['__profile__']
del mdb.models['Model-1']

# reference point for the discrete rigid sphere
RPS = mySphere.ReferencePoint(point=(0.0, 0.0, 0.0))
idxRPS = mySphere.referencePoints.keys()[0]

# Creating assembly
myAssembly = myModel.rootAssembly
myAssembly.DatumCsysByDefault(CARTESIAN)
myInstance = myAssembly.Instance(name='Truss-1', part=myTruss, dependent=ON)
myInstance2 = myAssembly.Instance(name='RigidSphere-1', part=mySphere, dependent=ON)
myAssembly.translate(instanceList=('RigidSphere-1',), vector=O3)
myAssembly.rotate(angle=90.0, axisDirection=(1.0,0.0, 0.0), axisPoint=O3, instanceList=('RigidSphere-1', ))

# Creating Step
myModel.StaticStep(name='Step-1', previous='Initial',
                   timePeriod=100.0, maxNumInc=1000, initialInc=100.0, minInc=0.001,
                   maxInc=100.0)

myModel.steps['Step-1'].setValues(initialInc=1.0, maxNumInc=
    100, noStop=OFF, timeIncrementationMethod=FIXED)

# Fixation of the rigid Sphere
RefPnt = myAssembly.Set(name='RefPnt_Sphere', referencePoints=(myInstance2.referencePoints[idxRPS], ))
myModel.EncastreBC(createStepName='Initial', localCsys=None, name='DBC_RP', region=RefPnt)

# Defining displacement history
myModel.TabularAmplitude(name='DispHistory', timeSpan=STEP,
                         smooth=SOLVER_DEFAULT, data=((0.0, 0.0), (100.0, 4.0)))

# DBC for the moving base
coords_bottom = [((i[0], i[1], i[2]),) for i in nodes[bottom_nds]]
verty = []
for i in coords_bottom:
    verty.append(myInstance.vertices.findAt(i))

region = myAssembly.Set(vertices=verty, name='BottomNodes')
myModel.DisplacementBC(name='MovingBase',
                       createStepName='Step-1', region=region, u1=1.0, u2=0.0, u3=0.0, ur1=UNSET,
                       ur2=UNSET, ur3=UNSET, amplitude='DispHistory', fixed=OFF,
                       distributionType=UNIFORM, fieldName='', localCsys=None)

# Contact interactions

# Master surface definition
surfy = myInstance2.faces.findAt(((Ox + Rad, Oy, Oz),))
master_region = myAssembly.Surface(name='MasterSurf', side1Faces=surfy)

# slave nodes definition
coords_slave = [((i[0], i[1], i[2]),) for i in nodes[slave_nds]]
verty = []
for i in coords_slave:
    verty.append(myInstance.vertices.findAt(i))
slave_region = myAssembly.Set(vertices=verty, name='SlaveNodes')

# contact definition
myModel.ContactProperty('IntProp-1')
myModel.interactionProperties['IntProp-1'].TangentialBehavior(
    dependencies=0, directionality=ISOTROPIC, elasticSlipStiffness=None,
    formulation=PENALTY, fraction=0.005, maximumElasticSlip=FRACTION,
    pressureDependency=OFF, shearStressLimit=None, slipRateDependency=OFF,
    table=((mu,),), temperatureDependency=OFF)
myModel.interactionProperties['IntProp-1'].NormalBehavior(
    allowSeparation=ON, constraintEnforcementMethod=DEFAULT,
    pressureOverclosure=HARD)
myModel.SurfaceToSurfaceContactStd(adjustMethod=NONE,
                                   clearanceRegion=None, createStepName='Initial', datumAxis=None,
                                   initialClearance=OMIT, interactionProperty='IntProp-1', master=
                                   master_region, name=
                                   'Contact1', slave=slave_region,
                                   sliding=FINITE, thickness=ON)

# Discretization

myTruss.seedPart(deviationFactor=0.1, minSizeFactor=0.1, size=1.0)
myTruss.setElementType(elemTypes=(ElemType(elemCode=T3D2, elemLibrary=STANDARD), ), regions=(
    myTruss.edges, ))
myTruss.generateMesh()

# Job definition
mdb.Job(atTime=None, contactPrint=OFF, description='', echoPrint=OFF,
    explicitPrecision=SINGLE, getMemoryFromAnalysis=True, historyPrint=OFF,
    memory=90, memoryUnits=PERCENTAGE, model='TrussModel', modelPrint=OFF,
    multiprocessingMode=DEFAULT, name='Job-Truss', nodalOutputPrecision=SINGLE,
    numCpus=1, numGPUs=0, queue=None, resultsFormat=ODB, scratch='', type=
    ANALYSIS, userSubroutine='', waitHours=0, waitMinutes=0)