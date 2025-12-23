import imageio
from os import listdir, walk
from os.path import isfile, join
from pdb import set_trace


folder = "mu0.5"
mypath = "~/fem/3D Optimization contact/Simple_1node_Diagrams/SLSQP-con_230628-0841/plots/"

# set_trace()

images = []
formats = ['png', 'jpeg']


# filelist = list(set(range(1,47))-{0,8,9,10,11,12,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44})
filelist = [1,2,3,4,5,6,8,9,10,11,32,50,51,52,53,54,55,56,57,58,59,60,62,63,65,66,68,69,70,71,72,73,75,105,106,107,108,109,110,111,112,113,114,
            140,147,148,166,167,168,169,170,171,172,181,213,217,218,219,220]

for num in range(1,100):
# for num in filelist:
    f= "Minimization"+str(num)+".png"
    images.append(imageio.imread(mypath+f))
# images = [imageio.imread(mypath+f) for f in listdir(mypath) if f[-3:] in formats]

imageio.mimsave(mypath+"cropped"+'movie.gif', images, duration = 0.05,)