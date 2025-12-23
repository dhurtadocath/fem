import imageio
from os import listdir, walk
from os.path import isfile, join
from pdb import set_trace


mypath = "~/fem/Leonint/ContactPotato cTNone 100stp kn5maxGN0.005/"

# set_trace()

images = []
formats = ['png', 'jpeg']


filelist = list(set(range(1,148))-{18,19,30,31,36,39,40,49,51,53,54,55,56,57,59,64,65,66,71,74,75,76,77,78,79,84,85,87,88,89,93,94,98,101,102,109,110,112,115,116,117,126,130,131,132,144,147})
# filelist = [1,2,3,4,5,6,8,9,10,11,32,50,51,52,53,54,55,56,57,58,59,60,62,63,65,66,68,69,70,71,72,73,75,105,106,107,108,109,110,111,112,113,114,
#             140,147,148,166,167,168,169,170,171,172,181,213,217,218,219,220]

# for num in range(100):
for num in filelist:
    f= "fig"+str(num)+".png"
    images.append(imageio.imread(mypath+f))
# images = [imageio.imread(mypath+f) for f in listdir(mypath) if f[-3:] in formats]

imageio.mimsave(mypath+"cropped"+'movie.gif', images, duration = 0.025,)