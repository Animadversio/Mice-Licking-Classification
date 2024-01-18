import scipy.io as sio
from os.path import join
import os
import platform 
# if on osx
if platform.system() == 'Darwin':
    rootdir = r"/Users/binxuwang/Library/CloudStorage/OneDrive-HarvardUniversity/SabatiniShijiaLickingClassifier"
    dataroot = join(rootdir, "Data")
elif platform.system() == 'Linux':
    rootdir = r"/n/holylabs/LABS/kempner_fellows/Users/binxuwang/Projects/SabatiniShijiaLickingClassifier"
    dataroot = join(rootdir, "Data")
elif platform.system() == 'Windows':
    raise NotImplementedError("Windows is not supported yet")

def load_data(datadir):
    print("Loading data from", datadir)
    print(os.listdir(datadir))
    neural_mat = sio.loadmat(join(datadir, "allLicksNeural_5msBin_41Bins_20_1_20.mat"))
    lastlicksId_mat = sio.loadmat(join(datadir, 'lastLickIdentity_5msBin_41Bins_20_1_20.mat'))
    firstboutId_mat = sio.loadmat(join(datadir, 'firstLickBoutIdentity_5msBin_41Bins_20_1_20.mat'))
    neural_mat = neural_mat['allLicksNeural']
    lastlicksId_mat = lastlicksId_mat['lastLickIdentity']
    firstboutId_mat = firstboutId_mat['firstLickBoutIdentity']
    # print all shape
    print("Neural data", neural_mat.shape)
    print("Last lick Id", lastlicksId_mat.shape)
    print("First bout Id", firstboutId_mat.shape)
    return neural_mat, lastlicksId_mat, firstboutId_mat