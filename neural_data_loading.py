import scipy.io as sio
from os.path import join
import os
import platform 
# if on osx
if platform.system() == 'Darwin':
    rootdir = r"/Users/binxuwang/Library/CloudStorage/OneDrive-HarvardUniversity/SabatiniShijiaLickingClassifier/Data"
elif platform.system() == 'Linux':
    rootdir = r"/n/holylabs/LABS/kempner_fellows/Users/binxuwang/Projects/SabatiniShijiaLickingClassifier/Data"
elif platform.system() == 'Windows':
    raise NotImplementedError("Windows is not supported yet")

def load_data():
    print("Loading data from", rootdir)
    print(os.listdir(rootdir))
    neural_mat = sio.loadmat(join(rootdir, "allLicksNeural_5msBin_41Bins_20_1_20.mat"))
    lastlicksId_mat = sio.loadmat(join(rootdir, 'lastLickIdentity_5msBin_41Bins_20_1_20.mat'))
    firstboutId_mat = sio.loadmat(join(rootdir, 'firstLickBoutIdentity_5msBin_41Bins_20_1_20.mat'))
    neural_mat = neural_mat['allLicksNeural']
    lastlicksId_mat = lastlicksId_mat['lastLickIdentity']
    firstboutId_mat = firstboutId_mat['firstLickBoutIdentity']
    # print all shape
    print("Neural data", neural_mat.shape)
    print("Last lick Id", lastlicksId_mat.shape)
    print("First bout Id", firstboutId_mat.shape)
    return neural_mat, lastlicksId_mat, firstboutId_mat