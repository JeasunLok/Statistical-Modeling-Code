import scipy.io as sio

def Load_data(data_path):
    data = sio.loadmat(data_path)
    C = data["CC"]
    R = data["R"]
    MKT = data["ft"]
    return C, R, MKT

def Load_data_split(data_path):
    data = sio.loadmat(data_path)
    Ctrain = data["Ctrn"]
    Ctest = data["Ctst"]
    Rtrain = data["Rtrn"]
    Rtest = data["Rtst"]
    fttrain = data["fttrn"]
    fttest = data["fttst"]
    return Ctrain, Ctest, Rtrain, Rtest, fttrain, fttest