import numpy as np

def homo_transform(mat, points):
    if mat.shape[1] != points.shape[1]:
        points = np.concatenate([points, np.ones((points.shape[0],1))], axis=1)
    homo = points @ mat.T
    result = (homo[:,:homo.shape[1]-1].T / homo[:,-1]).T
    return result
