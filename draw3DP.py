import matplotlib.pyplot as plt
import numpy as np

def get_cmap(n, name='jet'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

def savePointCloudColorSeg(points, label, num, filename):
    f = open(filename, "w")

    f.write("ply\n")
    f.write("format ascii 1.0\n")
    f.write("element vertex 2048\n")
    f.write("property float x\n")
    f.write("property float y\n")
    f.write("property float z\n")
    f.write("property uchar red\n")
    f.write("property uchar green\n")
    f.write("property uchar blue\n")
    f.write("end_header\n")
    
    cmap = get_cmap(num)
    for i in range(np.max(label)+1):
        index = np.where(label==i)
        x = points[index, 0]
        y = points[index, 1]
        z = points[index, 2]
        p_num = x.shape[1]

        for j in range(p_num):
            f.write(str(x[0,j])+" "+str(y[0,j])+" "+str(z[0,j])+" "+str(int(255*cmap(i)[0]))+" "+str(int(255*cmap(i)[1]))+" "+str(int(255*cmap(i)[2]))+"\n")
    
    f.close()