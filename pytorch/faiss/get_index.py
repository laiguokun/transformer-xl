import faiss
import time
import math
import numpy as np

def get_phi(xb): 
    return (xb ** 2).sum(1).max()

def augment_xb(xb, phi=None): 
    norms = (xb ** 2).sum(1)
    if phi is None: 
        phi = norms.max()
    extracol = np.sqrt(phi - norms)
    return np.hstack((xb, extracol.reshape(-1, 1)))

def augment_xq(xq): 
    extracol = np.zeros(len(xq), dtype='float32')
    return np.hstack((xq, extracol.reshape(-1, 1)))


ngpu = 8
gpu_resources = []
tempmem = -1

for i in range(ngpu):
    res = faiss.StandardGpuResources()
    if tempmem >= 0:
        res.setTempMemory(tempmem)
    gpu_resources.append(res)

def make_vres_vdev(i0=0, i1=-1):
    " return vectors of device ids and resources useful for gpu_multiple"
    vres = faiss.GpuResourcesVector()
    vdev = faiss.IntVector()
    if i1 == -1:
        i1 = ngpu
    for i in range(i0, i1):
        vdev.push_back(i)
        vres.push_back(gpu_resources[i])
    return vres, vdev


xb = np.load('/home/laiguokun/ssd/tmp/hidden_cache.npy').astype('float32')
print('begin augment')
xb = augment_xb(xb)
nq = 1024
nb, d = xb.shape
print(nb, d)
co = faiss.GpuMultipleClonerOptions()
co.useFloat16 = True
co.useFloat16CoarseQuantizer = False
co.usePrecomputed = False
co.indicesOptions = 0
co.verbose = True
co.shard = True

res = faiss.StandardGpuResources()
start_time = time.time()
centroid_num = int(math.sqrt(nb) * 10)
print(centroid_num)
index = faiss.index_factory(d, "OPQ64_256,IVF{},PQ64".format(centroid_num))
vres, vdev = make_vres_vdev()
index = faiss.index_cpu_to_gpu_multiple(vres, vdev, index, co)
index.train(xb)
index.add(xb)
cpu_index = faiss.index_gpu_to_cpu(index)
faiss.write_index(cpu_index, '/home/laiguokun/ssd/tmp/wt103_dot.index')

