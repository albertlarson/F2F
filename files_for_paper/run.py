from trapdoor.f2f import *

basin_idxs = [0,1,2,3]
lags = [0,1,2,3,4,5,6,7,14]
data_split = [0.65,0.7,0.75,0.8,.85,0.9,.95]

def zZz(x,y):
    return str(x).zfill(y)

config = []
idx = 0
for i in basin_idxs:
    for j in lags:
        for k in data_split:
            config.append((idx,i,j,k))
            idx+=1
                     
iters = list(range(9))
import time

for i in iters:
    t0 = time.time()
    os.mkdir(f'/work/albertl_uri_edu/fluxtoflow/files_for_paper/zultz/runpng/{i}')
    os.mkdir(f'/work/albertl_uri_edu/fluxtoflow/files_for_paper/zultz/runtxt/{i}')
    macroscale_results = []
    for idx,x in enumerate(config):
        rsq,kge,nse = f2f(x[0],x[1],x[2],x[3],
                          f'/work/albertl_uri_edu/fluxtoflow/files_for_paper/zultz/runtxt/{i}/{zZz(x[0],3)}_{x[1]}_{zZz(x[2],2)}_{str(int(x[3]*100))}',
                          f'/work/albertl_uri_edu/fluxtoflow/files_for_paper/zultz/runpng/{i}/{zZz(x[0],3)}_{x[1]}_{zZz(x[2],2)}_{str(int(x[3]*100))}')
        macroscale_results.append((idx,rsq,kge,nse))
    macroscale_results = np.asarray(macroscale_results)
    np.save(f'/work/albertl_uri_edu/fluxtoflow/files_for_paper/zultz/runrsqkgense/macroscale_results_{i}.npy',macroscale_results)
    t1 = time.time()
    print(f'iter: {i}/{iters} \ntime: {round((t1-t0)/60,3)}')