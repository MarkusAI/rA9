import h5py

def save(params,Path):
    file= h5py.File(Path,'w')
    file.create_dataset('LIF',data=params)
    file.close()

