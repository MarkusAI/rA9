import h5py

def save(params,spike_lists,Path):# saving weights and spike_lists
    file= h5py.File(Path,'w')
    file.create_dataset('parameters',data=params)
    file.create_dataset('spike_lists',data=spike_lists)
    file.close()

