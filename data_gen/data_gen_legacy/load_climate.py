import scipy as sci
from scipy.interpolate import RegularGridInterpolator
import numpy as np

def climate_downsample(data, ds_t=0, ds_hw=8,v_sigma=0, interp_method='linear'):
    if len(data.shape) == 3:
        sigma = [ds_t//2, v_sigma, v_sigma]
    elif len(data.shape) == 4:  
        # assume data shape is [t, c, h, w]
        sigma = [ds_t//2, 0, v_sigma, v_sigma]
    gaussian_filtered_data = sci.ndimage.gaussian_filter(data, sigma=sigma)
    
    interp = RegularGridInterpolator(
        tuple([np.arange(s) for s in list(gaussian_filtered_data.shape)]),
        values=gaussian_filtered_data, method=interp_method
    )


    ds_lst = [ds_factor if ds_factor != 0 else 1 for ds_factor in [ds_t, ds_hw, ds_hw]]

    if len(data.shape) == 4:
        data = data.transpose(0, 2, 3, 1)

    meshgrid_list = [np.linspace(0, gaussian_filtered_data.shape[ds_idx]-1, 
                        gaussian_filtered_data.shape[ds_idx]//ds_factor)
                        for ds_idx, ds_factor in enumerate(ds_lst)]
    meshgrid = np.meshgrid(*meshgrid_list, indexing='ij')
    lres_coord = np.stack(meshgrid, axis=-1)

    if len(data.shape) == 4:
        lres_coord = lres_coord.transpose(0, 3, 1, 2)

    space_time_crop_lres = interp(lres_coord)
    return space_time_crop_lres

if __name__ =="__main__":
    import h5py
    import matplotlib.pyplot as plt
    data = h5py.File('/pscratch/sd/j/junyi012/climate_data/climate_whole_map_c1.h5')
    temp = data['fields'][()]
    for sig in [0,1,2,4]:
        temp_s2_sig2 = climate_downsample(temp, ds_t=0, ds_hw=4, v_sigma=sig)
        print(temp_s2_sig2.shape)
        # write to h5
        with h5py.File(f'/pscratch/sd/j/junyi012/climate_data/climate_preprocessed_s4_sig{sig}.h5', 'w') as f:
            f.create_dataset('fields', data=temp_s2_sig2)
            f.close()
    val = []
    # val.append(temp_s2_sig2[30,:,:])
    # for sig in ["0","1","2","4"]:
    #     with h5py.File(f'/pscratch/sd/j/junyi012/climate_data/climate_preprocessed_s2_sig{sig}.h5', 'r') as f:
    #         print(f[f'fields'][()].std())
    #         d1 = f[f'fields'][()][30,:,:]
    #         val.append(d1)
    #         f.close()
    # # further_downsample = climate_downsample(val[0], ds_t=0, ds_hw=0, v_sigma=50)
    # # val.append(further_downsample)
    # fig,ax = plt.subplots(1,4)
    # i=0
    # for a in ax:
    #     a.imshow(val[i])
    #     i+=1
    # fig.savefig('test.png')
    # i=0
    # fig,ax = plt.subplots(1,4)
    # for a in ax:
    #     a.imshow(np.exp(val[i]-val[0]))
    #     i+=1
    # fig.savefig('test_err.png')