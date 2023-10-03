import scipy as sci
from scipy.interpolate import RegularGridInterpolator
import numpy as np

def climate_downsample(data, ds_t=0, ds_hw=8, interp_method='linear'):
    if len(data.shape) == 3:
        sigma = [ds_t//2, ds_hw//2, ds_hw//2]
    elif len(data.shape) == 4:  
        # assume data shape is [t, c, h, w]
        sigma = [ds_t//2, 0, ds_hw//2, ds_hw//2]
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