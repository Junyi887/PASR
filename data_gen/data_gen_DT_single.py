import jax
import jax.numpy as jnp
import jax_cfd.base as cfd
from jax_cfd.base import grids
from jax_cfd.base import boundaries
from jax_cfd.base import funcutils
from jax_cfd.base import pressure
import numpy as np
import seaborn
import xarray
import matplotlib.pyplot as plt
# check old data
import h5py

# f = h5py.File('/pscratch/sd/j/junyi012/Decay_Turbulence_small/train/Decay_turb_small_128x128_125.h5', 'r')
# w_old = f["tasks"]["vorticity"][()]
# print(f["tasks"].keys())
# print("old data",w_old.shape)
import argparse
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--res', type=int, default=128)
parser.add_argument('--scale', type=int, default=4)
parser.add_argument('--seed', type=int, default=110)
args=parser.parse_args()

size = args.res
scale = args.scale
density = 1.
viscosity = 1e-3
seed = args.seed

#dt = 0.001
max_velocity = 2.0
cfl_safety_factor = 0.5

# Define the physical dimensions of the simulation.
grid_hr = cfd.grids.Grid((size, size), domain=((0, 2 * jnp.pi), (0, 2 * jnp.pi)))
grid_ref = cfd.grids.Grid((1024, 1024), domain=((0, 2 * jnp.pi), (0, 2 * jnp.pi)))
# Construct a random initial velocity. The `filtered_velocity_field` function
# ensures that the initial velocity is divergence free and it filters out
# high frequency fluctuations.
print(f"generate initial velocity for seed {seed}")
v0_hr = cfd.initial_conditions.filtered_velocity_field(
    jax.random.PRNGKey(seed), grid_hr, max_velocity)
u_hr = v0_hr[0].data
v_hr = v0_hr[1].data


# Choose a time step
dt_ref = cfd.equations.stable_time_step(
    max_velocity, cfl_safety_factor, viscosity, grid_ref)
print("cfl dt lr = ", dt_ref)
dt_hr = cfd.equations.stable_time_step(
    max_velocity, cfl_safety_factor, viscosity, grid_hr)
print("cfl dt hr = ", dt_hr)

final_time = 10
outer_steps = 200
dt = dt_ref
inner_steps = (final_time // dt) // outer_steps
import time
# Define a step function and use it to compute a trajectory.
 

start_time = time.time()
step_fn_hr = cfd.funcutils.repeated(
    cfd.equations.semi_implicit_navier_stokes(
        density=density, viscosity=viscosity, dt=dt, grid=grid_hr),
    steps=inner_steps)
rollout_fn_hr = jax.jit(cfd.funcutils.trajectory(step_fn_hr, outer_steps))
time2,trajectory_hr = jax.device_get(rollout_fn_hr(v0_hr))
end_time = time.time()

print(f"simulation time = {end_time-start_time}s at args.res = {size}")
# load into xarray for visualization and analysis
ds_hr = xarray.Dataset(
    {
        'u': (('time', 'x', 'y'), trajectory_hr[0].data),
        'v': (('time', 'x', 'y'), trajectory_hr[1].data),
    },
    coords={
        'x': grid_hr.axes()[0],
        'y': grid_hr.axes()[1],
        'time': dt * inner_steps * np.arange(outer_steps)
    }
)    

# def vorticity(ds):
#   return (ds.v.differentiate('x') - ds.u.differentiate('y')).rename('vorticity')

# ds_lr['vorticity'] = vorticity(ds_lr)
# ds_hr['vorticity'] = vorticity(ds_hr)
# # (ds_lr.pipe(vorticity).thin(time=20)
# #  .plot.imshow(col='time', cmap=seaborn.cm.icefire, robust=True, col_wrap=5))
# '''load to h5 file'''
# import h5py
# u_lr = ds_lr['u'].values
# v_lr = ds_lr['v'].values
# w_lr = ds_lr['vorticity'].values
# u_hr = ds_hr['u'].values
# v_hr = ds_hr['v'].values
# w_hr = ds_hr['vorticity'].values
# t = ds_lr['time'].values
# print(f"u_lr.shape = {u_lr.shape}, v_lr.shape = {v_lr.shape}, vorticity_lr.shape = {w_lr.shape}")
# print(f"u_hr.shape = {u_hr.shape}, v_hr.shape = {v_hr.shape}, vorticity_hr.shape = {w_hr.shape}")
# print(f"type of u_lr = {type(u_lr)}, type of v_lr = {type(v_lr)}, type of vorticity_lr = {type(w_lr)}")
# print(f"type of u_hr = {type(u_hr)}, type of v_hr = {type(v_hr)}, type of vorticity_hr = {type(w_hr)}")
# # trim data
# u_hr = u_hr[50:550].astype(jnp.float32)
# v_hr = v_hr[50:550].astype(jnp.float32)
# w_hr = w_hr[50:550].astype(jnp.float32)
# u_lr = u_lr[50:550].astype(jnp.float32)
# v_lr = v_lr[50:550].astype(jnp.float32)
# w_lr = w_lr[50:550].astype(jnp.float32)
# t = t[50:550]  
# with h5py.File(f'decay_turb_lres_sim_res{size}_s{scale}_{seed}.h5', 'w') as f:
#     tasks = f.create_group('tasks')
#     tasks.create_dataset('u', data=u_hr)
#     tasks.create_dataset('v', data=v_hr)
#     tasks.create_dataset('vorticity', data=w_hr)
#     tasks.create_dataset('t', data=t)
#     tasks.create_dataset('u_lr', data=u_lr)
#     tasks.create_dataset('v_lr', data=v_lr)
#     tasks.create_dataset('vorticity_lr', data=w_lr)

# print(t)
import matplotlib.pyplot as plt     

# corrlation 
# baseline_palette = seaborn.color_palette('YlGnBu', n_colors=7)[1:]
# correlation = summary.vorticity_correlation.sel(time=slice(20)).compute()