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
import argparse
import os

def vorticity(ds):
  return (ds.v.differentiate('x') - ds.u.differentiate('y')).rename('vorticity')

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--res', type=int, default=1024)
parser.add_argument('--lr_res', type=int, default=512)

parser.add_argument('--seed', type=int, default=110)
args=parser.parse_args()

size = args.res

density = 1.
viscosity = 1e-3
seed = args.seed
outer_steps = 2000
final_time = 10 

LR_res = args.lr_res
max_velocity = 2.0
cfl_safety_factor = 0.5
output_dir = '/pscratch/sd/j/junyi012/DT_multi_resolution_unpaired/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
# Define the physical dimensions of the simulation.
grid_hr = cfd.grids.Grid((size, size), domain=((0, 2 * jnp.pi), (0, 2 * jnp.pi)))
grid_lr = cfd.grids.Grid((LR_res,LR_res), domain=((0, 2 * jnp.pi), (0, 2 * jnp.pi)))
# Construct a random initial velocity. The `filtered_velocity_field` function
# ensures that the initial velocity is divergence free and it filters out
# high frequency fluctuations.
import os
if os.path.exists(f'data_gen/DT_IC/initial_velocity_{seed}_u.npy') and os.path.exists(f'data_gen/DT_IC/initial_velocity_{seed}_v.npy'):
    u_hr = np.load(f'data_gen/DT_IC/initial_velocity_{seed}_u.npy')
    v_hr = np.load(f'data_gen/DT_IC/initial_velocity_{seed}_v.npy')
    print("load initial velocity from file")
else:
    print(f"generate initial velocity for seed {seed}")
    v0_hr = cfd.initial_conditions.filtered_velocity_field(
        jax.random.PRNGKey(seed), grid_hr, max_velocity)
    u_hr = v0_hr[0].data
    v_hr = v0_hr[1].data
    np.save(f'data_gen/DT_IC/initial_velocity_{seed}_u.npy',u_hr)
    np.save(f'data_gen/DT_IC/initial_velocity_{seed}_v.npy',v_hr)

scale_factor = size//LR_res
u_lr = u_hr[::scale_factor,::scale_factor]
v_lr = v_hr[::scale_factor,::scale_factor]


def load_lr_IC(u_lr,v_lr,iterations: int = 3, maximum_velocity: float = max_velocity):
    GridVariable = grids.GridVariable
    GridVariableVector = grids.GridVariableVector
    def _max_speed(v):
        return jnp.linalg.norm(jnp.array([u.data for u in v]), axis=0).max()
    boundary_conditions = []
    velocity_components = []
    velocity_components.append(u_lr)
    velocity_components.append(v_lr)
    boundary_conditions.append(boundaries.periodic_boundary_conditions(grid_lr.ndim))
    boundary_conditions.append(boundaries.periodic_boundary_conditions(grid_lr.ndim))
    velocity= cfd.initial_conditions.wrap_variables(velocity_components, grid_lr, boundary_conditions)
    def project_and_normalize(v: GridVariableVector):
        v = pressure.projection(v)
        vmax = _max_speed(v)
        v = tuple(
            grids.GridVariable(maximum_velocity * u.array / vmax, u.bc) for u in v)
        return v
  # Due to numerical precision issues, we repeatedly normalize and project the
  # velocity field. This ensures that it is divergence-free and achieves the
  # specified maximum velocity.
    return funcutils.repeated(project_and_normalize, iterations)(velocity)

v0_lr = load_lr_IC(u_lr,v_lr)

# fig,ax = plt.subplots(2,2,figsize=(5,5))
# ax[0,0].imshow(v0_hr[0].data)
# ax[0,1].imshow(v0_lr[0].data)
# ax[1,0].imshow(v0_hr[1].data)
# ax[1,1].imshow(v0_lr[1].data)
# fig.savefig(f'data_gen/initial_velocity_{seed}.png')

# Choose a time step
dt_lr = cfd.equations.stable_time_step(
    max_velocity, cfl_safety_factor, viscosity, grid_lr)
print("cfl dt lr = ", dt_lr)
dt_hr = cfd.equations.stable_time_step(
    max_velocity, cfl_safety_factor, viscosity, grid_hr)
print("cfl dt hr = ", dt_hr)

inner_steps = (final_time // dt_hr) // outer_steps
print(f"inner_steps = {inner_steps}")
# Define a step function and use it to compute a trajectory.

step_fn_lr = cfd.funcutils.repeated(
    cfd.equations.semi_implicit_navier_stokes(
        density=density, viscosity=viscosity, dt=dt_hr, grid=grid_lr),
    steps=inner_steps)
rollout_fn_lr = jax.jit(cfd.funcutils.trajectory(step_fn_lr, outer_steps))
time,trajectory_lr = jax.device_get(rollout_fn_lr(v0_lr))

# load into xarray for visualization and analysis
ds_lr = xarray.Dataset(
    {
        'u': (('time', 'x', 'y'), trajectory_lr[0].data),
        'v': (('time', 'x', 'y'), trajectory_lr[1].data),
    },
    coords={
        'x': grid_lr.axes()[0],
        'y': grid_lr.axes()[1],
        'time': dt_hr * inner_steps * np.arange(outer_steps)
    }
)    


ds_lr['vorticity'] = vorticity(ds_lr)

# (ds_lr.pipe(vorticity).thin(time=20)
#  .plot.imshow(col='time', cmap=seaborn.cm.icefire, robust=True, col_wrap=5))
'''load to h5 file'''
import h5py
u_lr = ds_lr['u'].values
v_lr = ds_lr['v'].values
w_lr = ds_lr['vorticity'].values
t = ds_lr['time'].values


# trim data

u_lr = u_lr.astype(jnp.float32)
v_lr = v_lr.astype(jnp.float32)
w_lr = w_lr.astype(jnp.float32)
# t = t[50:550]  
# check if directory exists otherwise create it
print(f"u_lr.shape = {u_lr.shape}, v_lr.shape = {v_lr.shape}, vorticity_lr.shape = {w_lr.shape}, args.lr_res = {args.lr_res}")

with h5py.File(output_dir + f'DT_lres_sim_res_{LR_res}_{seed}.h5', 'w') as f:
    tasks = f.create_group('tasks')
    tasks.create_dataset('u', data=u_lr)
    tasks.create_dataset('v', data=v_lr)
    tasks.create_dataset('vorticity', data=w_lr)
    tasks.create_dataset('t', data=t)

print(t)
import matplotlib.pyplot as plt     
# from jax_cfd.data import evaluation

# corrlation 
# baseline_palette = seaborn.color_palette('YlGnBu', n_colors=7)[1:]
# correlation = ds1.vorticity_correlation.sel(time=slice(20)).compute()