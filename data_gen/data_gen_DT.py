import jax
import jax.numpy as jnp
import jax_cfd.base as cfd
import numpy as np
import seaborn
import xarray

# check old data
import h5py
f = h5py.File('/pscratch/sd/j/junyi012/Decay_Turbulence_small/train/Decay_turb_small_128x128_125.h5', 'r')
w_old = f["tasks"]["vorticity"][()]
print(f["tasks"].keys())
print("old data",w_old.shape)

size = 128
density = 1.
viscosity = 1e-3
seed = 125
inner_steps = 20
outer_steps = 2000
#dt = 0.001
max_velocity = 2.0
cfl_safety_factor = 0.5

# Define the physical dimensions of the simulation.
grid = cfd.grids.Grid((size, size), domain=((0, 2 * jnp.pi), (0, 2 * jnp.pi)))

# Construct a random initial velocity. The `filtered_velocity_field` function
# ensures that the initial velocity is divergence free and it filters out
# high frequency fluctuations.
v0 = cfd.initial_conditions.filtered_velocity_field(
    jax.random.PRNGKey(seed), grid, max_velocity)

# Choose a time step.
dt = cfd.equations.stable_time_step(
    max_velocity, cfl_safety_factor, viscosity, grid)
print("cfl dt = ", dt)
dt = 0.001
# Define a step function and use it to compute a trajectory.
step_fn = cfd.funcutils.repeated(
    cfd.equations.semi_implicit_navier_stokes(
        density=density, viscosity=viscosity, dt=dt, grid=grid),
    steps=inner_steps)
rollout_fn = jax.jit(cfd.funcutils.trajectory(step_fn, outer_steps))
time,trajectory = jax.device_get(rollout_fn(v0))



# load into xarray for visualization and analysis
ds = xarray.Dataset(
    {
        'u': (('time', 'x', 'y'), trajectory[0].data),
        'v': (('time', 'x', 'y'), trajectory[1].data),
    },
    coords={
        'x': grid.axes()[0],
        'y': grid.axes()[1],
        'time': dt * inner_steps * np.arange(outer_steps)
    }
)
     
def vorticity(ds):
  return (ds.v.differentiate('x') - ds.u.differentiate('y')).rename('vorticity')

ds['vorticity'] = vorticity(ds)

(ds.pipe(vorticity).thin(time=20)
 .plot.imshow(col='time', cmap=seaborn.cm.icefire, robust=True, col_wrap=5))

import h5py

u = ds['u'].values
v = ds['v'].values
w = ds['vorticity'].values
t = ds['time'].values
print(f"u.shape = {u.shape}, v.shape = {v.shape}, vorticity.shape = {w.shape}, t.shape = {t.shape}")
print(f"type of u = {type(u)}, type of v = {type(v)}, type of vorticity = {type(w)}, type of t = {type(t)}")
with h5py.File('data.h5', 'w') as f:
    tasks = f.create_group('tasks')
    tasks.create_dataset('u', data=u)
    tasks.create_dataset('v', data=v)
    tasks.create_dataset('vorticity', data=w)
    tasks.create_dataset('t', data=t)
print(t)
import matplotlib.pyplot as plt     
fig,ax = plt.subplots(1,2,figsize=(10,5))
ax[0].imshow(w[-1])
ax[1].imshow(w_old[-1])
fig.savefig('vorticity.png')
fig,ax = plt.subplots(1,2,figsize=(10,5))
ax[0].imshow(w[100])
ax[1].imshow(w_old[100])
ax[1].set_title(t[100])
fig.savefig('vorticity2.png')