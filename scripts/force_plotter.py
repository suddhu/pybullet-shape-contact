#!/usr/bin/python3

import json, os
import numpy as np
import pdb
import matplotlib.pyplot as plt
import math
import ik.helper as ik

def is_outlier(points, thresh=1.5):
    """
    Returns a boolean array with True if points are outliers and False 
    otherwise.

    Parameters:
    -----------
        points : An numobservations by numdimensions array of observations
        thresh : The modified z-score to use as a threshold. Observations with
            a modified z-score (based on the median absolute deviation) greater
            than this value will be classified as outliers.

    Returns:
    --------
        mask : A numobservations-length boolean array.

    References:
    ----------
        Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
        Handle Outliers", The ASQC Basic References in Quality Control:
        Statistical Techniques, Edward F. Mykytka, Ph.D., Editor. 
    """
    if len(points.shape) == 1:
        points = points[:,None]
    median = np.median(points, axis=0)
    diff = np.sum((points - median)**2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return modified_z_score > thresh
    
def cross2d(a, b):
    return a[0]*b[1] - a[1]*b[0]

def standardRad(t):
  if (t >= 0):
    t = math.fmod(t + np.pi, 2*np.pi) - np.pi
  else:
    t = math.fmod(t - np.pi, -2*np.pi) + np.pi
  return t;

# path = "/home/suddhu/software/pybullet-shape-contact/data/contour_following/all_contact_shape=rect1_rep=0004.json"
path = "/home/suddhu/software/pybullet-shape-contact/data/simple_push/box_push.json"

with open(path) as data_file:    
    mat = json.load(data_file)

all_contacts = np.array(mat["all_contacts"])

# contact_points = all_contacts[:,0:2]
# contact_normals = all_contacts[:,3:5]
# force_mags = all_contacts[:,6].reshape(-1,1)
# contact_forces = np.multiply(force_mags, -contact_normals)
# obj_poses = all_contacts[:,[10,11,13]]

# simple sim
contact_points = all_contacts[:,0:2]
contact_normals = all_contacts[:,2:4]
force_mags = all_contacts[:,4].reshape(-1,1)
contact_forces = np.multiply(force_mags, contact_normals)
# contact_forces = contact_normals

obj_poses = all_contacts[:,5:8]

t = np.arange(len(contact_forces))


# transform contact_points and contact_forces to poses 
length = len(obj_poses) - 1
ls_c = 0.0344 # pressure distribution constant
# ls_c = 3.3985e-05
err = np.zeros((length, 2))
comps1 = np.zeros((length, 2))
comps2 = np.zeros((length, 2))
f_vec = np.zeros((length, 3))

norm_err = np.zeros(length)

v_vec = np.zeros((length, 3))
for i in range(length):
    x_now = obj_poses[i,:]
    x_next = obj_poses[i + 1,:]
    t_c = ik.transform_to_frame2d(contact_points[i,:], x_now)
    f = ik.transform_to_frame2d(contact_forces[i,:], x_now)
    # f = contact_forces[i,:]
    # print('t_c:', t_c, " f: ", f, "f_g: ", contact_forces[i,:],  "c_n_g: ", contact_normals[i,:], 'force mags: ', force_mags[i])
    m = cross2d(t_c, f)  # moment = r cross F

    v = (x_next[0:2] - x_now[0:2])
    # print('v before: ', v)
    v = ik.rotate_to_frame2d(v, x_now) # rotate wrt theta
    # print('v after: ', v)

    omega = standardRad((x_next[2] - x_now[2]))
    # omega = (x_next[2] - x_now[2])

    f_x = f[0]
    f_y = f[1]
    v_x = v[0]
    v_y = v[1]
    c2 = ls_c*ls_c

    c = (v_x/omega) * (m/f_x)

    # err[i, :] = np.array([v_x/omega - c2 * f_x/m, v_y/omega - c2 * f_y/m])
    # err[i, :] = np.array([v_x/omega - c2 * f_x/m, v_y/omega - c2 * f_y/m])

    err[i, :] = np.array([v_x * m - f_x * omega * c2, v_y * m - f_y * omega * c2])
    comps1[i, :] = np.array([v_x * m, f_x * omega * c2])
    comps2[i, :] = np.array([v_y * m, f_y * omega * c2])

    f_vec[i, :] = np.array([f_x, f_y, m])
    # err[i, :] = np.array([c*c, c2])
    v_vec[i, :] = np.array([v_x, v_y, omega])
    # print('x_now: ', x_now, 'x_next:', x_next)

    # print('fx: ', f_x, 'f_y: ', f_y, 'v_x: ', v_x, 'v_y: ', v_y, 'c2: ', c2, 'm: ', m, 'omega:', omega)
    # print('err: ', err[i,:])
    # pdb.set_trace()

plt.subplot(4, 1, 1)
plt.plot(range(length), f_vec[:,0], color='red', linestyle='-', linewidth=2, label='f_x')
plt.plot(range(length), f_vec[:,1], color='yellow', linestyle='-', linewidth=2, label='f_y')
plt.plot(range(length), f_vec[:,2], color='green', linestyle='-', linewidth=2, label='m')
plt.xlabel('time step')
plt.ylabel('force')
plt.title('Forces vs. time')
plt.grid(True)
plt.legend()

plt.subplot(4, 1, 2)
plt.plot(range(length), v_vec[:, 0], color='red', linestyle='-', linewidth=2, label='v_x')
plt.plot(range(length), v_vec[:, 1], color='yellow', linestyle='-', linewidth=2, label='v_y')
plt.plot(range(length), v_vec[:, 2], color='green', linestyle='-', linewidth=2, label='omega')
plt.xlabel('time step')
plt.ylabel('Obj meas')
plt.title('Obj meas vs. time')
plt.grid(True)
plt.legend()

ax1 = plt.subplot(4, 1, 3)
x = err[:,0]
x = x[~is_outlier(x)]
length = len(x)
plt.plot(range(length), x, color='black', linestyle='-', linewidth=1, label='v_x * m - f_x * omega * c^2')
plt.plot(range(length), comps1[~is_outlier(err[:,0]), 0], color='green', linestyle='--', linewidth=1, label='v_x * m ')
plt.plot(range(length), comps1[~is_outlier(err[:,0]), 1], color='blue', linestyle='--', linewidth=1, label='f_x * omega * c^2')
plt.axhline(y=0.0, color='y', linestyle='--')
plt.xlabel('time step')
plt.ylabel('X comps')
plt.title('Err x components vs. time')
plt.grid(True)
plt.legend()

ax1_ylim = ax1.get_ylim()[1] - ax1.get_ylim()[0]

ax2 = plt.subplot(4, 1, 4)
x = err[:,1]
x = x[~is_outlier(x)]
length = len(x)
plt.plot(range(length), x, color='black', linestyle='-', linewidth=1, label='v_y * m - f_y * omega * c^2')
plt.plot(range(length), comps2[~is_outlier(err[:,1]), 0], color='green', linestyle='--', linewidth=1, label='v_y * m')
plt.plot(range(length), comps2[~is_outlier(err[:,1]), 1], color='blue', linestyle='--', linewidth=1, label='f_y * omega * c^2')
plt.axhline(y=0.0, color='y', linestyle='--')
plt.xlabel('time step')
plt.ylabel('Error')
plt.title('Err y components vs. time')
plt.grid(True)
plt.legend()

ax2_ylim = ax2.get_ylim()[1] - ax2.get_ylim()[0]

if ax1_ylim > ax2_ylim:
  ax2.set_ylim(ax1.get_ylim())
else:
  ax1.set_ylim(ax2.get_ylim())


plt.show(block = False)

length = len(obj_poses) - 1
temp_err = np.zeros((length, 2))

c_vals = np.linspace(0, 0.04, 20)
err_x = np.zeros((len(c_vals)))
err_y = np.zeros((len(c_vals)))

k = 0
for ls_c in c_vals:
  print('k: ', k, ' c: ', ls_c)
  for i in range(length):
    x_now = obj_poses[i,:]
    x_next = obj_poses[i + 1,:]
    t_c = ik.transform_to_frame2d(contact_points[i,:], x_now)
    f = ik.transform_to_frame2d(contact_forces[i,:], x_now)
    m = cross2d(t_c, f)  # moment = r cross F

    v = (x_next[0:2] - x_now[0:2])
    v = ik.rotate_to_frame2d(v, x_now) # rotate wrt theta

    omega = standardRad((x_next[2] - x_now[2]))

    f_x = f[0]
    f_y = f[1]
    v_x = v[0]
    v_y = v[1]
    c2 = ls_c*ls_c

    # temp_err[i, :] = np.abs(np.array([v_x/omega - c2 * f_x/m, v_y/omega - c2 * f_y/m]))
    temp_err[i, :]  = np.array([v_x * m - f_x * omega * c2, v_y * m - f_y * omega * c2])
    # print(temp_err[i, :])

  err_x[k] = np.sqrt(np.mean((temp_err[:,0])**2)) 
  err_y[k] = np.sqrt(np.mean((temp_err[:,1])**2))

  # print('err x:', err_x[k])
  # print('err y:', err_y[k])
  k += 1
    
fig, ax = plt.subplots()
plt.ion()

idx_x = np.where(err_x == err_x.min())
idx_y = np.where(err_y == err_y.min())

plt.plot(c_vals, err_x, color='red', linestyle='-', linewidth=1, label='Error X')
plt.plot(c_vals, err_y, color='blue', linestyle='-', linewidth=1, label='Error Y')
plt.plot(c_vals[idx_x], err_x[idx_x], 'ro',  markersize=5)
plt.plot(c_vals[idx_y], err_y[idx_y], 'bo',  markersize=5)

plt.axhline(y=0.0, color='k', linestyle='--')
plt.xlabel('c')
plt.ylabel('Error')
plt.title('Error vs. c')
plt.grid(True)
plt.legend()

plt.show(block = True)
