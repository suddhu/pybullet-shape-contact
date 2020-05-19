#!/usr/bin/python3

import json, os
import numpy as np
import pdb
import matplotlib.pyplot as plt
import math
import ik.helper as ik


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
f_vec = np.zeros((length, 2))

norm_err = np.zeros(length)

v_vec = np.zeros((length, 2))
for i in range(length):
    x_now = obj_poses[i,:]
    x_next = obj_poses[i + 1,:]
    t_c = ik.transform_to_frame2d(contact_points[i,:], x_now)
    f = ik.transform_to_frame2d(contact_forces[i,:], x_now)
    # f = contact_forces[i,:]
    print('t_c:', t_c, " f: ", f, "f_g: ", contact_forces[i,:],  "c_n_g: ", contact_normals[i,:], 'force mags: ', force_mags[i])
    m = cross2d(t_c, f)  # moment = r cross F

    v = (x_next[0:2] - x_now[0:2])
    print('v before: ', v)
    v = ik.rotate_to_frame2d(v, x_now) # rotate wrt theta
    print('v after: ', v)

    omega = standardRad((x_next[2] - x_now[2]))

    f_x = f[0]
    f_y = f[1]
    v_x = v[0]
    v_y = v[1]
    c2 = ls_c*ls_c

    c = (v_x/omega) * (m/f_x)
    err[i, :] = np.array([v_x * m - f_x * omega * c2, v_y * m - f_y * omega * c2])
    comps1[i, :] = np.array([v_x * m, f_x * omega * c2])
    comps2[i, :] = np.array([v_y * m, f_y * omega * c2])

    f_vec[i, :] = np.array([f_x, f_y])
    # err[i, :] = np.array([c*c, c2])
    v_vec[i, :] = np.array([v_x, v_y])
    print('x_now: ', x_now, 'x_next:', x_next)

    print('fx: ', f_x, 'f_y: ', f_y, 'v_x: ', v_x, 'v_y: ', v_y, 'c2: ', c2, 'm: ', m, 'omega:', omega)
    print('err: ', err[i,:])
    # pdb.set_trace()

plt.subplot(2, 2, 1)
plt.plot(range(length), f_vec[:,0], color='red', linestyle='-', linewidth=2, label='f_x')
plt.plot(range(length), f_vec[:,1], color='yellow', linestyle='-', linewidth=2, label='f_y')
plt.xlabel('time step')
plt.ylabel('force')
plt.title('Forces vs. time')
plt.grid(True)
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(range(length), v_vec[:, 0], color='red', linestyle='-', linewidth=2, label='v_x')
plt.plot(range(length), v_vec[:, 1], color='yellow', linestyle='-', linewidth=2, label='v_y')
plt.xlabel('time step')
plt.ylabel('Obj meas')
plt.title('Obj meas vs. time')
plt.grid(True)
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(range(length), err[:, 0], color='black', linestyle='-', linewidth=2, label='v_x * m - f_x * omega * c^2')
plt.plot(range(length), comps1[:, 0], color='green', linestyle='--', linewidth=2, label='v_x * m ')
plt.plot(range(length), comps1[:, 1], color='blue', linestyle='--', linewidth=2, label='f_x * omega * c^2')
plt.xlabel('time step')
plt.ylabel('X comps')
plt.title('Err x components vs. time')
plt.grid(True)
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(range(length), err[:, 1], color='black', linestyle='-', linewidth=2, label='v_y * m - f_y * omega * c^2')
plt.plot(range(length), comps2[:, 0], color='green', linestyle='--', linewidth=2, label='v_y * m')
plt.plot(range(length), comps2[:, 1], color='blue', linestyle='--', linewidth=2, label='f_y * omega * c^2')
plt.xlabel('time step')
plt.ylabel('Error')
plt.title('Err y components vs. time')
plt.grid(True)
plt.legend()



plt.show()