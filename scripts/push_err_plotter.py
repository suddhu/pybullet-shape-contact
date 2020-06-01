#!/usr/bin/python3

import json, os
import numpy as np
import pdb
import matplotlib.pyplot as plt
import math
import ik.helper as ik
import argparse
from config.shape_db import ShapeDB
from ik.helper import wraptopi, matrix_from_xyzrpy, transform_back_frame2d
import matplotlib.patches as patches
import tf.transformations as tfm

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

def traj_plot(i, x_actual, x_meas, shape_polygon_3d, contact_points, contact_normals, pusher_pos, hasContact, probe_radius, ax): 
    ax.clear()

    # 1: plot ground turth 
    T = matrix_from_xyzrpy([x_actual[i, 0], x_actual[i, 1], 0], [0, 0, x_actual[i, 2]])
    shape_polygon_3d_world = np.dot(T, shape_polygon_3d.T)     # ground truth shape
    gt = patches.Polygon(shape_polygon_3d_world.T[:,0:2], closed=True, linewidth=2, linestyle='--', fill=False, ec='black')
    ax.add_patch(gt)

    # centroid
    centroid = np.array([0, 0, 0])
    centroid = np.hstack((centroid, 1))
    t_c = np.dot(T, centroid.T)
    ax.plot(t_c[0], t_c[1], 'k.',  markersize=5)

    # plot computed 
    T = matrix_from_xyzrpy([x_meas[i, 0], x_meas[i, 1], 0], [0, 0, x_meas[i, 2]])
    shape_polygon_3d_world = np.dot(T, shape_polygon_3d.T)     # ground truth shape
    meas = patches.Polygon(shape_polygon_3d_world.T[:,0:2], closed=True, linewidth=2, linestyle='-', fill=False, ec='green')
    ax.add_patch(meas) 

    # centroid
    t_c = np.dot(T, centroid.T)
    ax.plot(t_c[0], t_c[1], 'g.',  markersize=5)
    
    # probe
    probe = patches.Circle((pusher_pos[i,0], pusher_pos[i,1]), probe_radius, facecolor="red")
    ax.add_patch(probe)

    probe = patches.Circle((pusher_pos[i,2], pusher_pos[i,3]), probe_radius, facecolor="red")
    ax.add_patch(probe)

    # 2: plot contact point 
    if hasContact[i, 0]:
        ax.plot(contact_points[i, 0], contact_points[i, 1], 'ko',  markersize=6)
    if hasContact[i, 1]:
        ax.plot(contact_points[i, 2], contact_points[i, 3], 'ko',  markersize=6)

    if hasContact[i, 0]:
        # 3: plot contact normal
        ax.arrow(contact_points[i, 0], contact_points[i, 1], 
            contact_normals[i, 0]*0.02, contact_normals[i, 1]*0.02, 
            head_width=0.001, head_length=0.01, fc='y', ec='g')
    if hasContact[i, 1]:
        # 3: plot contact normal
        ax.arrow(contact_points[i, 2], contact_points[i, 3], 
            contact_normals[i, 2]*0.02, contact_normals[i, 3]*0.02, 
            head_width=0.001, head_length=0.01, fc='y', ec='g')

    ax.set_xlim(-0.2, 0.2)
    ax.set_ylim(-0.2, 0.2)
    ax.set_title('timestamp: ' + str(i))
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_title('Sim timestamp: ' + str(i))
    plt.draw()
    plt.pause(0.001)

def get_motion_cone_wrt_object(x_cylinder_object, x_contact_object, normvec_object, fric, ls_c):
    theta = math.atan(fric)

    fr = ik.rotate_back_frame2d(normvec_object, np.array([0, 0, -theta]))
    fl = ik.rotate_back_frame2d(normvec_object, np.array([0, 0, theta]))
    
    pr = np.array([fr[0], fr[1], cross2d(x_contact_object, fr)])
    pl = np.array([fl[0], fl[1], cross2d(x_contact_object, fl)])

    c2 = ls_c*ls_c

    qr = np.array([c2 * fr[0], c2 * fr[1], pr[2]])
    ql = np.array([c2 * fl[0], c2 * fl[1], pl[2]])

    xc3 = np.array([x_contact_object[0], x_contact_object[1], 0])
    vr_tmp = np.cross(np.array([0,0,qr[2]]), xc3)
    vl_tmp = np.cross(np.array([0,0,ql[2]]), xc3)

    vr = qr[0:2] + vr_tmp[0:2]
    vl = ql[0:2] + vl_tmp[0:2]
    
    return ql, qr, vl, vr

def pushit_slip_cylinder(x_now, probe_center_now, probe_center_next, contact_point, contact_normal, fric, ls_c):
    x_cylinder_object = ik.transform_to_frame2d(probe_center_now, x_now)
    normvec_object = ik.rotate_to_frame2d(contact_normal, x_now)
    x_contact_object = ik.transform_to_frame2d(contact_point, x_now)

    ql, qr, vl, vr = get_motion_cone_wrt_object(x_cylinder_object, x_contact_object, normvec_object, fric, ls_c)

    vp = ik.rotate_to_frame2d(probe_center_next - probe_center_now, x_now)

    if ((cross2d(vr, vp) >= 0) and (cross2d(vl, vp) <= 0)):
        c2 = ls_c*ls_c
        xc1_2 = x_contact_object[0]*x_contact_object[0]
        xc2_2 = x_contact_object[1]*x_contact_object[1]
        xc12 = x_contact_object[0] * x_contact_object[1]
        denom = c2 + xc1_2 + xc2_2
        vx = ((c2 + xc1_2) * vp[0]  +  xc12 * vp[1])
        vy = ((c2 + xc1_2) * vp[1]  +  xc12 * vp[0])
        omega = (x_contact_object[0] * vy - x_contact_object[1] * vx) / c2
    else:
        if(cross2d(vp, vr) > 0):
            vb = vr
            qb = qr
        else:
            vb = vl
            qb = ql
        kappa = np.dot(vp, normvec_object) / np.dot(vb, normvec_object)
        q = kappa * qb
        vx = q[0]
        vy = q[1]
        omega = q[2]

    v_global = ik.rotate_back_frame2d(np.array([vx, vy]), x_now)

    x_delta = np.array([v_global[0], v_global[1], omega])
    return x_delta

def run(shape):
#   path = "/home/suddhu/software/pybullet-shape-contact/data/contour_following/"
#   path = path + shape
  print("path: ", shape)
  path = shape

  with open(path) as data_file:    
      mat = json.load(data_file)

  has_contacts = np.array(mat["has_contact"])
  contact_points = np.array(mat["contact_point"])
  contact_normals = np.array(mat["contact_normal"])
  obj_poses = np.array(mat["pose_true"])
  pusher_pos = np.array(mat["pusher"])

  # transform contact_points and contact_forces to poses 
  length = len(obj_poses) - 1
  ls_c = 0.0344 # pressure distribution constant
  fric = 0.25
  err = np.zeros((length, 3))
  comps1 = np.zeros((length, 3))
  comps2 = np.zeros((length, 3))

#   for i in range(length):

#     x_now = obj_poses[i,:]
#     x_next = obj_poses[i + 1,:]

#     # pdb.set_trace()
#     # pushit_slip_cylinder
#     x_delta = pushit_slip_cylinder(x_now, pusher_pos[i, 0:2], pusher_pos[i + 1, 0:2], contact_points[i, :], contact_normals[i, :], fric, ls_c)
 
#     x_delta_actual = x_next - x_now
#     err[i,:] = x_delta_actual - x_delta
#     comps1[i,:] = x_delta
#     comps2[i, :] = x_delta_actual
#     # print('x_delta: ', x_delta, 'x_delta_actual: ', x_delta_actual)

#   plt.subplot(4, 1, 1)
#   plt.plot(range(length), err[:,0], color='red', linestyle='-', linewidth=2, label='x')
#   plt.plot(range(length), err[:,1], color='yellow', linestyle='-', linewidth=2, label='y')
#   plt.plot(range(length), err[:,2], color='green', linestyle='-', linewidth=2, label='theta')
#   plt.axhline(y=0.0, color='k', linestyle='--')
#   plt.xlabel('time step')
#   plt.ylabel('Error')
#   plt.title('Error vs. time step')
#   plt.grid(True)
#   plt.legend()

#   plt.subplot(4, 1, 2)
#   plt.plot(range(length), comps1[:,0], color='blue', linestyle='-', linewidth=2, label='x pred')
#   plt.plot(range(length), comps2[:,0], color='orange', linestyle='-', linewidth=2, label='x actual')
#   plt.axhline(y=0.0, color='k', linestyle='--')
#   plt.xlabel('time step')
#   plt.ylabel('X computed')
#   plt.title('X vs. time step')
#   plt.grid(True)
#   plt.legend()

#   plt.subplot(4, 1, 3)
#   plt.plot(range(length), comps1[:,1], color='blue', linestyle='-', linewidth=2, label='y pred')
#   plt.plot(range(length), comps2[:,1], color='orange', linestyle='-', linewidth=2, label='y actual')
#   plt.axhline(y=0.0, color='k', linestyle='--')
#   plt.xlabel('time step')
#   plt.ylabel('Y computed')
#   plt.title('Y vs. time step')
#   plt.grid(True)
#   plt.legend()

#   plt.subplot(4, 1, 4)
#   plt.plot(range(length), comps1[:,2], color='blue', linestyle='-', linewidth=2, label='theta pred')
#   plt.plot(range(length), comps2[:,2], color='orange', linestyle='-', linewidth=2, label='theta actual')
#   plt.axhline(y=0.0, color='k', linestyle='--')
#   plt.xlabel('time step')
#   plt.ylabel('theta computed')
#   plt.title('theta vs. time step')
#   plt.grid(True)
#   plt.legend()

  ## trajectory 
  shape_id = 'rect1'
  shape_db = ShapeDB()
  shape = shape_db.shape_db['rect1']['shape'] # shape of the objects presented as polygon.
  shape_type = shape_db.shape_db['rect1']['shape_type']
  probe_radius = 0.00313
  
  if shape_type == 'poly':
      shape_polygon_3d = np.hstack((np.array(shape), np.zeros((len(shape), 1)), np.ones((len(shape), 1))))
  elif shape_type == 'ellip':
      shape = shape[0]
  elif shape_type == 'polyapprox':
      shape_polygon_3d = np.hstack((np.array(shape[0]), np.zeros((len(shape[0]), 1)), np.ones((len(shape[0]), 1))))
    
  x_meas = np.zeros((len(obj_poses), 3))
  x_actual = obj_poses

#   pdb.set_trace()

  x_now = obj_poses[0, :]
  x_meas[0, :] = x_now
  fig, ax = plt.subplots()
  ax.axis('equal')
  plt.ion()

  for i in range(length):
    # pushit_slip_cylinder
    if has_contacts[i, 0]:
        x_delta = pushit_slip_cylinder(x_now, pusher_pos[i, 0:2], pusher_pos[i + 1, 0:2], contact_points[i, 0:2], contact_normals[i, 0:2], fric, ls_c)
        x_now += x_delta

    if has_contacts[i, 1]:
        x_delta = pushit_slip_cylinder(x_now, pusher_pos[i, 2:4], pusher_pos[i + 1, 2:4], contact_points[i, 2:4], contact_normals[i, 2:4], fric, ls_c)
        x_now += x_delta

    x_meas[i + 1, :] = x_now

    if (i % 10 == 0):
        traj_plot(i, x_actual, x_meas, shape_polygon_3d, contact_points,  contact_normals, pusher_pos, has_contacts, probe_radius, ax)
    print('x_delta: ', x_delta)

  plt.show(block = True)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--json", type=str, default="rect1", help="Shape ID (eg: rect1, ellip2, hex)")
  args = parser.parse_args()
  run(args.json)

#   run("/home/suddhu/software/pybullet-shape-contact/data/contour_following/all_contact_shape=rect1_rep=0079.json")
