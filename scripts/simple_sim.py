#!/usr/bin/python3

# Sudharshan Suresh (suddhu@cmu.edu), Jan 2020
# Simulation functions for contour following

import pybullet as p
import pybullet_data
import time
import math
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.transforms as tfs
import pdb
from ik.helper import wraptopi, matrix_from_xyzrpy, transform_back_frame2d
from config.shape_db import ShapeDB
import tf.transformations as tfm
import json, os
import subprocess, glob
import argparse
import force_plotter as fp

STATIC_VELOCITY_THRESHOLD = 1e-4

fig, ax = plt.subplots()
ax.axis('equal')
plt.ion()
plt.show()

colname =  [
  "hasContact"
  "x of contact position", 
  "y of contact position", 
  "x of contact normal", 
  "y of contact normal", 
  "force magnitude",
  "x of pusher position", 
  "y of pusher position", 
  "x of ground truth object pose", 
  "y of ground truth object pose", 
  "yaw of ground truth object pose",
 ]

class Sim():
    def __init__(self, shape_id, withVis=False, withForcePlot = False):

        self.start_time = time.time()
        self.plot = withVis
        self.wfp = withForcePlot

        # connect to pybullet server
        p.connect(p.GUI)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI,0)
        p.resetDebugVisualizerCamera( cameraDistance=0.2, cameraYaw=-30, cameraPitch=-90, cameraTargetPosition=[0,0,0])

        # set additional path to find kuka model
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setTimeStep(1. / 240.)

        self.center_world = [0, 0, 0]

        # set gravity
        p.setGravity(0, 0, -10)

        # set simulation length
        self.limit = 3000
        self.threshold = 0.000  # the threshold force for contact, need to be tuned
        self.probe_radius = 0.00313
        self.length = 0.115

        self.pusher_pose = [0, -(self.probe_radius + 0.08), self.length]
        # self.pusher_pose_2 = [0.02, -(self.probe_radius + 0.06), 0.01]

        # pre-define the trajectory/force vectors
        self.pose_true = np.zeros((self.limit, 3))
        self.pusher_position = np.zeros((2, self.limit, 2))
        self.contactPt = np.zeros((2, self.limit, 2))
        self.contactForce = np.zeros((2, self.limit))
        self.contactNormal = np.zeros((2, self.limit, 2))
        self.hasContact = np.full((2, self.limit), False)
        
        self.shape_id = shape_id
        shape_db = ShapeDB()
        shape = shape_db.shape_db[self.shape_id]['shape'] # shape of the objects presented as polygon.
        self.shape_type = shape_db.shape_db[self.shape_id]['shape_type']

        if self.shape_type == 'poly':
            self.shape_polygon_3d = np.hstack((np.array(shape), np.zeros((len(shape), 1)), np.ones((len(shape), 1))))
        elif self.shape_type == 'ellip':
            self.shape = shape[0]
        elif self.shape_type == 'polyapprox':
            self.shape_polygon_3d = np.hstack((np.array(shape[0]), np.zeros((len(shape[0]), 1)), np.ones((len(shape[0]), 1))))

        # Block: add block and update physical parameters
        shape_moment = [1e-3, 1e-3, shape_db.shape_db[self.shape_id]['moment_of_inertia']]
        self.shape_mass = shape_db.shape_db[self.shape_id]['mass']
        fric = 0.25

        urdf_file = "/home/suddhu/software/pybullet-shape-contact/models/shapes/" + self.shape_id + ".urdf"
        self.box = p.loadURDF(urdf_file, [self.center_world[0], self.center_world[0], 0.01] )

        p.changeDynamics(self.box, -1, mass=self.shape_mass, lateralFriction=fric,
                         localInertiaDiagonal=shape_moment, collisionMargin = 1e-6)

        all_dynamics = p.getDynamicsInfo(self.box, -1)
        print('file: ', urdf_file, '\n','mass: ', all_dynamics[0],
             ' lat_fric: ', all_dynamics[1], ' moment of inertia: ', all_dynamics[2],
              ' centroid: ', all_dynamics[3], ' spin_fric: ', all_dynamics[7])

        urdf_file = "/home/suddhu/software/pybullet-shape-contact/models/shapes/pusher.urdf"

        self.centroid = np.hstack((np.array(all_dynamics[3]), 1))

        self.pusher = p.loadURDF(urdf_file, self.pusher_pose)
        self.cid = p.createConstraint(self.pusher, -1, -1, -1, p.JOINT_FIXED, [0, 0, 0], [0, 0, 0], self.pusher_pose)
        self.numJoints = p.getNumJoints(self.pusher)
        print('self.numJoints: ', self.numJoints)
        p.changeDynamics(self.pusher, 0, mass=self.shape_mass, lateralFriction=1.0, collisionMargin = 1e-6)
        p.changeDynamics(self.pusher, 1, mass=self.shape_mass, lateralFriction=1.0, collisionMargin = 1e-6)

        # self.pusher_2 = p.loadURDF(urdf_file, self.pusher_pose_2)
        # self.cid_2 = p.createConstraint(self.pusher_2, -1, -1, -1, p.JOINT_FIXED, [0, 0, 0], [0, 0, 0], self.pusher_pose_2)
        # p.changeDynamics(self.pusher_2, -1, mass=self.shape_mass, lateralFriction=1.0)

        urdf_file = "/home/suddhu/software/pybullet-shape-contact/models/ground_plane/ground_plane.urdf"

        # add plane to push on (slightly below the base of the robot)
        self.planeId = p.loadURDF(urdf_file, [0, 0, 0], useFixedBase=True)

        p.changeDynamics(self.planeId, -1, lateralFriction=1.0, collisionMargin = 1e-6)

        all_dynamics = p.getDynamicsInfo(self.planeId, -1)

        print('shape file: ', urdf_file, '\n','mass: ', all_dynamics[0],
             ' lat_fric: ', all_dynamics[1], ' moment of inertia: ', all_dynamics[2],
              ' centroid: ', all_dynamics[3], ' spin_fric: ', all_dynamics[7])

        # input('Click Enter!')

    def observe_block(self, blockID, linkID = -1):
        if linkID == -1:
            state = p.getBasePositionAndOrientation(blockID)
        else:
            state = p.getLinkState(blockID, linkID)
        xb = state[0][0]
        yb = state[0][1]
        roll, pitch, yaw = p.getEulerFromQuaternion(state[1])
        return np.array((xb, yb, yaw))  

    def static_environment(self):
        v, va = p.getBaseVelocity(self.box)

        # print(v[0:2])
        if (np.linalg.norm(v[0:2]) > STATIC_VELOCITY_THRESHOLD) or (
                np.linalg.norm(va) > STATIC_VELOCITY_THRESHOLD):
        # if (np.linalg.norm(v[0:2]) > STATIC_VELOCITY_THRESHOLD):
            # print('vnorm = {}'.format(np.linalg.norm(v)))
            # print('vanorm = {}'.format(np.linalg.norm(va)))
            return False
        return True

    def simulate(self):
        self.contact_count = 0

        num = 1
        filename = 'all_contact_shape=%s_rep=%04d' % (self.shape_id, num)
        dir_base = "/home/suddhu/software/pybullet-shape-contact/data/contour_following"
        jsonfilename = dir_base+'/%s.json' % filename
        while os.path.isfile(jsonfilename):
            num = num + 1
            filename = 'all_contact_shape=%s_rep=%04d' % (self.shape_id, num)
            jsonfilename = dir_base+'/%s.json' % filename

        self.direc = np.array([0, 1.0, 0.0])
        step_size = 1.0e-3
        while True:
            # time.sleep(1./240.)

            pusher_pos = self.observe_block(self.pusher) + self.direc*step_size

            # limitForce = self.shape_mass*10*0.8
            # print(limitForce)
            if (self.contact_count > 0):
                if not self.hasContact[0, self.contact_count - 1]:
                    orn = p.getQuaternionFromEuler([0, 0, pusher_pos[2] + 0.1]) ## anti clock
                # elif not self.hasContact[self.contact_count - 1, 1]:
                #     orn = p.getQuaternionFromEuler([0, 0, pusher_pos[2] - 0.1]) ## clock
            else:
                orn = p.getQuaternionFromEuler([0, 0, pusher_pos[2]]) ## ee orientation
            p.changeConstraint(self.cid, jointChildPivot=[pusher_pos[0],  pusher_pos[1], self.length], jointChildFrameOrientation=orn, maxForce=5)

            p.stepSimulation()
            # pdb.set_trace()
            contactInfo_1 = p.getContactPoints(self.box, self.pusher, linkIndexA= -1, linkIndexB= 0)
            contactInfo_2 = p.getContactPoints(self.box, self.pusher, linkIndexA= -1, linkIndexB= 1)

            box_pos = self.observe_block(self.box)
            pusher_pos_1 = self.observe_block(self.pusher, 0)  
            pusher_pos_2 = self.observe_block(self.pusher, 1)  

            # get the net contact force between robot and block
            if ((len(contactInfo_1)>0) or (len(contactInfo_2)>0)):
                # pdb.set_trace()
                self.pusher_position[0, self.contact_count, :] = pusher_pos_1[0:2]
                self.pusher_position[1, self.contact_count, :] = pusher_pos_2[0:2]
                self.pose_true[self.contact_count, :] = box_pos
                
                # 1st pusher
                f_c_temp_1 = f_c_temp_2 = 0
                if (len(contactInfo_1)>0):
                    for c in range(len(contactInfo_1)):
                        f_c_temp_1 += contactInfo_1[c][9]
                    
                    self.contactForce[0, self.contact_count] = f_c_temp_1
                    self.contactPt[0, self.contact_count, :] =  contactInfo_1[0][5][:2]
                    self.contactNormal[0, self.contact_count, :] = contactInfo_1[0][7][:2]
                    self.hasContact[0, self.contact_count] = True

                # 2nd pusher 
                if (len(contactInfo_2)>0):
                    for c in range(len(contactInfo_2)):
                        f_c_temp_2 += contactInfo_2[c][9]
                    
                    self.contactForce[1, self.contact_count] = f_c_temp_2
                    self.contactPt[1, self.contact_count, :] =  contactInfo_2[0][5][:2]
                    self.contactNormal[1, self.contact_count, :] = contactInfo_2[0][7][:2]
                    self.hasContact[1, self.contact_count] = True

                if self.hasContact[0, self.contact_count] and self.hasContact[1, self.contact_count]:
                    good_normal = (self.contactNormal[0, self.contact_count, :] + self.contactNormal[1, self.contact_count, :])/2.0
                    # print('good normal: ', good_normal)
                    self.direc = np.dot(tfm.euler_matrix(0,0,1.7*np.pi/3) , np.multiply(-1,good_normal).tolist() + [0] + [1])[0:3]
                elif self.hasContact[0, self.contact_count]:
                    good_normal = self.contactNormal[0, self.contact_count, :]
                    self.direc = np.dot(tfm.euler_matrix(0,0,1.7*np.pi/3) , np.multiply(-1,good_normal).tolist() + [0] + [1])[0:3]
                # else:
                #     good_normal = self.contactNormal[self.contact_count, 2:4]
                #     self.direc = np.dot(tfm.euler_matrix(0,0,2) , np.multiply(-1,good_normal).tolist() + [0] + [1])[0:3]  
                        
                if self.plot and self.contact_count % 10 == 0:
                    self.plotter(self.contact_count)
                
                self.contact_count += 1

            # 3.5 break if we collect enough
            if self.contact_count == self.limit:
                break
        
        skip = 1

        with open(jsonfilename, 'w') as outfile:
            json.dump({'has_contact': self.hasContact[:, ::skip].tolist(),
                        'contact_force': self.contactForce[:, ::skip].tolist(),
                        'contact_normal': self.contactNormal[:, ::skip, :].tolist(),
                        'contact_point': self.contactPt[:, ::skip, :].tolist(),
                        'pose_true': self.pose_true[::skip, :].tolist(),
                        'pusher': self.pusher_position[:, ::skip, :].tolist(),
                        '__title__': colname, 
                        "shape_id": self.shape_id,
                        "probe_radius": self.probe_radius,
                        "offset": self.center_world, 
                        "limit": self.limit}, outfile, sort_keys=True, indent=1)      
        print('file: ', jsonfilename)
        if self.wfp:
            fp.run(jsonfilename)
        return

    def plotter(self, i): 
        ax.clear()
        # 1: plot object
        T = matrix_from_xyzrpy([self.pose_true[i, 0], self.pose_true[i, 1], 0], [0, 0, self.pose_true[i, 2]])

        # ground truth shape
        if self.shape_type == 'poly' or self.shape_type == 'polyapprox':
            shape_polygon_3d_world = np.dot(T, self.shape_polygon_3d.T)
            gt = patches.Polygon(shape_polygon_3d_world.T[:,0:2], closed=True, linewidth=2, linestyle='-', fill=True, fc='grey')
        elif self.shape_type == 'ellip':
            scale, shear, angles, trans, persp = tfm.decompose_matrix(T)
            gt = patches.Ellipse(trans[0:2], self.shape[0]*2, self.shape[1]*2, angle=angles[2]/np.pi*180.0, fill=False, linewidth=1, linestyle='dashed')
        ax.add_patch(gt)
        
        # for j in range(i):
        #     ax.plot(self.traj[j, 0], self.traj[j, 1], 'b.',  markersize=5,  alpha=0.01)
        
        # centroid
        t_c = np.dot(T, self.centroid.T)
        ax.plot(t_c[0], t_c[1], 'k.',  markersize=5)

        # probes
        probe = patches.Circle((self.pusher_position[0][i][0], self.pusher_position[0][i][1]), self.probe_radius, facecolor="red")
        ax.add_patch(probe)
        probe = patches.Circle((self.pusher_position[1][i][0], self.pusher_position[1][i][1]), self.probe_radius, facecolor="red")
        ax.add_patch(probe)
        ax.arrow(self.pusher_position[0][i][0], self.pusher_position[0][i][1], 
            self.direc[0]*0.02, self.direc[1]*0.02, 
            head_width=0.001, head_length=0.01, fc='r', ec='r')
        ax.arrow(self.pusher_position[1][i][0], self.pusher_position[1][i][1], 
            self.direc[0]*0.02, self.direc[1]*0.02, 
            head_width=0.001, head_length=0.01, fc='r', ec='r')

        # 2: plot contact point 
        if self.hasContact[0][i]:
            ax.plot(self.contactPt[0][i][0], self.contactPt[0][i][1], 'ko',  markersize=6)
            ax.arrow(self.contactPt[0][i][0], self.contactPt[0][i][1], 
                self.contactNormal[0][i][0]*0.02, self.contactNormal[0][i][1]*0.02, 
                head_width=0.001, head_length=0.01, fc='y', ec='g')

        if self.hasContact[1][i]:
            ax.plot(self.contactPt[1][i][0], self.contactPt[1][i][1], 'ko',  markersize=6)
            ax.arrow(self.contactPt[1][i][0], self.contactPt[1][i][1], 
                self.contactNormal[1][i][0]*0.02, self.contactNormal[1][i][1]*0.02, 
                head_width=0.001, head_length=0.01, fc='y', ec='g')

        

        plt.xlim(-0.2, 0.2)
        plt.ylim(-0.2, 0.2)
        plt.title('timestamp: ' + str(i))
        fig.canvas.set_window_title(self.shape_id)
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')
        plt.title('Sim timestamp: ' + str(i))
        plt.draw()
        plt.pause(0.001)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", type=str, default="rect1", help="Shape ID (eg: rect1, ellip2, hex)")
    parser.add_argument("--vis", type=int, default="0", help="Visualize 2D pushing")
    parser.add_argument("--wfp", type=int, default="0", help="Plot force profile")
    args = parser.parse_args()
    s = Sim(args.shape, bool(args.vis))
    s.simulate()