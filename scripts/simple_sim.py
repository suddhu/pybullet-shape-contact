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

STATIC_VELOCITY_THRESHOLD = 1e-4

fig, ax = plt.subplots()
ax.axis('equal')
plt.ion()
plt.show()

colname =  [
  "x of contact position", 
  "y of contact position", 
  "x of contact normal", 
  "y of contact normal", 
  "force magnitude",
  "x of ground truth object pose", 
  "y of ground truth object pose", 
  "yaw of ground truth object pose",
 ]

class Sim():
    def __init__(self, shape_id):

        self.start_time = time.time()

        # connect to pybullet server
        p.connect(p.GUI)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI,0)
        p.resetDebugVisualizerCamera( cameraDistance=0.5, cameraYaw=-30, cameraPitch=-90, cameraTargetPosition=[0,0,0])

        # set additional path to find kuka model
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        self.pusher_pose = [-0.00, -0.05, 0]
        self.origin = [0, 0, 0]

        # set gravity
        p.setGravity(0, 0, -10)

        # set simulation length
        self.limit = 200
        self.threshold = 0.000  # the threshold force for contact, need to be tuned

        # pre-define the trajectory/force vectors
        # pre-define the trajectory/force vectors
        self.traj = np.zeros((self.limit, 3))
        self.contactPt = np.zeros((self.limit, 2))
        self.contactForce = np.zeros((self.limit, ))
        self.contactNormal = np.zeros((self.limit, 2))

        self.t = 0

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
        self.fric = 0.25

        urdf_file = "/home/suddhu/software/pybullet-shape-contact/models/shapes/" + self.shape_id + ".urdf"
        print()
        self.box = p.loadURDF(urdf_file, self.origin)

        p.changeDynamics(self.box, -1, mass=self.shape_mass, lateralFriction=self.fric,
                         localInertiaDiagonal=shape_moment)

        all_dynamics = p.getDynamicsInfo(self.box, -1)
        print('file: ', urdf_file, '\n','mass: ', all_dynamics[0],
             ' lat_fric: ', all_dynamics[1], ' moment of inertia: ', all_dynamics[2],
              ' centroid: ', all_dynamics[3], ' spin_fric: ', all_dynamics[7])

        urdf_file = "/home/suddhu/software/pybullet-shape-contact/models/shapes/pusher.urdf"

        self.pusher = p.loadURDF(urdf_file, self.pusher_pose)
        self.cid = p.createConstraint(self.pusher, -1, -1, -1, p.JOINT_FIXED, [0, 0, 0], [0, 0, 0], self.pusher_pose)

        p.changeDynamics(self.pusher, -1, mass=self.shape_mass, lateralFriction=1.0)

        all_dynamics = p.getDynamicsInfo(self.pusher, -1)

        print('file: ', urdf_file, '\n','mass: ', all_dynamics[0],
             ' lat_fric: ', all_dynamics[1], ' moment of inertia: ', all_dynamics[2],
              ' centroid: ', all_dynamics[3], ' spin_fric: ', all_dynamics[7])

        urdf_file = "/home/suddhu/software/pybullet-shape-contact/models/ground_plane/ground_plane.urdf"

        # add plane to push on (slightly below the base of the robot)
        self.planeId = p.loadURDF(urdf_file, [0, 0, 0], useFixedBase=True)

        p.changeDynamics(self.planeId, -1, lateralFriction=1.0)

        all_dynamics = p.getDynamicsInfo(self.planeId, -1)

        print('shape file: ', urdf_file, '\n','mass: ', all_dynamics[0],
             ' lat_fric: ', all_dynamics[1], ' moment of inertia: ', all_dynamics[2],
              ' centroid: ', all_dynamics[3], ' spin_fric: ', all_dynamics[7])

        # input('Click Enter!')

    def observe_block(self, blockID):
        blockPose = p.getBasePositionAndOrientation(blockID)
        xb = blockPose[0][0]
        yb = blockPose[0][1]
        roll, pitch, yaw = p.getEulerFromQuaternion(blockPose[1])
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
        alpha = .2
        self.simTime = 0
        all_contact = []

        filename = 'box_push'
        dir_base = "/home/suddhu/software/pybullet-shape-contact/data/simple_push"
        jsonfilename = dir_base+'/%s.json' % filename
        p.setTimeStep(1. / 240.)

        delX = np.array([1e-3, 1e-3, 0.0])
        while True:
            # time.sleep(1./240.)

            pusher_pos = self.observe_block(self.pusher)
            pusher_pos = pusher_pos + delX
            # force = alpha * direc #(box_pos - pusher_pos)
            # p.resetBaseVelocity(self.pusher, linearVelocity=[-0.05, 0, 0])
            p.changeConstraint(self.cid, np.append(pusher_pos[0:2], 0.03), maxForce=200)

            # limitForce = self.shape_mass*self.fric*10
            # local_force = np.array([-3*limitForce, 0, 0])
            # local_push_pt = np.array([0.045, 0.045/3.0, 0.00])
            # push_pt = transform_back_frame2d(local_push_pt[0:2], box_pos)
            # force = transform_back_frame2d(local_force[0:2], box_pos)

            # p.applyExternalForce(objectUniqueId=self.box, linkIndex=-1,
            #                     forceObj=np.append(force, 0.0), posObj=local_push_pt, flags=p.LINK_FRAME)


            p.stepSimulation()

            contactInfo = p.getContactPoints(self.box, self.pusher)
            box_pos = self.observe_block(self.box)

            # while not self.static_environment():
            #     # pdb.set_trace()
            #     for _ in range(10):
            #         # time.sleep(1./240.)
            #         p.stepSimulation()

            # self.contactForce[self.simTime] = np.linalg.norm(force)
            # self.contactPt[self.simTime, :] =  push_pt[0:2]
            # self.contactNormal[self.simTime, :] =  force[0:2] /np.linalg.norm(force[0:2])
            # self.traj[self.simTime, :] = box_pos

            # all_contact.append(
            # self.contactPt[self.simTime, 0:2].tolist() + 
            # self.contactNormal[self.simTime, 0:2].tolist() + 
            # [self.contactForce[self.simTime]] + 
            # self.traj[self.simTime, :].tolist())

            # self.plotter(self.simTime)

            # print('contact pt: ', self.contactPt[self.simTime, :])
            # print(len(all_contact), ' Applied force magnitude = {}'.format(f_c_temp))
            # print(len(all_contact), ' Applied force vector = {}'.format(np.linalg.norm(f_c_temp)))
            # self.simTime = self.simTime + 1
            

            f_c_temp = 0

            # get the net contact force between robot and block
            if len(contactInfo)>0:
                # print("contact!")
                for c in range(len(contactInfo)):
                    f_c_temp += contactInfo[c][9]
                
                if f_c_temp > self.threshold:
                    # print("f_c_temp: ", f_c_temp)
                    self.contactForce[self.simTime] = f_c_temp
                    self.contactPt[self.simTime, :] =  contactInfo[0][5][:2]
                    self.contactNormal[self.simTime, :] = contactInfo[0][7][:2]
                    print(self.contactNormal[self.simTime, :])
                    self.traj[self.simTime, :] = box_pos

                    all_contact.append(
                    self.contactPt[self.simTime, 0:2].tolist() + 
                    self.contactNormal[self.simTime, 0:2].tolist() + 
                    [self.contactForce[self.simTime]] + 
                    self.traj[self.simTime, :].tolist())

                    self.plotter(self.simTime)
                    print('contact pt: ', self.contactPt[self.simTime, :])
                    print(len(all_contact), ' Applied force magnitude = {}'.format(f_c_temp))
                    print(len(all_contact), ' Applied force vector = {}'.format(np.linalg.norm(contactInfo[0][7][:2])))
                    print('penet: ', contactInfo[0][8])
                    self.simTime = self.simTime + 1
                    # pdb.set_trace()

            # 3.5 break if we collect enough
            if len(all_contact) == self.limit:
                break
                


        with open(jsonfilename, 'w') as outfile:
            json.dump({'all_contacts': all_contact[5:],
                '__title__': colname, 
                    "shape_id": self.shape_id,
                    "limit": self.limit}, outfile, sort_keys=True, indent=1)
                    
        return

    def plotter(self, i): 
        ax.clear()
        # 1: plot object
        xb = self.traj[i][0]
        yb = self.traj[i][1]
        # t = wraptopi(self.traj[i][4])
        t = self.traj[i][2]
        T = matrix_from_xyzrpy([xb, yb, 0], [0, 0, t])

        # ground truth shape
        if self.shape_type == 'poly' or self.shape_type == 'polyapprox':
            shape_polygon_3d_world = np.dot(T, self.shape_polygon_3d.T)
            gt = patches.Polygon(shape_polygon_3d_world.T[:,0:2], closed=True, linewidth=2, linestyle='dashed', fill=False)
        elif self.shape_type == 'ellip':
            scale, shear, angles, trans, persp = tfm.decompose_matrix(T)
            gt = patches.Ellipse(trans[0:2], self.shape[0]*2, self.shape[1]*2, angle=angles[2]/np.pi*180.0, fill=False, linewidth=1, linestyle='dashed')
        ax.add_patch(gt)

        
        for j in range(i):
            ax.plot(self.traj[j, 0], self.traj[j, 1], 'b.',  markersize=5)

        # probe
        # probe = patches.Circle((self.traj[i][0], self.traj[i][1]), self.probe_radius, facecolor="black", alpha=0.4)
        # ax.add_patch(probe)

        # contact points, normals
        # 2: plot contact point 
        ax.plot(self.contactPt[i][0], self.contactPt[i][1], 'rX',  markersize=12)

        # 3: plot contact normal
        ax.arrow(self.contactPt[i][0], self.contactPt[i][1], 
            self.contactNormal[i][0]*0.05, self.contactNormal[i][1]*0.05, 
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
    s = Sim("rect1")
    s.simulate()
