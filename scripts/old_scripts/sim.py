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
from ik.helper import wraptopi, matrix_from_xyzrpy
from config.shape_db import ShapeDB
import tf.transformations as tfm
import json, os
import subprocess, glob

fig, ax = plt.subplots()
ax.axis('equal')
plt.ion()
plt.show()

colname =  [
  "x of contact position", 
  "y of contact position", 
  "z of contact position", 
  "x of contact normal", 
  "y of contact normal", 
  "z of contact normal", 
  "force magnitude",
  "x of pusher position", 
  "y of pusher position", 
  "z of pusher position",
  "x of ground truth object pose", 
  "y of ground truth object pose", 
  "z of ground truth object pose", 
  "yaw of ground truth object pose",
 ]

class Sim():
    def __init__(self, shape_id, withVis=True, withVid=False):

        self.start_time = time.time()
        self.record = withVid

        ## pybullet setup
        if withVis:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI,0)
        p.resetDebugVisualizerCamera( cameraDistance=1.0, cameraYaw=-30, cameraPitch=-50, cameraTargetPosition=[0,0,0])
        # set gravity
        p.setGravity(0, 0, -9.8)
        # set additional path to find kuka model
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        self.center_world = [-0.4, 0, 0]
        self.shape_id = shape_id
        shape_db = ShapeDB()
        shape = shape_db.shape_db[self.shape_id]['shape'] # shape of the objects presented as polygon.
        self.shape_type = shape_db.shape_db[self.shape_id]['shape_type']
        shape_moment = [1e-3, 1e-3, shape_db.shape_db[self.shape_id]['moment_of_inertia']]
        shape_mass = shape_db.shape_db[self.shape_id]['mass']

        if self.shape_type == 'poly':
            self.shape_polygon_3d = np.hstack((np.array(shape), np.zeros((len(shape), 1)), np.ones((len(shape), 1))))
        elif self.shape_type == 'ellip':
            self.shape = shape[0]
        elif self.shape_type == 'polyapprox':
            self.shape_polygon_3d = np.hstack((np.array(shape[0]), np.zeros((len(shape[0]), 1)), np.ones((len(shape[0]), 1))))

        fric = 0.25

        ## load URDFs
        # add plane to push on
        self.planeId = p.loadURDF("/home/suddhu/software/pybullet-shape-contact/models/ground_plane/ground_plane.urdf", [0, 0, -0.05], useFixedBase=True)
        p.changeDynamics(self.planeId, -1, lateralFriction=fric)
        # add the robot at the origin with fixed base
        self.kukaId = p.loadURDF("/home/suddhu/software/pybullet-shape-contact/models/kuka_iiwa/model.urdf", [0, 0, 0.0], useFixedBase=True)
        # add block
        self.blockId = p.loadURDF("/home/suddhu/software/pybullet-shape-contact/models/shapes/" + self.shape_id + ".urdf", self.center_world)
        p.changeDynamics(self.blockId, -1, mass=shape_mass, lateralFriction=fric, localInertiaDiagonal=shape_moment)
        all_dynamics = p.getDynamicsInfo(self.blockId, -1)
        print('shape: ', self.shape_id , '\n','mass: ', all_dynamics[0],
             ' lat_fric: ', all_dynamics[1], ' moment of inertia: ', all_dynamics[2],
              ' centroid: ', all_dynamics[3], ' spin_fric: ', all_dynamics[7])

        # pdb.set_trace()
        self.centroid = np.hstack((np.array(all_dynamics[3]), 1))

        # get useful robot information
        self.kukaEndEffectorIndex = 7
        self.numJoints = p.getNumJoints(self.kukaId)
        self.block_level = 0.03

        # reset the base
        p.resetBasePositionAndOrientation(self.kukaId, [0, 0, 0.0], [0, 0, 0, 1])
        # reset joint states to nominal pose
        #joint damping coefficents
        self.jd = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
        self.orn = p.getQuaternionFromEuler([0, math.pi, 0]) ## ee orientation
        self.rp = [0, 0, 0, 0.5 * math.pi, 0, -math.pi * 0.5 * 0.66, 0.5*math.pi, 0]
        for i in range(self.numJoints):
            p.resetJointState(self.kukaId, i, self.rp[i])

        ## sim params
        self.simLength = 10000
        self.limit = 500
        self.step_size = 0.002
        self.explore_radius = 0.20
        self.init_prods = 1
        if self.init_prods > 1 and self.init_prods % 2 == 1:  # correct it if not even nor 1
            self.init_prods += 1

        # populate start poses for small probing in opposite directions
        self.start_configs = []
        # generate the start poses 
        jmax = (self.init_prods+1) // 2
        kmax = 2 if self.init_prods > 1 else 1
        dth = 2*np.pi / self.init_prods
        for j in range(jmax):
            for k in range(kmax):
                i = j + k * (self.init_prods//2)
                start_pos = [self.center_world[0] + self.explore_radius*np.cos(i*dth), 
                                self.center_world[1] + self.explore_radius*np.sin(i*dth)]
                direc = [-np.cos(i*dth), -np.sin(i*dth)]
                self.start_configs.append([start_pos, direc ])

        eePos = self.start_configs[0][0] + [self.block_level]
        # self.moveToPos(eePos) 
        self.cid = p.createConstraint(self.kukaId, self.kukaEndEffectorIndex, -1, -1, p.JOINT_FIXED, [0, 0, 0], [0, 0, 0], eePos, self.orn)
        
        print(eePos)
        for _ in range(10000):
            p.stepSimulation()


        # pre-define vectors
        self.traj = np.zeros((self.simLength, 5))
        self.contactPt = np.zeros((self.simLength, 2))
        self.contactForce = np.zeros((self.simLength, ))
        self.contactNormal = np.zeros((self.simLength, 2))
        self.scan_contact_pts = []

        self.threshold = 0.0  # the threshold force for contact, need to be tuned
        self.probe_radius = 0.010

        input('Click Enter!')

    def plotter(self, i): 
        ax.clear()
        # 1: plot object
        xb = self.traj[i][2]
        yb = self.traj[i][3]
        # t = wraptopi(self.traj[i][4])
        t = self.traj[i][4]
        T = matrix_from_xyzrpy([xb, yb, 0], [0, 0, t])

        # ground truth shape
        if self.shape_type == 'poly' or self.shape_type == 'polyapprox':
            shape_polygon_3d_world = np.dot(T, self.shape_polygon_3d.T)
            gt = patches.Polygon(shape_polygon_3d_world.T[:,0:2], closed=True, linewidth=2, linestyle='dashed', fill=False)
        elif self.shape_type == 'ellip':
            scale, shear, angles, trans, persp = tfm.decompose_matrix(T)
            gt = patches.Ellipse(trans[0:2], self.shape[0]*2, self.shape[1]*2, angle=angles[2]/np.pi*180.0, fill=False, linewidth=1, linestyle='dashed')
        ax.add_patch(gt)

        # centroid
        t_c = np.dot(T, self.centroid.T)
        ax.plot(t_c[0], t_c[1], 'k.',  markersize=5)

        # probe
        probe = patches.Circle((self.traj[i][0], self.traj[i][1]), self.probe_radius, facecolor="black", alpha=0.4)
        ax.add_patch(probe)

        # contact points, normals
        if (self.contactPt[i][0] != 0) and (self.contactPt[i][1] != 0):                 
            # 2: plot contact point 
            ax.plot(self.contactPt[i][0], self.contactPt[i][1], 'rX',  markersize=12)

            # 3: plot contact normal
            ax.arrow(self.contactPt[i][0], self.contactPt[i][1], 
                self.contactNormal[i][0]*0.05, self.contactNormal[i][1]*0.05, 
                head_width=0.001, head_length=0.01, fc='y', ec='g')

            # 4: plot contour direction
            ax.arrow(self.contactPt[i][0], self.contactPt[i][1], 
                self.direc[0]*0.01, self.direc[1]*0.01, 
                fc='y', ec='y', ls='-')
                
        plt.xlim(self.traj[0][2] - self.explore_radius, self.traj[0][2] +  self.explore_radius)
        plt.ylim(self.traj[0][3] - self.explore_radius, self.traj[0][3] +  self.explore_radius)
        plt.title('timestamp: ' + str(i))
        fig.canvas.set_window_title(self.shape_id)
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')
        plt.title('Sim timestamp: ' + str(i) + '          '  + '# contacts: ' + str(len(self.scan_contact_pts)))
        plt.draw()
        plt.pause(0.001)

        # record and save
        if self.record: 
            plt.savefig(self.plotPath + "/%03d.png" % i)


    def moveToPos(self, pos): 
        # p.stepSimulation()
        plt.pause(0.001)
        # compute the inverse kinematics
        jointPoses = p.calculateInverseKinematics(self.kukaId,
                                                self.kukaEndEffectorIndex,
                                                pos,
                                                self.orn,
                                                jointDamping=self.jd)
        for k in range(self.numJoints):
            p.setJointMotorControl2(bodyIndex=self.kukaId,
                                jointIndex=k,
                                controlMode=p.POSITION_CONTROL,
                                targetPosition=jointPoses[k],
                                targetVelocity=0)
                                
    def simulate(self):
        
        num = 1
        filename = 'all_contact_shape=%s_rep=%04d' % (self.shape_id, num)
        dir_base = "/home/suddhu/software/pybullet-shape-contact/data/contour_following"
        jsonfilename = dir_base+'/%s.json' % filename
        while os.path.isfile(jsonfilename):
            num = num + 1
            filename = 'all_contact_shape=%s_rep=%04d' % (self.shape_id, num)
            jsonfilename = dir_base+'/%s.json' % filename

        if self.record:
            # start logging
            vid_base = "/home/suddhu/software/pybullet-shape-contact/data/video/"
            os.mkdir(vid_base + filename)
            self.vidPath = vid_base + filename + "/sim.mp4"
            logId = p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, self.vidPath)
            print("logging video: ", self.vidPath)

            self.plotPath = vid_base + filename
            print("logging plot: ", self.plotPath)


        all_contact = []
        self.direc = [0, 0]

        self.simTime = 0
        ## --------------------
        ## Rough probing
        ## --------------------
        for i, (start_pos, self.direc) in enumerate(reversed(self.start_configs)):
            # ax.clear()
            curr_pos = start_pos
            
            for w in range(self.numJoints):
                p.resetJointState(self.kukaId, w, self.rp[w])

            j = 0
            path = []
            while True:
                # move along rough probe
                curr_pos = (np.array(curr_pos) + np.array(self.direc) * self.step_size).tolist()

                eePos = curr_pos + [self.block_level]
                path.append(eePos)
                self.moveToPos(eePos) 

                # get joint states
                blockPose = p.getBasePositionAndOrientation(self.blockId)
                xb = blockPose[0][0]
                yb = blockPose[0][1]
                _, _, yaw = p.getEulerFromQuaternion(blockPose[1])

                # get contact information
                contactInfo = p.getContactPoints(self.kukaId, self.blockId)

                f_c_temp = 0

                # get the net contact force between robot and block
                if len(contactInfo)>0:
                    for c in range(len(contactInfo)):
                        f_c_temp += contactInfo[c][9]
                    
                    # print("f_c_temp: ", f_c_temp)
                    self.contactForce[self.simTime] = f_c_temp
                    self.contactPt[self.simTime, :] =  contactInfo[0][5][:2]
                    self.contactNormal[self.simTime, :] = contactInfo[0][7][:2]
                    self.scan_contact_pts.append(contactInfo[0][5])

                self.traj[self.simTime, :] = np.array([curr_pos[0], curr_pos[1], xb, yb, yaw])

                # plot
                if (self.simTime % 1 == 0):
                    self.plotter(self.simTime)
                    
                # If in contact, break
                if f_c_temp > self.threshold: 
                    all_contact.append(
                    self.contactPt[self.simTime, 0:2].tolist() + [0] + 
                    self.contactNormal[self.simTime, 0:2].tolist() + [0] + 
                    [self.contactForce[self.simTime]] + 
                    self.traj[self.simTime, 0:2].tolist() + [0] +
                    self.traj[self.simTime, 2:4].tolist() + [0] + 
                    [self.traj[self.simTime, 4]])

                    self.simTime = self.simTime + 1
                    revpath =  path[-len(path)//10:]

                    # if the last one, stay in contact and do exploration from there.
                    if i == (len(self.start_configs) - 1):
                        break

                    for rev in reversed(revpath):
                        self.moveToPos(rev) 
                    break

                # #if too far and no contact break.
                if j > self.explore_radius*2/self.step_size:
                    self.simTime = self.simTime + 1
                    break

                # increment counters
                j = j + 1
                self.simTime = self.simTime + 1

        if len(self.scan_contact_pts) == 0:
            print("Error: Cannot touch the object")
            return


        good_normal = self.contactNormal[self.simTime - 1]
        self.direc = np.dot(tfm.euler_matrix(0,0,2) , good_normal.tolist() + [0] + [1])[0:2]

        ## --------------------
        ## Contour following, use the normal to move along the block
        ## --------------------
        while True:
            # 3.1 move 
            # pdb.set_trace()
            angle = 2

            curr_pos = (np.array(curr_pos) + np.array(self.direc) * self.step_size).tolist()
            
            eePos = curr_pos + [self.block_level]
            self.moveToPos(eePos) 

            # get joint states
            blockPose = p.getBasePositionAndOrientation(self.blockId)
            xb = blockPose[0][0]
            yb = blockPose[0][1]
            _, _, yaw = p.getEulerFromQuaternion(blockPose[1])
            
            # get contact information
            contactInfo = p.getContactPoints(self.kukaId, self.blockId)

            if (np.linalg.norm(self.contactPt[self.simTime, :] - self.contactPt[self.simTime - 1, :]) < 0.01):
                angle = 1

            # get the net contact force between robot and block
            if len(contactInfo)>0:
                f_c_temp = 0
                for c in range(len(contactInfo)):
                    f_c_temp += contactInfo[c][9]
                
                self.contactForce[self.simTime] = f_c_temp
                self.contactPt[self.simTime, :] =  contactInfo[0][5][:2]
                self.contactNormal[self.simTime, :] = contactInfo[0][7][:2]
                good_normal = self.contactNormal[self.simTime, :]
                self.direc = np.dot(tfm.euler_matrix(0,0,angle) , good_normal.tolist() + [0] + [1])[0:2]
                self.traj[self.simTime, :] = np.array([curr_pos[0], curr_pos[1], xb, yb, yaw])

                if f_c_temp > self.threshold:
                    # print("f_c_temp: ", f_c_temp)
                    self.scan_contact_pts.append(contactInfo[0][5])   
                    # pdb.set_trace()   
                    all_contact.append(
                    self.contactPt[self.simTime, 0:2].tolist() + [0] + 
                    self.contactNormal[self.simTime, 0:2].tolist() + [0] + 
                    [self.contactForce[self.simTime]] + 
                    self.traj[self.simTime, 0:2].tolist() + [0] +
                    self.traj[self.simTime, 2:4].tolist() + [0] + 
                    [self.traj[self.simTime, 4]])


            # plot
            if (self.simTime % 1 == 0):
                self.plotter(self.simTime)
                
            # increment counters
            self.simTime = self.simTime + 1

            # 3.5 break if we collect enough
            if len(all_contact) == self.limit:
                break

        with open(jsonfilename, 'w') as outfile:
            json.dump({'all_contacts': all_contact,
                       '__title__': colname, 
                         "shape_id": self.shape_id,
                         "probe_radius": self.probe_radius,
                         "offset": self.center_world, 
                         "limit": self.limit}, outfile, sort_keys=True, indent=1)

        # convert to mp4 and delete images
        if self.record:
            p.stopStateLogging(logId)
            os.chdir(self.plotPath)
            subprocess.call([
            'ffmpeg', '-framerate', '8', '-i', '%03d.png', '-r', '30', '-pix_fmt', 'yuv420p',
            'plot.mp4'])

            for file_name in glob.glob("*.png"):
                os.remove(file_name)

        return

    # reset robot to nominal pose
    def resetSim(self):
        for i in range(self.numJoints):
            p.resetJointState(self.kukaId, i, self.rp[i])

        p.resetBasePositionAndOrientation(self.blockId, [-0.4, 0, 0.0], [0, 0, 0, 1])

if __name__ == "__main__":
    s = Sim()
    s.simulate()
