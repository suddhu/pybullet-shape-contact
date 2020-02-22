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

fig, ax = plt.subplots()
ax.axis('equal')
plt.ion()
plt.show()

class Sim():
    def __init__(self, withVis=True):
        # connect to pybullet server
        if withVis:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)

        # set additional path to find kuka model
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # add plane to push on (slightly below the base of the robot)
        self.planeId = p.loadURDF("plane.urdf", [0, 0, -0.05], useFixedBase=True)

        # add the robot at the origin with fixed base
        self.kukaId = p.loadURDF("/home/suddhu/software/pybullet-shape-contact/models/kuka_iiwa/model.urdf", [0, 0, 0], useFixedBase=True)

        # reset the base
        p.resetBasePositionAndOrientation(self.kukaId, [0, 0, 0], [0, 0, 0, 1])

        # get useful robot information
        self.kukaEndEffectorIndex = 7
        self.numJoints = p.getNumJoints(self.kukaId)

        # add the block - we'll reset its position later
        self.blockId = p.loadURDF("/home/suddhu/software/pybullet-shape-contact/models/block_big.urdf", [-0.4, 0, .1])
        # p.resetBasePositionAndOrientation(self.blockId, [-0.4, 0, 0.1], [0, 0, 0, 1])

        # reset joint states to nominal pose
        self.rp = [0, 0, 0, 0.5 * math.pi, 0, -math.pi * 0.5 * 0.66, 0, 0]
        for i in range(self.numJoints):
            p.resetJointState(self.kukaId, i, self.rp[i])

        # get the joint ids
        self.jointInds = [i for i in range(self.numJoints)]

        # set gravity
        p.setGravity(0, 0, -9.8)

        # set simulation length
        self.simLength = 5000

        #joint damping coefficents
        self.jd = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

        # set robot init config: start moving from here
        self.robotInitPoseCart = [-0.4, -0.2, 0.01] # (x,y,z)
        self.orn = p.getQuaternionFromEuler([0, -math.pi, 0])

        # pre-define the trajectory/force vectors
        self.traj = np.zeros((self.simLength, 5))
        self.contactPt = np.zeros((self.simLength, 3))
        self.contactForce = np.zeros((self.simLength, ))
        self.contactNormal = np.zeros((self.simLength, 3))

        # reset sim time
        self.t = 0

        self.shape_id = 'rect0'
        shape_db = ShapeDB()
        shape = shape_db.shape_db[self.shape_id]['shape'] # shape of the objects presented as polygon.
        self.shape_type = shape_db.shape_db[self.shape_id]['shape_type']
        if self.shape_type == 'poly':
            self.shape_polygon_3d = np.hstack((np.array(shape), np.zeros((len(shape), 1)), np.ones((len(shape), 1))))
        elif self.shape_type == 'ellip':
            self.shape = shape[0]
        elif self.shape_type == 'polyapprox':
            self.shape_polygon_3d = np.hstack((np.array(shape[0]), np.zeros((len(shape[0]), 1)), np.ones((len(shape[0]), 1))))
    
    def plotter(self, i): 
        ax.clear()
        # 1: plot object
        xb = self.traj[i][2]
        yb = self.traj[i][3]
        t = self.traj[i][4]

        T = matrix_from_xyzrpy([xb, yb, 0], [0, 0, t])

        if self.shape_type == 'poly' or self.shape_type == 'polyapprox':
            shape_polygon_3d_world = np.dot(T, self.shape_polygon_3d.T)
            obj = patches.Polygon(shape_polygon_3d_world.T[:,0:2], closed=True, linewidth=2, linestyle='dashed', fill=False)
        elif self.shape_type == 'ellip':
            scale, shear, angles, trans, persp = tfm.decompose_matrix(T)
            obj = patches.Ellipse(trans[0:2], self.shape[0]*2, self.shape[1]*2, angle=angles[2]/np.pi*180.0, fill=False, linewidth=1, linestyle='solid')
        ax.add_patch(obj)

        circle = patches.Circle((self.traj[i][0], self.traj[i][1]), 0.020, facecolor="black", alpha=0.4)
        ax.add_patch(circle)

        # 2: plot contact point 
        ax.plot(self.contactPt[i][0], self.contactPt[i][1], 'g*')

        # 3: plot contact normal
        ax.arrow(self.contactPt[i][0], self.contactPt[i][1], 
            self.contactNormal[i][0]*0.01, self.contactNormal[i][1]*0.01, 
            head_width=0.001, head_length=0.01, fc='y', ec='g')

        plt.xlim(self.traj[0][2] - 0.25 , self.traj[0][2] + 0.25)
        plt.ylim(self.traj[0][3] - 0.25 , self.traj[0][3] + 0.25)
        plt.title(str(i))
        plt.draw()
        plt.pause(0.01)
        # input("Press [enter] to continue.")
        obj.remove()
        circle.remove()

    def simulate(self, theta=0.1):
        
        pushDir = self.getPushDir(theta)
        pushStep = self.getPushStep()

        x = self.robotInitPoseCart[0]
        y = self.robotInitPoseCart[1]
        z = self.robotInitPoseCart[2]

        for simTime in range(self.simLength):
            p.stepSimulation()

            # set end effector pose
            d = pushDir * pushStep
            x += d[0]
            y += d[1]
            z += d[2]
            eePos = [x, y, z]

            # compute the inverse kinematics
            jointPoses = p.calculateInverseKinematics(self.kukaId,
                                                      self.kukaEndEffectorIndex,
                                                      eePos,
                                                      self.orn,
                                                      jointDamping=self.jd)
            for i in range(self.numJoints):
                p.setJointMotorControl2(bodyIndex=self.kukaId,
                                      jointIndex=i,
                                      controlMode=p.POSITION_CONTROL,
                                      targetPosition=jointPoses[i],
                                      targetVelocity=0,
                                      force=500,
                                      positionGain=0.3,
                                      velocityGain=1)

            # get joint states
            ls = p.getLinkState(self.kukaId, self.kukaEndEffectorIndex)
            blockPose = p.getBasePositionAndOrientation(self.blockId)
            xb = blockPose[0][0]
            yb = blockPose[0][1]
            roll, pitch, yaw = p.getEulerFromQuaternion(blockPose[1])

            # get contact information
            contactInfo = p.getContactPoints(self.kukaId, self.blockId)

            # get the net contact force between robot and block
            if len(contactInfo)>0:
                f_c_temp = 0

                for i in range(len(contactInfo)):
                    f_c_temp += contactInfo[i][9]
                
                print("f_c_temp: ", f_c_temp)
                self.contactForce[simTime] = f_c_temp
                self.contactPt[simTime, :] =  contactInfo[0][5]
                self.contactNormal[simTime, :] = contactInfo[0][7]

            self.traj[simTime, :] = np.array([x, y, xb, yb, yaw])

            if (simTime % 100 == 0):
                self.plotter(simTime)

        # contact force mask - get rid of trash in the beginning
        self.contactForce[:300] = 0
        self.contactNormal[:300, :] = 0
        self.contactPt[:300, :] = 0
        return self.traj

    def resetSim(self, withRandom):
        # reset robot to nominal pose
        for i in range(self.numJoints):
            p.resetJointState(self.kukaId, i, self.rp[i])

        # reset block pose
        if withRandom:
            # define nominal block pose
            nom_pose = np.array([-0.4, 0.0, 0.0]) # (x,y,theta)

            # define uncertainty bounds
            pos_bd = np.array([0.01, 0.01, 0.1])

            # initialize array
            blockInitPose = np.empty_like(pos_bd)

            for i in range(nom_pose.shape[0]):
                pert = np.random.uniform(-pos_bd[i], pos_bd[i])
                blockInitPose[i] = nom_pose[i] + pert

            blockInitOri = p.getQuaternionFromEuler([0, 0, blockInitPose[-1]])
            p.resetBasePositionAndOrientation(self.blockId, [blockInitPose[0], blockInitPose[1], 0.1], blockInitOri)
        else:
            p.resetBasePositionAndOrientation(self.blockId, [-0.4, 0, 0.1], [0, 0, 0, 1])


    def getPushDir(self, theta):
        # get unit direction of the push
        return np.array([math.sin(theta), math.cos(theta), 0.])

    def getPushStep(self):
        return 0.00005


if __name__ == "__main__":
    s = Sim()
    s.simulate()
