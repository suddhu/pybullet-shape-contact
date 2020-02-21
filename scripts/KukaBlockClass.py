import pybullet as p
import pybullet_data
import time
import math
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import pdb


class KukaBlock():
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
        self.kukaId = p.loadURDF("kuka_iiwa/model.urdf", [0, 0, 0], useFixedBase=True)

        # reset the base
        p.resetBasePositionAndOrientation(self.kukaId, [0, 0, 0], [0, 0, 0, 1])

        # get useful robot information
        self.kukaEndEffectorIndex = 6
        self.numJoints = p.getNumJoints(self.kukaId)

        # add the block - we'll reset its position later
        self.blockId = p.loadURDF("/home/suddhu/software/pybullet-shape-contact/models/block_big.urdf", [-0.4, 0, .1])
        # p.resetBasePositionAndOrientation(self.blockId, [-0.4, 0, 0.1], [0, 0, 0, 1])

        # reset joint states to nominal pose
        self.rp = [0, 0, 0, 0.5 * math.pi, 0, -math.pi * 0.5 * 0.66, 0]
        for i in range(self.numJoints):
            p.resetJointState(self.kukaId, i, self.rp[i])

        # get the joint ids
        self.jointInds = [i for i in range(self.numJoints)]

        # set gravity
        p.setGravity(0, 0, -10)

        # set simulation length
        self.simLength = 5000

        # set joint damping
        #joint damping coefficents
        self.jd = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

        # set robot init config
        self.robotInitPoseCart = [-0.4, -0.2, 0.01] # (x,y,z)
        self.orn = p.getQuaternionFromEuler([0, -math.pi, 0])

        # pre-define the trajectory/force vectors
        self.traj = np.zeros((self.simLength, 5))
        self.contactPt = np.zeros((self.simLength, 3))
        self.contactForce = np.zeros((self.simLength, ))
        self.contactNormal = np.zeros((self.simLength, 3))
        self.contactCount = np.empty_like(self.contactForce)

        # reset sim time
        self.t = 0

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
                
                self.contactForce[simTime] = f_c_temp
                self.contactPt[simTime, :] =  contactInfo[0][6]
                self.contactNormal[simTime, :] = contactInfo[0][7]
                self.contactCount[simTime] = len(contactInfo)

            self.traj[simTime, :] = np.array([x, y, xb, yb, yaw])

        # contact force mask - get rid of trash in the beginning
        self.contactForce[:300] = 0
        self.contactNormal[:300, :] = 0
        self.contactPt[:300, :] = 0

        return self.traj, self.contactPt, self.contactForce, self.contactNormal

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
    kb = KukaBlock()
    kb.simulate()
