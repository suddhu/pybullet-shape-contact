import logging
import math
import pybullet as p
import time

import numpy as np
import pybullet_data
from matplotlib import pyplot as plt

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG,
                    format='[%(levelname)s %(asctime)s %(pathname)s:%(lineno)d] %(message)s',
                    datefmt='%m-%d %H:%M:%S')
logging.getLogger('matplotlib.font_manager').disabled = True


def rotate_wrt_origin(xy, theta):
    return (xy[0] * math.cos(theta) - xy[1] * math.sin(theta),
            xy[0] * math.sin(theta) + xy[1] * math.cos(theta))


def angular_diff(a, b):
    """Angle difference from b to a (a - b)"""
    d = a - b
    if d > math.pi:
        d -= 2 * math.pi
    elif d < -math.pi:
        d += 2 * math.pi
    return d


def get_dx(px, cx):
    dpos = cx[:2] - px[:2]
    dyaw = angular_diff(cx[2], px[2])
    dx = np.r_[dpos, dyaw]
    return dx


def dx_to_dz(px, dx):
    dz = np.zeros_like(dx)
    # dyaw is the same
    dz[2] = dx[2]
    dz[:2] = rotate_wrt_origin(dx[:2], px[2])
    return dz


init_block_pos = [0.0, 0.0]
init_block_yaw = -0.

physics_client = p.connect(p.GUI)
p.setTimeStep(1. / 240.)
p.setRealTimeSimulation(False)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

blockId = p.loadURDF("./block_big.urdf", tuple(init_block_pos) + (-0.02,),
                     p.getQuaternionFromEuler([0, 0, init_block_yaw]))
planeId = p.loadURDF("plane.urdf", [0, 0, -0.05], useFixedBase=True)
p.resetDebugVisualizerCamera(cameraDistance=0.5, cameraYaw=0, cameraPitch=-85,
                             cameraTargetPosition=[0, 0, 1])

STATIC_VELOCITY_THRESHOLD = 1e-6


def _observe_block(blockId):
    blockPose = p.getBasePositionAndOrientation(blockId)
    xb = blockPose[0][0]
    yb = blockPose[0][1]
    roll, pitch, yaw = p.getEulerFromQuaternion(blockPose[1])
    return np.array((xb, yb, yaw))


def _static_environment():
    v, va = p.getBaseVelocity(blockId)
    if (np.linalg.norm(v) > STATIC_VELOCITY_THRESHOLD) or (
            np.linalg.norm(va) > STATIC_VELOCITY_THRESHOLD):
        return False
    return True


p.setGravity(0, 0, -10)
p.changeDynamics(blockId, -1, lateralFriction=1.0)
p.changeDynamics(planeId, -1, lateralFriction=0.5)
F = 6.0*10.0*0.5
MAX_ALONG = 0.075 + 0.2

for _ in range(100):
    p.stepSimulation()

N = 100
yaws = np.zeros(N)
z_os = np.zeros((N, 3))
for simTime in range(N):
    # observe difference from pushing
    px = _observe_block(blockId)
    yaws[simTime] = px[2]
    p.applyExternalForce(blockId, -1, [-F, 0, 0], [0.55/2, 0, 0.025], p.LINK_FRAME)
    p.stepSimulation()
    while not _static_environment():
        for _ in range(100):
            p.stepSimulation()
    cx = _observe_block(blockId)
    # difference in world frame
    dx = get_dx(px, cx)
    dz = dx_to_dz(px, dx)
    z_os[simTime] = dz
    logger.info("dx %s dz %s", dx, dz)
    time.sleep(0.1)
logger.info(z_os.std(0) / np.abs(np.mean(z_os, 0)))
plt.scatter(yaws, z_os[:, 2])
plt.xlabel('yaw')
plt.ylabel('dyaw')
plt.show()