import numpy as np
import matplotlib.pyplot as plt
from ik.helper import norm
import tf.transformations as tfm
import pdb 

fig, ax = plt.subplots()
ax.axis('equal')
plt.ion()
plt.show()


nstart = 10
if nstart > 1 and nstart % 2 == 1:  # correct it if not even nor 1
    nstart += 1
    
step_size = 0.002
explore_radius = 0.12
center_world = [0.350, -0.03, 0]
limit = 10000  # number of data points to be collected

# 2. Rough probing
scan_contact_pts = []

# 2.1 populate start poses for small probing in opposite direction
start_configs = []

jmax = (nstart+1) // 2
kmax = 2 if nstart > 1 else 1
dth = 2*np.pi / nstart

print(jmax, kmax)
for j in range(jmax):
    for k in range(kmax):
        i = j + k * (nstart//2)
        start_pos = [center_world[0] + explore_radius*np.cos(i*dth), 
                        center_world[1] + explore_radius*np.sin(i*dth)]
        direc = [-np.cos(i*dth), -np.sin(i*dth)]
        start_configs.append([ start_pos, direc ])
        print('direc', direc)


print('start_configs: ', start_configs)

# each rough probe
for i, (start_pos, direc) in enumerate(reversed(start_configs)):
    # ax.clear()
    curr_pos = start_pos

    j = 0
    while True:
        # move along rough probe
        curr_pos = np.array(curr_pos) + np.array(direc) * step_size

        ax.plot(curr_pos[0], curr_pos[1], 'g*')
        ax.arrow(curr_pos[0], curr_pos[1], direc[0]*0.01, direc[1]*0.01, head_width=0.001, head_length=0.01, fc='y', ec='b')
        plt.draw()
        plt.pause(0.1)
        plt.title(str(i))

        # If in contact, break
        if j > 10: 
            # pdb.set_trace()
            normal = [direc, 0]
            scan_contact_pts.append(curr_pos)
            break

        #if too far and no contact break.
        j = j + 1

if len(scan_contact_pts) == 0:
    print("Error: Cannot touch the object")


# 3. Contour following, use the normal to move along the block
while True: 
    direc = np.dot(tfm.euler_matrix(0,0,2) , normal)
    curr_pos = np.array(curr_pos) + direc * step_size

    # 3.4 record the data if in contact
    ax.plot(curr_pos[0], curr_pos[1], 'g*')
    ax.arrow(curr_pos[0], curr_pos[1], direc[0]*0.01, direc[1]*0.01, head_width=0.001, head_length=0.01, fc='y', ec='b')
    plt.draw()
    plt.pause(0.1)
    plt.title(str(i))

    # 3.5 break if we collect enough
    # if len(contact_pts) > limit:
    #     break