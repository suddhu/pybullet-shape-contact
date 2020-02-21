import KukaBlockClass
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pdb
import math

def plotter(traj, pt, contacts, normals): 

    plt.plot(traj[2][:], traj[3][:], 'r.')


def main():
    # choose to use visualization - don't if collecting large amounts of data
    withVis = True
    kb = KukaBlockClass.KukaBlock(withVis)

    # number of simulations to run, use 1 with visualization
    simRuns = 5

    # push direction with respect to the positive y axis
    # used only if random init is off
    theta = 0.5
    withRand = True

    # plotting utility
    axis_name = ['x robot (m)', 'y robot (m)', 'x block (m)', 'y block (m)', 'block rotation (rads)']
    fig = plt.figure()

    # simulate and plot trajectories
    for simCount in range(simRuns):
        # simulation resets everytime it's called
        kb.resetSim(withRandom=withRand)
        traj, pt, f, n = kb.simulate(theta)

        plt.clf()
        plotter(traj, pt, f, n)
        # plt.xlim(-0.5, 0.5)
        # plt.ylim(-0.5, 0.5)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.draw()
        plt.pause(0.1)




if __name__ == "__main__":
    main()
