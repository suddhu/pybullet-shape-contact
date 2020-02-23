import Sim
import matplotlib.pyplot as plt
import pdb

def main():
    # choose to use visualization - don't if collecting large amounts of data
    withVis = True
    withRand = False

    explorer = Sim.Sim(withVis)

    # push direction with respect to the positive y axis

    # simulate and plot trajectories
    explorer.resetSim(withRandom=withRand)
    traj = explorer.simulate()

if __name__ == "__main__":
    main()
