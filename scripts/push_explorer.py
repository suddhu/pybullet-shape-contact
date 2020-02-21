import Sim
import matplotlib.pyplot as plt
import pdb

def main():
    # choose to use visualization - don't if collecting large amounts of data
    withVis = True
    explorer = Sim.Sim(withVis)

    # number of simulations to run, use 1 with visualization
    simRuns = 5

    # push direction with respect to the positive y axis
    # used only if random init is off
    theta = 0.0
    withRand = True
    # simulate and plot trajectories
    for simCount in range(simRuns):
        # simulation resets everytime it's called
        explorer.resetSim(withRandom=withRand)
        traj = explorer.simulate(theta)

if __name__ == "__main__":
    main()
