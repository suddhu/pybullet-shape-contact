import Sim
import matplotlib.pyplot as plt
import pdb, sys

def main(argv):
    # choose to use visualization - don't if collecting large amounts of data
    withVis = True
    withRand = False

    if (not sys.argv[1]):
        sys.exit("No shape specifed!") 
    else:
        shape_id = str(sys.argv[1])
    
    explorer = Sim.Sim(shape_id, withVis)
    # simulate and plot trajectories
    explorer.resetSim(withRandom=withRand)
    traj = explorer.simulate()

if __name__ == "__main__":
    main(sys.argv)
