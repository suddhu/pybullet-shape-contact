import Sim
import matplotlib.pyplot as plt
import pdb, sys
import argparse

def main(shape_id, withVid):
    # choose to use visualization - don't if collecting large amounts of data
    withVis = True
    withRand = False
    
    explorer = Sim.Sim(shape_id, withVis, withVid)
    # simulate and plot trajectories
    explorer.resetSim(withRandom=withRand)
    traj = explorer.simulate()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", type=str, default="rect1", help="Shape")
    parser.add_argument("--vid", type=int, default="0", help="Record video")
    args = parser.parse_args()

    main(args.shape, bool(args.vid))
