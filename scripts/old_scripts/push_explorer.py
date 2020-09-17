#!/usr/bin/python3

# Sudharshan Suresh (suddhu@cmu.edu), Jan 2020
# Run contour following pybullet simulation for different shape models

import sim
import matplotlib.pyplot as plt
import pdb, sys
import argparse

def main(shape_id, withVis, withVid):
    explorer = sim.Sim(shape_id, withVis, withVid) # init sim
    explorer.resetSim() # reset block and end effector
    explorer.simulate() # simulate contour following and log data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", type=str, default="rect1", help="Shape ID (eg: rect1, ellip2, hex)")
    parser.add_argument("--vis", type=int, default="1", help="Visualize 2D pushing")
    parser.add_argument("--vid", type=int, default="0", help="Record video")
    args = parser.parse_args()

    main(args.shape, bool(args.vis), bool(args.vid))
