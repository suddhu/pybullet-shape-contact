# pybullet-shape-contact

Object shape exploration pybullet simulator for shape and pose recovery work. Codebase setup with a little help from my friends: [pybullet](https://github.com/bulletphysics/bullet3/tree/master/examples/pybullet), [public-push-est](https://github.com/mcubelab/push-est-public), and [pybullet-kuka-block-push](https://github.com/Nima-Fazeli/pybullet-kuka-block-push).

## Notes
- Hardcoded some paths, so please change anything that says `/home/suddhu/software/`
- Requires pybullet, argparse, and a few other miscellaneous tools

## Run

```
python3 push_explorer --shape rect1
```
- Shapes: ` rect1, rect2, rect3, hex, ellip1, ellip2, ellip3, tri1, tri2, tri3 `( `butter` is no good due to concave collisions). Models from [More than a Million Ways to Be Pushed: A High-Fidelity Experimental Dataset of Planar Pushing](https://arxiv.org/abs/1604.04038).
- Output saved as a `.json` file with the following information: 
  ```
  ["x of contact position", 
  "y of contact position", 
  "z of contact position", ( = 0)
  "x of contact normal", 
  "y of contact normal", 
  "z of contact normal", ( = 0)
  "x of pusher position", 
  "y of pusher position", 
  "z of pusher position", ( = 0)
  "x of ground truth object pose", 
  "y of ground truth object pose", 
  "z of ground truth object pose", ( = 0)
  "yaw of ground truth object pose"]
  ```

