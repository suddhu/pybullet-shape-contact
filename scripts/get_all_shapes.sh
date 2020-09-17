#!/bin/bash
for i in {1..50}
do  
    echo "Collect rect1, hex, ellip2 for dataset# ${i}"
    if [ $i == 1 ]; then
    python3 push_explorer.py --shape rect1 --peturb 0
    python3 push_explorer.py --shape hex --peturb 0
    python3 push_explorer.py --shape ellip2 --peturb 0
    else
    python3 push_explorer.py --shape rect1 
    python3 push_explorer.py --shape hex
    python3 push_explorer.py --shape ellip2
    fi
done

# all shapes
# python3 simple_sim.py --shape rect1 
# python3 simple_sim.py --shape rect2
# python3 simple_sim.py --shape rect3
# python3 simple_sim.py --shape ellip1
# python3 simple_sim.py --shape ellip2 
# python3 simple_sim.py --shape ellip3
# python3 simple_sim.py --shape hex
# python3 simple_sim.py --shape tri1
# python3 simple_sim.py --shape tri2
# python3 simple_sim.py --shape tri3
