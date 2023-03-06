## 0702

- Test the current simulation time

100 step-1.5~1.7s 

- Test the current objective calculation time

0.005~0.007 for one rollout

- Current issue

Opponent hit the wall but the simulation is not done(Solved)


## 0712

Use the stability as one objective. This can be calculated from the distance between the car and the current nearest point in trajectory.

For example, run pure pursuit in General1 map for 10s, start at index 1, maxL: 1.8, maxP: 0.8 , minL: 0.5 , minP: 0.5

**v_scale 1.0, D 0.0**: 122.60

**v_scale 1.0, D 5.0**: 126.55

**v_scale 1.0, D 20.0**: 113.50, but wiggling

**v_scale 0.7, D 0.0**: 57.15

**v_scale 0.6, D 0.0**: 59.63


## 0713

The L for the pure pursuit might exceed the length of the current trajectory, now fix it. 


TODO:
check the cost function again, make sure there can be a balance
curvature, abs speed, collision with speed

Currently done.

## 0714

speed check

use 1 workers, 2 oppo, 10 scenes to test:

- With 1080 beams
![](2022-07-14_14-08.png)


- With 108 beams
![](2022-07-14_14-13.png)

save 5s, 600 rollouts will be 150s

- Use 15 points for each trajectory instead of 20
![](2022-07-14_14-25.png)

save 3s, 600 rollouts will be 90s

- njit for collision check with opponent
![](2022-07-14_14-49.png)

save 1s, 600 rollouts will be 30s

TODO:
analyze: 
the mapping of final agent (dist, singularity)
the data extract from the rollout 

design more reasonable objective

## 0715
cost weights not normalize

## 0731
current experiments

07: version 2.10


08: version 2.20
fixed opponents. converge after 19200
because we use negative weights, if the weight of abs speed is negative, the car will choose the slowest velocity, affect the objectives. 

09: version 2.30
mix pareto front to the opponents, but not good(because the mixing rate is too large at the beginning). 
Some of the initial opponents have high speed but then is replace by slow agents. 


10: version 2.40
fixed opponents. 

11: version 2.5
with decay

12: version 2.6

13: version 2.7
fixed opponents, change the range of initial velocity

14: v2.8
try to move two weights: length and abs velocity 

15: v3.0
use ittc between opponent and ego car as the safety, also do some filter


16: v3.1


17: v3.2
2 objectives
opp with 0.95 discount

18: v3.3
3 objectives, opp with 0.9 discount


Idea:
use more difficult map, so the online planning will be helpful
because in some particular segment of the track, we should be more safe.


compare difference of different pareto front
same scenario different prototype
ittc
