[//]: # (Image References)

[image1]: https://github.com/arjunlikesgeometry/DRLND-Project-3/blob/master/P3.png
[image2]: https://github.com/arjunlikesgeometry/DRLND-Project-3/blob/master/MADDPG.png

### Introduction
In this project the MADDPG algorithm was used to solve the environment outlined in the readme. Previously in project 2, DDPG was used to solve the environment, however in this case the multi-agent nature of the task had to be taken into account to solve the task. MADDPG gave each agent it's own actor network so that they could make local observations and actions whilst still making use of the global information provided by the shared critic. 

### Algorithm and Network Architecture
![MADDPG][image2]
The algorithm above was taken from this <cite><a href="https://arxiv.org/pdf/1706.02275.pdf"><i>paper</i></a></cite>.

The hyperparameters were as follows:
```python
BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 128  # minibatch size
GAMMA = 0.99  # discount factor
TAU = 1e-3  # for soft update of target parameters
LR_ACTOR = 1e-4  # learning rate of the actor
LR_CRITIC = 1e-3  # learning rate of the critic
WEIGHT_DECAY = 0  # L2 weight decay
```
Changed learning rate of the actor to be smaller than the critic to stop the agent getting stuck. 


### Results
Episode 100	Average Score: 0.01

Episode 200	Average Score: 0.00

Episode 300	Average Score: 0.01

Episode 400	Average Score: 0.00

Episode 500	Average Score: 0.00

Episode 600	Average Score: 0.00

Episode 700	Average Score: 0.01

Episode 800	Average Score: 0.02

Episode 900	Average Score: 0.12

Episode 1000	Average Score: 0.13

Episode 1100	Average Score: 0.11

Episode 1200	Average Score: 0.24

Episode 1300	Average Score: 0.28

Episode 1400	Average Score: 0.13

Episode 1500	Average Score: 0.18

Episode 1600	Average Score: 0.26

Episode 1700	Average Score: 0.40

Episode 1730	Average Score: 0.51

Environment solved in 1630 episodes!	Average Score: 0.51

![Trained Agent][image1]

### Conclusion and Future Work



