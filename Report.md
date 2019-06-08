[//]: # (Image References)

[image1]: https://github.com/arjunlikesgeometry/DRLND-Project-3/blob/master/P3.png
[image2]: https://github.com/arjunlikesgeometry/DRLND-Project-3/blob/master/MADDPG.png

### Introduction
In this project the MADDPG algorithm was used to solve the environment outlined in the readme. Previously in project 2, DDPG was used to solve the environment, however in this case the multi-agent nature of the task had to be taken into account to solve the task. MADDPG gave each agent it's own actor network so that they could make local observations and actions whilst still making use of the global information provided by the shared critic. 

Overall, this code was modified from my project 2 repo to include individual rather than shared actors in the agent.py file. 

### Algorithm and Network Architecture
![MADDPG][image2]

The algorithm above was taken from this <cite><a href="https://arxiv.org/pdf/1706.02275.pdf"><i>paper</i></a></cite>. As mentioned above, MADDPG uses a shared critic just as in DDPG however the actors are updated independently of eachother. 

Just as in project 2, both the actor and critic neural networks are made up of three linear layers with the relu fuction used in between layers. The tanh function is still used as an output from the actors as this is a continous space where the actions have to be bounded. The details of the structure of each neural net is described in model.py where the Actor() class is used for both actors in agent.py:
```python
def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=400, fc2_units=300):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return F.tanh(self.fc3(x))


class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, seed, fcs1_units=400, fc2_units=300):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fcs1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fcs1 = nn.Linear(state_size, fcs1_units)
        self.fc2 = nn.Linear(fcs1_units+action_size, fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.fcs1.weight.data.uniform_(*hidden_init(self.fcs1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        xs = F.relu(self.fcs1(state))
        x = torch.cat((xs, action), dim=1)
        x = F.relu(self.fc2(x))
return self.fc3(x)
```

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
One of the main challenges of this task was the hyperparameter optomization. Initially the agents kept getting stuck in a loop whereby they would just move simultaneously towards eachother. This kept the score at zero for thousands of episodes and the agent may have never escaped this minimum. Increasing the learning rate of the critic seemed to solve this problem i.e. the agents didn't converge to this kind of behaviour straight away as they did previously. 

### Results
The results show that the environment was solved in 1630 espisodes i.e. this was the point after which the average score was greater than or equal than 0.5 for the next 100 episodes. The weights used to solve the environment have been saved in the checkpoint_actor1.pth checkpoint_actor2.pth and checkpoint_critic.pth files and may be loaded to see the performance of the trained model.

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
Although the agent did train, the environment is clearly very unstable, and over a larger number of episodes the agents could easily converge towards undesirable behaviour again. In the future I could look into stablising the convergence of the algorithm over a larger number of episodes by fine-tuning the exploration and noise parameters. I could also explore multi-agent versions of other deep reinforcemnet learning algorithms using similar mixing strategies, whereby some neural nets are shared and some are not. 


