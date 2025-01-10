# Lunar_Lander
First atempt at using reinforcement learning with Pytorch and OpenAI's gymnasium.

I used as reference the pytorch tutorial about reinforcement learning and what i learned from the Machine Learning course of Coursera.

https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

https://www.coursera.org/specializations/machine-learning-introduction

And to have an idea of what a README file about the topic look like I used the one of spvino as a reference

https://github.com/svpino/lunar-lander/blob/master/README.md

# Method
I used a Deep Q Network with replay memory

The Q Network is the same as the Pytorch tutorial

```
class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)


    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)
```

We select an action according to an ε-greedy policy.

# Analysis

To find the learning rate α and the discount factor γ I will train the model with different parameters and look at the cumulated reward obtained at the end of each episode

![Hyperparameters search](https://github.com/H0ruki/Lunar_Lander/blob/main/plot_cumulated_reward.png)


The gymnasium documentation tell us that the environement is considered solved if it scored 200 points.

So the two models trained with α=0.99 seems to be working well.

We could also play with the decay rate of ε.









(The code and the README are very messy i will improve it later)

