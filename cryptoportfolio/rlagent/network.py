import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
import datetime

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
from typing import Callable, Dict, List, Optional, Tuple, Type, Union


class CustomMLP(nn.Module):
    """
    Custom network for policy and value function.
    It receives as input the features extracted by the feature extractor.

    :param feature_dim: dimension of the features extracted with the features_extractor (e.g. features from a CNN)
    """

    def __init__(
        self,
        feature_dim: int = 13,
        layer_size: int = 100,
        n_layers: int = 0,
    ):
        super(CustomMLP, self).__init__()

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.feature_dim = feature_dim
        self.latent_dim_pi = feature_dim
        self.latent_dim_vf = feature_dim
        self.layer_size = layer_size
        self.n_layers = n_layers
        
        self.policy_net = self.build_network()
        self.value_net = self.build_network()

    def build_network(self):
        # Add the input layer
        layers = [nn.Linear(self.feature_dim, self.layer_size), nn.ReLU()]
        
        # Build the hidden layers given the n_layers
        for _ in range(self.n_layers):
            layers.append(nn.Linear(self.layer_size, self.layer_size))
            layers.append(nn.ReLU())
            
        # Add the output layer with the softmax function
        layers.append(nn.Linear(self.layer_size, self.feature_dim))
        layers.append(nn.ReLU())
        #layers.append(nn.Softmax(dim=0))
        
        """layers.append(nn.Linear(256, 256)); layers.append(nn.ReLU())
        layers.append(nn.Linear(256, 128)); layers.append(nn.ReLU())
        layers.append(nn.Linear(128, self.feature_dim))
        layers.append(nn.ReLU())"""
        
        return nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :return: (torch.Tensor, torch.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        #print("foward:", self.policy_net(features), sum(self.policy_net(features)), self.value_net(features), sum(self.value_net(features)))
        return self.policy_net(features), self.value_net(features)

    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        #print("foward_actor:", self.policy_net(features))
        return self.policy_net(features)

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        #print("foward_critic:", self.value_net(features))
        return self.value_net(features)


class CustomActorCriticPolicy(ActorCriticPolicy):
    """ Contains the CustomMLP """
    
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Callable[[float], float],
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,#nn.Tanh,
        layer_size: int = 64,
        n_layers: int = 0,
        *args,
        **kwargs,
    ):
        self.layer_size = layer_size
        self.n_layers = n_layers

        super(CustomActorCriticPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )
        # Disable orthogonal initialization
        self.ortho_init = False
        
    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = CustomMLP(self.features_dim, self.layer_size, self.n_layers)


class CustomCNN_PPO(BaseFeaturesExtractor):
    """ Reduces the dimensionality of the observation space by extracting features from it. """
    
    def __init__(
        self, 
        observation_space: gym.spaces.Box, 
        features_dim: int = 3,
        agent_env = None
    ):
        
        features_dim = observation_space.shape[1] + 1
        self.agent_env = agent_env
        super(CustomCNN_PPO, self).__init__(observation_space, features_dim)
        
        self.conv1 = nn.Conv2d(observation_space.shape[0], 2, (1,3))
        self.conv2 = nn.Conv2d(2, 20, (1,observation_space.shape[2] - 2))
        self.votes = nn.Conv2d(21, 1, (1,1))
        
        # Cash bias
        b = torch.zeros((1,1))
        #b = torch.ones((1,1))
        self.b = nn.Parameter(b)
        
        self.runtime_CNN = 0

    def forward(self, x):
                
        batch_size = self.agent_env._batch_size
        #print("batch size in CNN:", batch_size)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
                
        # Appending the previous weights to the input during the collection of the rollouts
        if x.shape[0] == 1:
            w = self.agent_env._PVM[-1][1:] # previous weights of all non-cash assets
            w = torch.from_numpy(w).view((1,1,12,1)).float()
        # Collecting the previous weights for the samples in the rollout buffer during training
        elif x.shape[0] == batch_size:
            #print("indices[0]", self.agent_env._indices)
            w = torch.zeros(batch_size,1,12,1)
            for row_num, index in enumerate(self.agent_env._indices[0]):
                o = torch.FloatTensor(self.agent_env._weight_storage[index][1:])
                o = o.view((1,12,1)).float()
                w[row_num] = o
        else: 
            raise ValueError(f"Input shape is not correct. x.shape: {x.shape}. "
                             f"x.shape[0] should be either <<1>> during the collection of the rollouts, "
                             f"or <<{batch_size}>> during the training process where batches are passed through "
                             "the CNN. ")
        
        # Writing the previous weights to the GPU
        w = to_gpu(w)
        x = torch.cat((x, w), dim=1)
        x = self.votes(x)
        x = torch.squeeze(x)  
        
        #if x.shape[0] == batch_size:
        if len(x.shape) > 1:
            # During training the dimension of the output is 2
            # instead of 1 during the collection of rollouts
            cash = self.b.repeat(batch_size, 1)
            x = torch.cat((cash, x), dim=1)
        else:
            # Collection of rollouts
            cash = self.b[0]
            x = torch.cat((cash, x), dim=0)

        #print("weights after CNN:", x)
        # Applying the softmax function to the output
        """
        if x.shape[0] == 13: 
            x = F.softmax(x, dim=0)
        if x.shape[0] == batch_size:
            x = F.softmax(x, dim=1)
        """
        
        return x
    
    
class CustomCNN_DDPG(BaseFeaturesExtractor):
    """ Reduces the dimensionality of the observation space by extracting features from it. """
    
    def __init__(
        self, 
        observation_space: gym.spaces.Box, 
        features_dim: int = 3, 
        agent_env = None
    ):
        
        features_dim = observation_space.shape[1] + 1
        self.agent_env = agent_env
        super(CustomCNN_DDPG, self).__init__(observation_space, features_dim)
        
        self.conv1 = nn.Conv2d(observation_space.shape[0], 2, (1,3))
        self.conv2 = nn.Conv2d(2, 20, (1,observation_space.shape[2] - 2))
        self.votes = nn.Conv2d(21, 1, (1,1))
        
        # Cash bias
        b = torch.zeros((1,1))
        #b = torch.ones((1,1))
        self.b = nn.Parameter(b)

    def forward(self, x):
        batch_size = self.agent_env._batch_size
        #print("batch size in CNN:", batch_size)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
                
        # Appending the previous weights to the input during the collection of the rollouts
        if x.shape[0] == 1:
            w = self.agent_env._PVM[-1][1:] # previous weights of all non-cash assets
            w = torch.from_numpy(w).view((1,1,12,1)).float()
        # Collecting the previous weights for the samples in the rollout buffer during training
        elif x.shape[0] == batch_size:
            w = torch.zeros(batch_size,1,12,1)
            for row_num, index in enumerate(self.agent_env._indices[0]):
                o = torch.FloatTensor(self.agent_env._weight_storage[index][1:])
                o = o.view((1,12,1)).float()
                w[row_num] = o
        else: 
            raise ValueError(f"Input shape is not correct. x.shape: {x.shape}. "
                             f"x.shape[0] should be either <<1>> during the collection of the rollouts, "
                             f"or <<{batch_size}>> during the training process where batches are passed through "
                             "the CNN. ")
        
        # Writing the previous weights to the GPU
        w = to_gpu(w)
        x = torch.cat((x, w), dim=1)
        x = self.votes(x)
        x = torch.squeeze(x)  
        
        #if x.shape[0] == batch_size:
        if len(x.shape) > 1:
            # During training the dimension of the output is 2
            # instead of 1 during the collection of rollouts
            cash = self.b.repeat(batch_size, 1)
            x = torch.cat((cash, x), dim=1)
        else:
            # Collection of rollouts
            cash = self.b[0]
            x = torch.cat((cash, x), dim=0)

        #print("weights after CNN:", x)
        # Applying the softmax function to the output
        """if x.shape[0] == 13: 
            x = F.softmax(x, dim=0)
        if x.shape[0] == batch_size:
            x = F.softmax(x, dim=1)"""

        return x
    
def to_gpu(tensor):
    
    # Moving the tensor to the GPU if available
    if torch.cuda.is_available():
        tensor = torch.FloatTensor(tensor)
        tensor = tensor.to("cuda")
    return tensor