from gflownet.utils.conditioning import Conditional
from gflownet import  LogScalar,ObjectProperties,LinScalar
from gflownet.config import Config
import numpy as np
import torch

import abc
from typing import Dict, Generic, Optional, TypeVar
from torch import Tensor


from base_model.filters.molecule_validation import is_odorant, score_molecule
from gflownet.utils.transforms import thermometer


Tin = TypeVar("Tin")
Tout = TypeVar("Tout")


class Conditional(abc.ABC, Generic[Tin, Tout]):
    def sample(self, n, train_it: Optional[int] = None):
        raise NotImplementedError()

    @abc.abstractmethod
    def transform(self, cond_info: Dict[str, Tensor], data: Tin) -> Tout:
        raise NotImplementedError()

    def encoding_size(self):
        raise NotImplementedError()

    def encode(self, conditional: Tensor) -> Tensor:
        raise NotImplementedError()

class SchedulingConditional(Conditional[LogScalar, LogScalar]):
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.num_training_steps = cfg.num_training_steps
        self.scheduling_fractions = [0.25,0.25,0.5] # Fractions used for a certain training regime
        self.cumulative_fractions = np.cumsum([0]+self.scheduling_fractions)[:-1] # Calculates the cumsum and removes the last value (1) and adds 0 to the start
        self.thresholds = torch.tensor(self.cumulative_fractions * self.num_training_steps) # Calculates thresholds (Training iterations where a regime should start)
        



    def encoding_size(self):
        return 1

    def sample(self, n, train_it):
        #torch.tensor([train_it >= t for t in self.thresholds]).int()

        
        return {"reward_scheduling": (train_it >= self.thresholds).int().unsqueeze(0).expand(n, -1), "encoding": torch.ones((n,1))}

    def transform(self, cond_info: Dict[str, Tensor], obj_properties) -> LogScalar:
        regime_encoding = cond_info["reward_scheduling"]
        reward = (obj_properties * regime_encoding).sum(dim=1, keepdim=True)  
        log_reward = reward.clamp(min=1e-30).log()  
        return log_reward

    def encode(self, conditional: Tensor) -> Tensor:
        print("CONDITIONAL")
        return torch.ones((conditional.shape[0], 1))
    

class SchedulingConditional_(Conditional[LogScalar, LogScalar]):
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.num_training_steps = cfg.num_training_steps
        self.scheduling_fractions = [0.25,0.25,0.5] # Fractions used for a certain training regime
        self.cumulative_fractions = np.cumsum([0]+self.scheduling_fractions)[:-1] # Calculates the cumsum and removes the last value (1) and adds 0 to the start
        self.thresholds = torch.tensor(self.cumulative_fractions * self.num_training_steps) # Calculates thresholds (Training iterations where a regime should start)
        

    def encoding_size(self):
        return 1

    def sample(self, n, train_it):
        #torch.tensor([train_it >= t for t in self.thresholds]).int()

        return {"reward_scheduling": (train_it >= self.thresholds).int().unsqueeze(0).expand(n, -1), "encoding": torch.ones((n,1))}

    def transform(self, cond_info: Dict[str, Tensor], obj_properties) -> LogScalar:
        regime_encoding = cond_info["reward_scheduling"]
        reward = (obj_properties * regime_encoding).sum(dim=1, keepdim=True)  
        log_reward = reward.clamp(min=1e-30).log()  
        return log_reward

    def encode(self, conditional: Tensor) -> Tensor:
        print("CONDITIONAL")
        return torch.ones((conditional.shape[0], 1))
    

    #Helper methods
    def has_odor(self,mol):
        return int(all(is_odorant(mol).astype(int)))
    
    def passes_all_mcf(self,mol):
        return int(all(score_molecule(mol)))
    
    def mean_mcf(self,mol):
        return score_molecule(mol).mean()
    


class RewardSchedulingConditional(Conditional[ObjectProperties, LinScalar]):
    def __init__(self, cfg: Config):
        
        
        self.num_thermometer_dim = 5

        self.num_training_steps = cfg.num_training_steps
        self.scheduling_fractions = [0.2,0.2,0.2,0.4] # Fractions used for a certain training regime
        self.num_objectives = len(self.scheduling_fractions)
        self.cumulative_fractions = np.cumsum([0]+self.scheduling_fractions)[:-1] # Calculates the cumsum and removes the last value (1) and adds 0 to the start
        self.thresholds = torch.tensor(self.cumulative_fractions * self.num_training_steps) # Calculates thresholds (Training iterations where a regime should start)

    def sample(self, n, train_it):

        #(train_it >= self.thresholds).int().unsqueeze(0).expand(n, -1)
        #preferences = (train_it >= self.thresholds).int().unsqueeze(0).expand(n, -1)
        #preferences = torch.as_tensor(preferences).float()
        #return {"preferences": preferences, "encoding": self.encode(preferences)}


        #preferences = (train_it >= self.thresholds).int()

        
        cum_preferences = (train_it >= self.thresholds).int()

        shift = torch.cat([cum_preferences[1:], torch.tensor([0], dtype=cum_preferences.dtype)])

        preferences = cum_preferences-shift
        preferences = preferences.unsqueeze(0).expand(n, -1)
        preferences = torch.as_tensor(preferences).float()

    


        return {"preferences": preferences, "encoding": torch.ones((n,1))}

    def transform(self, cond_info: Dict[str, Tensor], flat_reward: ObjectProperties) -> LinScalar:
        
        scalar_reward = (flat_reward * cond_info["preferences"]).sum(1)

        # Normalize by sum of preferences per row 
        norm_factor = cond_info["preferences"].sum(1, keepdim=True)  # Shape: (batch_size, 1)
        norm_factor = norm_factor.clamp(min=1.0)  # Avoid division by zero
        #zprint(f"Norm factor: {norm_factor.squeeze(1)}")
        scalar_reward = scalar_reward / norm_factor.squeeze(1)  # Normalize

        assert len(scalar_reward.shape) == 1, f"scalar_reward should be a 1D array, got {scalar_reward.shape}"
        return LinScalar(scalar_reward)

    def encoding_size(self):
        return max(1, self.num_thermometer_dim * self.num_objectives)

    def encode(self, conditional: Tensor) -> Tensor:
        if self.num_thermometer_dim > 0:
            return thermometer(conditional, self.num_thermometer_dim, 0, 1).reshape(conditional.shape[0], -1)
        else:
            return conditional.unsqueeze(1)
        

