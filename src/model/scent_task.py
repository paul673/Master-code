"""
Import packages
"""

# GFlowNET imports
from src.submodels.gflownet.config import Config # Config object required for initializing the model
from src.submodels.gflownet import GFNTask,LogScalar,ObjectProperties # Task is used for specifiyng the models objective (Reward)


# torch imports
import torch
from torch import Tensor
from torch.utils.data import Dataset

# Optuna imports for hyperparameter optimization
import optuna

# Data processing
import numpy as np
from numpy.linalg import norm


# Rdkit for Cheminformatics functions and drawing molecules
from rdkit import Chem

# Typing (Original library uses typing. This projects does not use typing consitently)
from typing import Dict


# OpenPOM imports
from src.submodels.openpom.functions import fragance_propabilities_from_smiles # Calculates odor probabilities for a given molecule
TARGET_SMILES = "COC1=C(C=CC(=C1)C=O)O" # Vanillin
TARGET_VEC = fragance_propabilities_from_smiles(TARGET_SMILES)[0] # Calculate the target vector based on Vanillin


# MCF and is_odorant imports
from src.filters.molecule_validation import is_odorant, score_molecule
# Function returning 1 if all filters pass
def hard_filter(mol):
    return int(int(is_odorant(mol)[0]) * score_molecule(mol).all())


"""
Task: Define model objective

The ScentTask is used to calculate rewards during online training and transforms both offline and online 
rewards based on different parameters. This model contains a simple Task object calculating rewards based
on cosine similarity to vanillin and by using the hard_filterfunction along side other filters to invalidate 
some online molecules. Other more complex implementations with multible objectives and reward schedluing have
been tried out previously, but they did not improve the model significantly.  
"""

class ScentTask(GFNTask):
    def __init__(self, dataset: Dataset, cfg: Config):
        self.dataset = dataset
        self.num_cond_dim = 1
        self.num_objectives = 138

    def sample_conditional_information(self, n: int, train_it: int) -> Dict[str, Tensor]:
        return {"encoding": torch.ones(n, 1)}

    def cond_info_to_logreward(self, cond_info: Dict[str, Tensor], obj_props: ObjectProperties) -> LogScalar:
        scalar_logreward = torch.as_tensor(obj_props).squeeze().clamp(min=1e-30).log()
        return LogScalar(scalar_logreward.flatten())
    
    def compute_obj_properties(self, mols):
        is_valid = torch.tensor([m.GetNumAtoms() > 1 and Chem.Descriptors.NumRadicalElectrons(m) == 0 and hard_filter(m) == 1 for m in mols ]).to(torch.bool)#.bool()
        if not is_valid.any():
            return ObjectProperties(torch.zeros((0,1))), is_valid
        
        valid_mols = [mol for mol, valid in zip(mols, is_valid) if valid]
        rs = torch.tensor(self.compute_reward_from_mols(valid_mols))
        return ObjectProperties(rs.reshape((-1, 1))), is_valid 

    
    def compute_reward_from_mols(self, mols):
        rewards = []
        for m in mols:
            m_smiles = Chem.MolToSmiles(m)
            m_vec = fragance_propabilities_from_smiles(m_smiles)[0]
            rewards.append(self.cosine_similarity(TARGET_VEC,m_vec))
        return rewards
    
    def cosine_similarity(self,vec1,vec2):
        return np.dot(vec1,vec2)/(norm(vec1)*norm(vec2))
    