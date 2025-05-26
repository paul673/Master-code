"""
BASE MODEL

This script contains the base model for this project. 

It can be run for both offline and online training (or both combined). And can be used with Optuna.


Offline training:

Dataset:
- OpenPOM based 
- Excluded:
    - Free radicals
    - More than 1 explicit hydrogen
    - odorless (Check again)
- Contains:
    - SMILES
    - OpenPOM predictions
    - Score (Cosine similarity for vanillin and the molecule based on OpenPOM predictions)
- Training- and testdata split 0.9

Reward: 
- Dataset score

Online training:

Validation function:
- Chemistry filters
    - LogPFilter
    - MolecularWeightFilter
    - HBABHBDFilter
    - TPSAFilter
    - NRBFilter
    - ToxicityFilter
    - TrivialRulesFilter
- Odorant classifier
- More than 2 atoms
- No free radicals

Reward function:
- Cosine similarity vanillin and the molecule based on OpenPOM predictions
    
"""



"""
Import packages
"""


# Main model
from src.model.scent_trainer import ScentTrainer
from src.model.scent_task import ScentTask, fragance_propabilities_from_smiles
from src.model.openpom_dataset import  OpenPOMDataset



# Validation pipeline