# torch imports
import torch
from torch.utils.data import Dataset



# Data processing
import numpy as np
import pandas as pd
from numpy.linalg import norm


# Rdkit for Cheminformatics functions and drawing molecules
from rdkit import Chem


import os


# Cosine similarity used for calculating rewards
def cosine_similarity(vec1,vec2):
    return np.dot(vec1,vec2)/(norm(vec1)*norm(vec2))

"""
Section: Offline data

This section processes the OpenPOM dataset to create a Dataset object for offline training.
Two OpenPOMDataset objects are created by the model for test and training data. 
The Dataset object returns an item by indexing it. 
"""

# Go one level up and then into /data/OpenPOM_probs.csv
OPENPOM_FILE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "OpenPOM_probs.csv"))

# OpenPOM vanillin x OpenPOM mol cosine similarity dataframe
def df_pom_pom(openpom_file=OPENPOM_FILE_PATH, vanilla_only=False):
    # Read Offline Molecules
    df_openpom = pd.read_csv(openpom_file)
    vanilla_index = df_openpom[df_openpom["nonStereoSMILES"]=="COc1cc(C=O)ccc1O"].index[0]

    def custom_function(row):
        vanilla_corr = df_openpom.iloc[vanilla_index][1:].to_numpy()  
        scent_values = row[1:].to_numpy()  
        
        # Apply cosine similarity between the rows scent values and the vanilla correlation vector
        return cosine_similarity(scent_values, vanilla_corr)  

    df_openpom['shift_reward'] = df_openpom.apply(custom_function, axis=1)
    return df_openpom

# Store dataframe in variable 
DF_OPENPOM = df_pom_pom()


# Create dataset
class OpenPOMDataset(Dataset):
    def __init__(self, openpom_file="openpomdata.csv", train=True, split_seed=142857, ratio=0.9):
        df=DF_OPENPOM
        df = df.reset_index(drop=True) 

        self.df = df
        rng = np.random.default_rng(split_seed)
        idcs = np.arange(len(self.df))
        rng.shuffle(idcs)
        if train:
            self.idcs = idcs[: int(np.floor(ratio * len(self.df)))]
        else:
            self.idcs = idcs[int(np.floor(ratio * len(self.df))) :]
        self.obj_to_graph = lambda x: x
        self.targets = self.df.keys()[1:]

    def setup(self, task, ctx):
        self.obj_to_graph = ctx.obj_to_graph


    def __len__(self):
        return len(self.idcs)

    def __getitem__(self, idx):
        # Returns the RDkit mol object and corresponding 138 scent labels
        m_smiles = self.df["nonStereoSMILES"][self.idcs[idx]]
        item =  (
            self.obj_to_graph(Chem.MolFromSmiles(m_smiles)),
            torch.tensor([self.df['shift_reward'][self.idcs[idx]]]).float(),
        )
        return item
    