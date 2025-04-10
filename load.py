import socket
from typing import Dict, List, Tuple
import os

import torch
from rdkit import Chem
from rdkit.Chem.rdchem import Mol as RDMol
from torch import Tensor
from gflownet.algo.trajectory_balance import TrajectoryBalance

from gflownet.envs.graph_building_env import GraphBuildingEnv
from gflownet import GFNTask, LogScalar, ObjectProperties
from gflownet.config import Config, init_empty
from gflownet.envs.mol_building_env import MolBuildingEnvContext
from gflownet.online_trainer import StandardOnlineTrainer

from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt
import seaborn as sns
import gflownet

cfg = init_empty(Config())

cfg.log_dir = "./logs/debug_run_mr6"
"""ea = event_accumulator.EventAccumulator(cfg.log_dir)
ea.Reload()


f, ax = plt.subplots(1, 3, figsize=(4*3,3))
ax[0].plot([i.value for i in ea.Scalars('train_loss')], color=sns.color_palette("Set2")[2])
ax[0].set_yscale('log')
ax[0].set_ylabel('TB loss')
ax[1].plot([i.value for i in ea.Scalars('train_sampled_reward_avg')], color=sns.color_palette("Set2")[2])
ax[1].set_ylabel('Average reward')
ax[1].set_xlabel('Iteration')
ax[2].plot([i.value for i in ea.Scalars('train_logZ')], color=sns.color_palette("Set2")[2])
ax[2].set_ylabel('Predicted logZ')
plt.tight_layout()
plt.show()"""



# Model class must be defined somewhere
state = torch.load(os.path.join(cfg.log_dir, "model_final_save.pt"), weights_only=False)
#model = statemodel.eval()
#state["models_state_dict"]
config = state["cfg"]
env_ctx = state["env_ctx"]
# Model class must be defined somewhere
#state = torch.load(os.path.join(cfg.log_dir, "model_state.pt"), weights_only=False)
#env_ctx = torch.load(os.path.join(cfg.log_dir, "env_ctx.pt"), weights_only=False)
#model = statemodel.eval()
#state["models_state_dict"]
#config = state["cfg"]
#env_ctx = env_ctx["env_ctx"]
#print(state.keys())
#for i, key in enumerate(state["models_state_dict"]):
    #print(f"Model {i}: {key}")  # See available models in checkpoint
algo = TrajectoryBalance(GraphBuildingEnv(),state["env_ctx"],state["cfg"])
model = gflownet.models.graph_transformer.GraphTransformerGFN(env_ctx,config)
print(len(state["models_state_dict"]))
#model.load_state_dict(state["models_state_dict"][0])
#model.eval()

