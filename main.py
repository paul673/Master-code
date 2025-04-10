import socket
from typing import Dict, List, Tuple

import torch
from rdkit import Chem
from rdkit.Chem.rdchem import Mol as RDMol
from torch import Tensor

from gflownet import GFNTask, LogScalar, ObjectProperties
from gflownet.config import Config, init_empty
from gflownet.envs.mol_building_env import MolBuildingEnvContext
from gflownet.online_trainer import StandardOnlineTrainer
import json
import os
import pathlib

from pom_models.functions import fragance_propabilities_from_smiles

class MakeRingsTask(GFNTask):
    """A toy task where the reward is the number of rings in the molecule."""

    def sample_conditional_information(self, n: int, train_it: int) -> Dict[str, Tensor]:
        return {"beta": torch.ones(n), "encoding": torch.ones(n, 1)}

    def cond_info_to_logreward(self, cond_info: Dict[str, Tensor], obj_props: ObjectProperties) -> LogScalar:
        scalar_logreward = torch.as_tensor(obj_props).squeeze().clamp(min=1e-30).log()
        return LogScalar(scalar_logreward.flatten())

    def compute_obj_properties(self, mols: List[RDMol]) -> Tuple[ObjectProperties, Tensor]:
        rs = torch.tensor([m.GetRingInfo().NumRings() for m in mols]).float()
        return ObjectProperties(rs.reshape((-1, 1))), torch.ones(len(mols)).bool()


class MakeRingsTrainer(StandardOnlineTrainer):
    def set_default_hps(self, cfg: Config):
        cfg.hostname = socket.gethostname()
        cfg.num_workers = 8
        cfg.algo.num_from_policy = 64
        cfg.model.num_emb = 128
        cfg.model.num_layers = 4

        cfg.algo.method = "TB"
        cfg.algo.max_nodes = 6
        cfg.algo.sampling_tau = 0.9
        cfg.algo.illegal_action_logreward = -75
        cfg.algo.train_random_action_prob = 0.0
        cfg.algo.valid_random_action_prob = 0.0
        cfg.algo.tb.do_parameterize_p_b = True

        cfg.replay.use = False

    def setup_task(self):
        self.task = MakeRingsTask()

    def setup_env_context(self):
        self.ctx = MolBuildingEnvContext(
            ["C"],
            charges=[0],  # disable charge
            chiral_types=[Chem.rdchem.ChiralType.CHI_UNSPECIFIED],  # disable chirality
            num_rw_feat=0,
            max_nodes=self.cfg.algo.max_nodes,
            num_cond_dim=1,
        )


def save_run(trial):
    state = {
        "models_state_dict": [trial.model.state_dict()],
        "cfg": trial.cfg,
        "env_ctx": trial.model.env_ctx,
        #"model": trial.model,
    }
    if trial.sampling_model is not trial.model:
        state["sampling_model_state_dict"] = [trial.sampling_model.state_dict()]
    fn = pathlib.Path(trial.cfg.log_dir) / "model_final_save.pt"
    with open(fn, "wb") as fd:
        torch.save(
            state,
            fd,
        )

def save_env_ctx(trial):



    env_ctx = {
        "env_ctx": trial.model.env_ctx,
    }

    fn = pathlib.Path(trial.cfg.log_dir) / "env_ctx.pt"
    with open(fn, "wb") as fd:
        torch.save(
            env_ctx,
            fd,
        )
    return


def main():
    """Example of how this model can be run."""
    config = init_empty(Config())
    config.print_every = 1
    config.log_dir = "./logs/debug_run_mr7"

    # For CPU
    config.device = torch.device('cpu')
    config.num_workers = 0

    # For reproduction
    config.seed = 1 

    config.num_training_steps = 10 #10_000
    
    config.algo.tb.do_parameterize_p_b = False # Dont know how to load the model with this parameter = True
    config.num_validation_gen_steps = 1
    


    trial = MakeRingsTrainer(config)
    trial.run()
    save_run(trial)


if __name__ == "__main__":
    main()


