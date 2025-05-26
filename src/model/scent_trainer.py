


# GFlowNET imports
from gflownet.config import Config # Config object required for initializing the model
from gflownet.utils.misc import create_logger # For logging
from gflownet.online_trainer import StandardOnlineTrainer # The Trainer class of this model modifies StandardOnlineTrainer
from gflownet.envs.mol_building_env import MolBuildingEnvContext # used to construct molecules from atoms

# torch imports
import torch


# Optuna imports for hyperparameter optimization
import optuna

# Data processing
import numpy as np


# Rdkit for Cheminformatics functions and drawing molecules
from rdkit import Chem


# Other imports
import socket
import gc # For garbage collection
import time

# Dataset
from src.model.openpom_dataset import OpenPOMDataset

# Task 
from src.model.scent_task import ScentTask


class ScentTrainer(StandardOnlineTrainer):

    def __init__(self, config: Config, print_config=True, optuna_trial=None):
        super().__init__(config,print_config)
        if optuna_trial:
            self.optuna_trial = optuna_trial

    def set_default_hps(self, cfg: Config):
        cfg.hostname = socket.gethostname()
        cfg.algo.method = "TB"
        cfg.algo.max_nodes = 20
        cfg.algo.sampling_tau = 0.9
        cfg.algo.valid_random_action_prob = 0.0
        cfg.num_workers = 8
        cfg.num_training_steps = 100000
        cfg.opt.learning_rate = 1e-6 #1e-4
        cfg.opt.weight_decay = 1e-8
        cfg.opt.momentum = 0.9
        cfg.opt.adam_eps = 1e-8
        cfg.opt.lr_decay = 20000
        cfg.opt.clip_grad_type = "total_norm" # Changed from norm
        cfg.opt.clip_grad_param = 10
        cfg.algo.num_from_policy = 32
        cfg.algo.num_from_dataset = 32
        cfg.algo.train_random_action_prob = 0.001
        cfg.algo.illegal_action_logreward = -75
        cfg.model.num_emb = 128
        cfg.model.num_layers = 4




    def setup_env_context(self):
        self.ctx = MolBuildingEnvContext(
            ['Br', 'C', 'Cl', 'F', 'Fe', 'N', 'Na', 'O', 'S'],
            #["C", "N", "F", "O"],
            #charges=[0],  # disable charge
            chiral_types=[Chem.rdchem.ChiralType.CHI_UNSPECIFIED],  # disable chirality
            expl_H_range=[0,1],
            num_cond_dim=self.task.num_cond_dim,
            max_nodes=self.cfg.algo.max_nodes,
        )

    def setup_data(self):
        self.training_data = OpenPOMDataset(train=True)
        self.test_data = OpenPOMDataset(train=False)

    def setup_task(self):
        self.task = ScentTask(dataset=self.training_data,cfg=self.cfg)

    def setup(self):
        super().setup()
        self.training_data.setup(self.task,self.ctx)
        self.test_data.setup(self.task,self.ctx)

    def run(self, logger=None):
        """Trains the GFN for `num_training_steps` minibatches, performing
        validation every `validate_every` minibatches.
        """
        if logger is None:
            logger = create_logger(logfile=self.cfg.log_dir + "/train.log")
        self.model.to(self.device)
        self.sampling_model.to(self.device)
        epoch_length = max(len(self.training_data), 1)
        valid_freq = self.cfg.validate_every
        # If checkpoint_every is not specified, checkpoint at every validation epoch
        ckpt_freq = self.cfg.checkpoint_every if self.cfg.checkpoint_every is not None else valid_freq
        train_dl = self.build_training_data_loader()
        valid_dl = self.build_validation_data_loader()
        if self.cfg.num_final_gen_steps:
            final_dl = self.build_final_data_loader()
        callbacks = self.build_callbacks()
        start = self.cfg.start_at_step + 1
        num_training_steps = self.cfg.num_training_steps
        logger.info("Starting training")
        start_time = time.time()

        self.track_training_loss  = 0
        self.track_optuna_losses = [] # Mean loss over the last 5 iterations

        self.num_it_no_progress = 0 
        self.min_loss = float('inf')
        self.patience = 15
        self.min_delta = 1e-6
        self.stop_training = False

        for it, batch in zip(range(start, 1 + num_training_steps), cycle(train_dl)):
            # the memory fragmentation or allocation keeps growing, how often should we clean up?
            # is changing the allocation strategy helpful?

            #if it % 1024 == 0:
            if it % 128 == 0:
                gc.collect()
                torch.cuda.empty_cache()
            epoch_idx = it // epoch_length
            batch_idx = it % epoch_length
            if self.replay_buffer is not None and len(self.replay_buffer) < self.replay_buffer.warmup:
                logger.info(
                    f"iteration {it} : warming up replay buffer {len(self.replay_buffer)}/{self.replay_buffer.warmup}"
                )
                continue
            info = self.train_batch(batch.to(self.device), epoch_idx, batch_idx, it)
            info["time_spent"] = time.time() - start_time
            start_time = time.time()
            self.log(info, it, "train")
            if it % self.print_every == 0:
                logger.info(f"iteration {it} : " + " ".join(f"{k}:{v:.2f}" for k, v in info.items()))


            #OPTUNA LOGG
            
            if hasattr(self, 'optuna_trial'):
                self.optuna_trial.set_user_attr(it, info)

            # OPTUNA 2
            self.track_optuna_losses.append(info["loss"])
            if it % 5 == 0:
                # OPTUNA HYPERPARAMETER OPTIMIZATION
                if hasattr(self, 'optuna_trial'):
                    mean_val_loss = np.mean(self.track_optuna_losses)
                    self.optuna_trial.report(mean_val_loss, it)
                    self.track_training_loss  = mean_val_loss
                    if self.optuna_trial.should_prune():
                        raise optuna.TrialPruned()

                self.track_optuna_losses = []

            #Patiecne Stopping
            #if info["loss"] < self.min_loss - self.min_delta:
            #    self.min_loss = info["loss"]
            #    self.num_it_no_progress = 0
           # else:
            #    self.num_it_no_progress += 1
            #    if self.num_it_no_progress >= self.patience:
            #        self.stop_training = True


            if valid_freq > 0 and it % valid_freq == 0:
                for batch in valid_dl:
                    info = self.evaluate_batch(batch.to(self.device), epoch_idx, batch_idx)
                    self.log(info, it, "valid")
                    logger.info(f"validation - iteration {it} : " + " ".join(f"{k}:{v:.2f}" for k, v in info.items()))

                
                end_metrics = {}
                for c in callbacks.values():
                    if hasattr(c, "on_validation_end"):
                        c.on_validation_end(end_metrics)
                self.log(end_metrics, it, "valid_end")
            if ckpt_freq > 0 and it % ckpt_freq == 0:
                self._save_state(it)

            if self.stop_training:
                logger.info(f"Stop training at iteration {it}")
                break

        self._save_state(num_training_steps)

        num_final_gen_steps = self.cfg.num_final_gen_steps
        final_info = {}
        if num_final_gen_steps:
            logger.info(f"Generating final {num_final_gen_steps} batches ...")
            for it, batch in zip(
                range(num_training_steps + 1, num_training_steps + num_final_gen_steps + 1),
                cycle(final_dl),
            ):
                if hasattr(batch, "extra_info"):
                    for k, v in batch.extra_info.items():
                        if k not in final_info:
                            final_info[k] = []
                        if hasattr(v, "item"):
                            v = v.item()
                        final_info[k].append(v)
                if it % self.print_every == 0:
                    logger.info(f"Generating objs {it - num_training_steps}/{num_final_gen_steps}")
            final_info = {k: np.mean(v) for k, v in final_info.items()}

            logger.info("Final generation steps completed - " + " ".join(f"{k}:{v:.2f}" for k, v in final_info.items()))
            self.log(final_info, num_training_steps, "final")

        # for pypy and other GC having implementations, we need to manually clean up
        del train_dl
        del valid_dl
        if self.cfg.num_final_gen_steps:
            del final_dl

# Required for iterating through batch
def cycle(it):
    while True:
        for i in it:
            yield i