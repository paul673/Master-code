# Odor recreation with GFlowNET's

## Introduction

A GFlowNET project with the objective to generate molecules with a given odor. the odor is currently restricted to vanillin, but the model can also be trained with different targets. 

Many different approaches for training such as *reward scheduling*, *multible objectives* or *reward masking* have beed evaluated during model development. Most trials are still contained in the `trials` directory. The file structure of the porjects has however changed since many of these trials have been run. Recreating these trials would therefore require the user to run the notebooks models from the project root directory with all the relvant files accessible. This might require some refactoring. Most of these notebooks contains *Dataset*, *Task* and *Trainer* objects (sometimes also *Conditionals*). The notebooks without these classes use the best variant found during this reasearch, which can be imported from `src.model`. 

The `src/submodules/` directory contains submodules used in this work such as the OpenPOM model to predict molecular odors and a RandomForest algorythm for determining if a molecule is odorant. 

## Installation 

This project is based on the [GFlowNET implementation](https://github.com/recursionpharma/gflownet) developed by recursionpharma. The library was modified slightly and is therefore contained in the `src/submodels/gflownet/` directory. The library was installed using pip during earlier stages of develpment, but moved into the project to make modifications to the library simpler.  

Installing the requirements for this project can be difficould since it uses multible libraries with conflicting versions. The commands below were used to set up the project. A detailed `requirements.txt` will be added later. A detailed installation guide has not been created due to time limitations. 

```
pip install --no-deps git+https://github.com/recursionpharma/gflownet.git@f106cde
pip install torch_geometric torch_sparse torch_scatter rdkit gitpython omegaconf wandb --find-links https://data.pyg.org/whl/torch-2.3.0+cpu.html
pip install botorch gpytorch linear_operator jaxtyping pyro-ppl --no-deps
```

