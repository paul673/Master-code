import deepchem as dc
from openpom.feat.graph_featurizer import GraphFeaturizer, GraphConvConstants
from .models import MPNNPOMModel
import pandas as pd
import os

MODULENAME = "pom_models"
base_dir = os.path.dirname(os.path.abspath(__file__))

def smiles_to_dataset(smiles):
    """
    Transforms a smiles string into a graph structure.
    """
    # Featurize the SMILES string
    featurizer = GraphFeaturizer()
    single_molecule = featurizer.featurize([smiles])

    # Create a NumpyDataset for the single molecule
    single_dataset = dc.data.NumpyDataset(single_molecule)
    return single_dataset


def get_train_ratios():
    #df = pd.read_csv(f"{MODULENAME}/train_ratios.csv", index_col=0)
    df = pd.read_csv(os.path.join(base_dir, "train_ratios.csv"), index_col=0)
    train_ratios = list(df["train_ratios"])
    return train_ratios

def get_model(model_dir="trained_model"):
    train_ratios = get_train_ratios()

    model = MPNNPOMModel(n_tasks = 138,
                            batch_size=128,
                            learning_rate=1e-4,
                            class_imbalance_ratio = train_ratios,
                            loss_aggr_type = 'sum',
                            node_out_feats = 100,
                            edge_hidden_feats = 75,
                            edge_out_feats = 100,
                            num_step_message_passing = 5,
                            mpnn_residual = True,
                            message_aggregator_type = 'sum',
                            mode = 'classification',
                            number_atom_features = GraphConvConstants.ATOM_FDIM,
                            number_bond_features = GraphConvConstants.BOND_FDIM,
                            n_classes = 1,
                            readout_type = 'set2set',
                            num_step_set2set = 3,
                            num_layer_set2set = 2,
                            ffn_hidden_list= [392, 392],
                            ffn_embeddings = 256,
                            ffn_activation = 'relu',
                            ffn_dropout_p = 0.12,
                            ffn_dropout_at_input_no_act = False,
                            weight_decay = 1e-5,
                            self_loop = False,
                            optimizer_name = 'adam',
                            log_frequency = 32,
                            model_dir = os.path.join(base_dir, model_dir),
                            device_name='cpu')
    
    # Restore the model from the checkpoint
    model.restore()
    return model

MODEL = get_model()
TRAINING_MODEL = get_model("training_model07")
TEST_MODEL = get_model("test_model03")
def fragance_propabilities_from_smiles(smiles):
    single_dataset = smiles_to_dataset(smiles)
    

    # Predict the probabilities for the single molecule
    predicted_probabilities = MODEL.predict(single_dataset)
    return predicted_probabilities

def fragance_propabilities_from_smiles_(smiles, model_name):
    single_dataset = smiles_to_dataset(smiles)
    

    # Predict the probabilities for the single molecule
    predicted_probabilities = model_name.predict(single_dataset)
    return predicted_probabilities

fragance_propabilities_from_smiles_train = lambda smiles: fragance_propabilities_from_smiles_(smiles, TRAINING_MODEL)
fragance_propabilities_from_smiles_test = lambda smiles: fragance_propabilities_from_smiles_(smiles, TEST_MODEL)