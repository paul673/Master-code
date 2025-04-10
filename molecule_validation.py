from laura import mcf
from rdkit import Chem
import numpy as np
from laura import fingerprint
from sklearn.ensemble import RandomForestClassifier
import pickle

from mcf_parameters import get_tuned_filter_list
FILENAME = "openpomdata.csv"
all_filters = get_tuned_filter_list(FILENAME,quantile=0.01)
# Load model
model_fname = "laura/odorant_classifier.pkl"
with open(model_fname,"rb") as f:
    clf = pickle.load(f)

# Fingerprint gen
mfpgen = fingerprint.make_mfpgen()

def score_molecule(mol):
    result =  np.array([f.apply(mol) for f in all_filters]).astype(int)
    #if result is not None and not np.isnan(result):
    return result
    #print(f"Warning: is_odorant failed on mol: {Chem.MolToSmiles(m)}")
    #return 0    


def validation_func(mol):
    #score = score_molecule(mol).astype(bool)

    # New
    odorant = is_odorant(mol)
    score = np.array([mol.GetNumAtoms() > 1])
    #combined_scores = np.concatenate((score,odorant), axis=0)
    combined_scores = np.concatenate((score,odorant), axis=0)
    
    return all(combined_scores)

def is_odorant(mol):
    smiles = Chem.MolToSmiles(mol)
    mfp = [fingerprint.smiles_to_embed(mfpgen, smiles)]
    return clf.predict(mfp).astype(bool)

if __name__ == "__main__":
    m = "COC1=C(C=CC(=C1)C=O)O"
    print(is_odorant(Chem.MolFromSmiles(m)).astype(int))
    print(score_molecule(Chem.MolFromSmiles(m)))