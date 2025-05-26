import numpy as np
from src.submodels.chemistry_filters import mcf
from rdkit import Chem 
from rdkit.Chem import Crippen, Descriptors, Mol, rdMolDescriptors
import pandas as pd


# OPENPOM data filename
FILENAME = "openpomdata.csv"

# Split filters into different groups 

# Filters based on a single value
base_filters = [
    mcf.LogPFilter(),
    mcf.MolecularWeightFilter(),
    mcf.HBABHBDFilter(),
    mcf.TPSAFilter(),
    mcf.NRBFilter(),
]

# Filters based on boolian values 
toxicity_filters = [
    mcf.ToxicityFilter()
    ]

# Filters based on an array of values
trivial_filters = [
    mcf.TrivialRulesFilter()
]

# All filters
all_filters = base_filters + toxicity_filters + trivial_filters


def property_molecule(mol: Mol) -> np.array:
    """
    Calculates the properties of a molecule based on different filter parameters.
    Args:
        mol (Mol): RDkit molecule

    Returns:
        np.array: Values for parameters checked for in the filters
    """
    base = np.array([f.get(mol) for f in base_filters]).astype(float)
    additional = np.array(trivial_filters[0].get_all(mol)).astype(float)
    return np.concatenate((base, additional), axis=0)


def read_openpom_data(filename:str) -> pd.DataFrame:
    """Creates a dataframe containing filter properties for all openpom molecules.

    Args:
        filename (str): filename to the openpom data csv

    Returns:
        pd.DataFrame: dataframe containing filterdata for openpom mols.
    """
    openpom_smiles = pd.read_csv(filename)["nonStereoSMILES"]
    openpom_smiles_to_mol = {s: Chem.MolFromSmiles(s) for s in openpom_smiles}
    data = {s:property_molecule(mol) for s, mol in openpom_smiles_to_mol.items()}
    df = pd.DataFrame.from_dict(
        data, 
        orient='index', 
        columns=[
            'logP', 
            'Mw', 
            'HBABHBD', 
            'TPSA', 
            'NRB', 
            'NO2', 
            'Heteroatoms',
            'AromaticRings',
            'AtomCount',
            'TripleBonds'
            ])
    return df


def calc_min_max(filename: str, quantile: float = 0.01) -> tuple[pd.core.series.Series,pd.core.series.Series]:
    df = read_openpom_data(filename)
    min_values = df.quantile(quantile)  # percentile (lower bound)
    max_values = df.quantile(1-quantile)  # percentile (upper bound)
    return min_values,max_values

def get_tuned_filter_list(filename: str, quantile: float = 0.01) -> list:
    minv, maxv= calc_min_max(filename, quantile=quantile)
    print("\n" + "#" * 60)
    print("#{:^58}#".format("Initialize MCFs"))
    print("#" * 60 + "\n")

    print("{:<38} {:>10} {:>10}".format("Filter", "Min", "Max"))
    print("-" * 60)

    print("{:<38} {:>10.2f} {:>10.2f}".format("LogP", minv["logP"], maxv["logP"]))
    print("{:<38} {:>10.2f} {:>10.2f}".format("Molecular Weight", minv["Mw"], maxv["Mw"]))
    print("{:<38} {:>10.2f} {:>10.2f}".format("HBAB + HBD", 0, maxv["HBABHBD"]))
    print("{:<38} {:>10.2f} {:>10.2f}".format("TPSA", 0, maxv["TPSA"]))
    print("{:<38} {:>10.2f} {:>10.2f}".format("Rotatable Bonds", 0, maxv["NRB"]))
    print("{:<38} {:>10.2f} {:>10.2f}".format("NO2 Groups", 0, maxv["NO2"]))
    print("{:<38} {:>10.2f} {:>10.2f}".format("Heteroatoms", 0, maxv["Heteroatoms"]))
    print("{:<38} {:>10.2f} {:>10.2f}".format("Aromatic Rings", 0, maxv["AromaticRings"]))
    print("{:<38} {:>10.2f} {:>10.2f}".format("Atom Count", 0, maxv["AtomCount"]))
    print("{:<38} {:>10.2f} {:>10.2f}".format("Triple Bonds", 0, maxv["TripleBonds"]))
    print("{:<38} {:>10.2f} {:>10.2f}".format("Oxygen Chain Length", 0, 1))

    print("\n" + "#" * 60)


    all_filters = [
        mcf.LogPFilter(min_logp=minv["logP"], max_logp=maxv["logP"]),
        mcf.MolecularWeightFilter(min_mw=minv["Mw"],max_mw=maxv["Mw"]),
        mcf.HBABHBDFilter(max_hbab_hbd=maxv["HBABHBD"]),
        mcf.TPSAFilter(max_tpsa=maxv["TPSA"]),
        mcf.NRBFilter(max_nrb=maxv["NRB"]),
        mcf.ToxicityFilter(),
        mcf.TrivialRulesFilter(
            max_no2=maxv["NO2"],
            max_heteroatoms=maxv["Heteroatoms"],
            max_aromatic_rings=maxv["AromaticRings"],
            max_atoms=maxv["AtomCount"],
            max_triple_bonds=maxv["TripleBonds"]
            ),
        mcf.OxygenChainsFilter(max_oxygen_chain_length=1)
    ]

    return all_filters


if __name__ == "__main__":
    #minv, maxv= calc_min_max(FILENAME)
    #print(maxv["Mw"])
    print(get_tuned_filter_list(FILENAME))