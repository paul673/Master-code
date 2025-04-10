from pom_models.functions import fragance_propabilities_from_smiles
import warnings

from rdkit import Chem

warnings.filterwarnings("ignore")

smiles = "COC1=C(C=CC(=C1)C=O)O"
smiles ="C"
smiles =""

m=Chem.MolFromSmiles(smiles)
print(m.GetNumAtoms())

print(fragance_propabilities_from_smiles(smiles))