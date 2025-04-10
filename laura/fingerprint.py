import numpy as np
import rdkit.Chem as Chem
import rdkit.Chem.rdFingerprintGenerator as rdFPGen
from rdkit import RDLogger


def make_mfpgen(radius=4, fpSize=2048):
    return rdFPGen.GetMorganGenerator(radius=radius, fpSize=fpSize)


def smiles_to_embed(mfpgen, smiles, dtype=np.float64, filter_warnings=True):
    try:
        if filter_warnings:
            RDLogger.DisableLog("rdApp.*")  # Suppresses RDKit warnings

        mol = Chem.MolFromSmiles(smiles)

        if filter_warnings:
            RDLogger.EnableLog("rdApp.*")  # Re-enables RDKit warnings

        if mol is None:
            return np.nan

        fp = mfpgen.GetFingerprint(mol)

        # https://github.com/rdkit/rdkit/discussions/3863
        fp = np.frombuffer(bytes(fp.ToBitString(), "utf-8"), "u1") - ord("0")
        return fp.astype(dtype)

    except TypeError:
        return np.nan
