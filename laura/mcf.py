from abc import ABC, abstractmethod

from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Mol, rdMolDescriptors


class MedicinalChemFilter(ABC):
    @abstractmethod
    def apply(self, mol: Mol) -> bool:
        """Return True if the molecule passes the filter."""

    @classmethod
    @abstractmethod
    def get(self, mol: Mol) -> float:
        """Returns value of molecule"""


# LogP Filter
class LogPFilter(MedicinalChemFilter):
    def __init__(self, min_logp: float = -2, max_logp: float = 7):
        self.min_logp = min_logp
        self.max_logp = max_logp

    def apply(self, mol: Mol) -> bool:
        logp = Crippen.MolLogP(mol)
        return self.min_logp < logp < self.max_logp
    
    @classmethod
    def get(cls, mol: Mol) -> float:
        logp = Crippen.MolLogP(mol)
        return logp


# Molecular Weight (MW) Filter
class MolecularWeightFilter(MedicinalChemFilter):
    def __init__(self, min_mw: float = 250, max_mw: float = 750):
        self.min_mw = min_mw
        self.max_mw = max_mw

    def apply(self, mol: Mol) -> bool:
        mw = Descriptors.MolWt(mol)
        return self.min_mw < mw < self.max_mw
    
    @classmethod
    def get(cls, mol: Mol) -> float:
        mw = Descriptors.MolWt(mol)
        return mw


# HBA + HBD Filter
class HBABHBDFilter(MedicinalChemFilter):
    def __init__(self, max_hbab_hbd: int = 10):
        self.max_hbab_hbd = max_hbab_hbd

    def apply(self, mol: Mol) -> bool:
        hba = rdMolDescriptors.CalcNumHBA(mol)
        hbd = rdMolDescriptors.CalcNumHBD(mol)
        return (hba + hbd) < self.max_hbab_hbd
    
    @classmethod
    def get(cls, mol: Mol) -> float:
        hba = rdMolDescriptors.CalcNumHBA(mol)
        hbd = rdMolDescriptors.CalcNumHBD(mol)
        return hba + hbd


# TPSA Filter
class TPSAFilter(MedicinalChemFilter):
    def __init__(self, max_tpsa: float = 150):
        self.max_tpsa = max_tpsa

    def apply(self, mol: Mol) -> bool:
        tpsa = rdMolDescriptors.CalcTPSA(mol)
        return tpsa < self.max_tpsa
    
    @classmethod
    def get(cls, mol: Mol) -> float:
        tpsa = rdMolDescriptors.CalcTPSA(mol)
        return tpsa



# Number of Rotatable Bonds (NRB) Filter
class NRBFilter(MedicinalChemFilter):
    def __init__(self, max_nrb: int = 10):
        self.max_nrb = max_nrb

    def apply(self, mol: Mol) -> bool:
        nrb = rdMolDescriptors.CalcNumRotatableBonds(mol)
        return nrb < self.max_nrb
    
    @classmethod
    def get(cls, mol: Mol) -> float:
        nrb = rdMolDescriptors.CalcNumRotatableBonds(mol)
        return nrb


# In-house Toxicity and Reactive Group Filters
class ToxicityFilter(MedicinalChemFilter):
    def __init__(self):
        self.toxic_smarts = [
            "[N+](=O)[O-]",  # Nitro groups
            "[#6](=O)[#8]",  # Carboxylic acid
            "[F,Cl,Br,I][C,c]=[C,c]",  # Halogen-substituted electrophiles
            "[S,s]~[S,s]",  # Disulfides
            "[C;R][C;R]",  # Strained rings
            "[N,n][O,o]",  # Hydroxamic acids
            "[N,n]~[N,n]",  # Diazo compounds
            "[C,c](=[O,o])[C,c](=[O,o])",  # Anhydrides
        ]
        self.toxic_patterns = [Chem.MolFromSmarts(smarts) for smarts in self.toxic_smarts]

    def apply(self, mol: Mol) -> bool:
        return all(not mol.HasSubstructMatch(pat) for pat in self.toxic_patterns)
    
    @classmethod
    def get(cls, mol: Mol) -> float:
        print("Not implementet")
        return 0.0



# Trivial Filtering Rules
class TrivialRulesFilter(MedicinalChemFilter):
    def __init__(
        self,
        max_no2: int = 2,
        #max_cl: int = 3,
        #max_br: int = 2,
        #max_f: int = 6,
        max_heteroatoms: int = 3,
        max_aromatic_rings: int = 5,
        max_atoms: int = 28,
        #max_oxygen: int =4,
        max_triple_bonds: int =2,
    ):
        self.max_no2 = max_no2
        #self.max_cl = max_cl
        #self.max_br = max_br
        #self.max_f = max_f
        self.max_aromatic_rings = max_aromatic_rings
        self.max_heteroatoms = max_heteroatoms
        self.max_atoms = max_atoms
        #self.max_oxygen = max_oxygen
        self.max_triple_bonds = max_triple_bonds
        self.num_radical_electrons = 0

    def apply(self, mol: Mol) -> bool:
        return (
            len(mol.GetSubstructMatches(Chem.MolFromSmarts("[N+](=O)[O-]"))) <= self.max_no2
            and Chem.Lipinski.NumHeteroatoms(mol) <=  self.max_heteroatoms #self.max_cl
            and rdMolDescriptors.CalcNumAromaticRings(mol) <= self.max_aromatic_rings
            and mol.GetNumAtoms() <= self.max_atoms
            and sum(1 for bond in mol.GetBonds() if bond.GetBondType() == Chem.rdchem.BondType.TRIPLE) <= self.max_triple_bonds
            and Chem.Descriptors.NumRadicalElectrons(mol) == self.num_radical_electrons
        )

    @classmethod
    def get(cls, mol: Mol) -> float:
        print("Not implementet")
        return 0.0
    
    def get_all(self, mol):
        no2 = len(mol.GetSubstructMatches(Chem.MolFromSmarts("[N+](=O)[O-]")))
        heteroatoms = Chem.Lipinski.NumHeteroatoms(mol)
        aromatic_rings = rdMolDescriptors.CalcNumAromaticRings(mol)
        atoms = mol.GetNumAtoms()
        triple = sum(1 for bond in mol.GetBonds() if bond.GetBondType() == Chem.rdchem.BondType.TRIPLE)
        return [no2, heteroatoms, aromatic_rings, atoms, triple]
    

class OxygenChainsFilter(MedicinalChemFilter):
    def __init__(
            self,
            max_oxygen_chain_length=1 
            ):
        self.max_oxygen_chain_length = max_oxygen_chain_length
        
    def iterate_molecule(self, neighbors, visited, traj, length):
        """Recursive method to iterate neighbors of a currently visited oxygen atom. Finds oxygen neighbors and attaches thoses to a trajectory.

        Args:
            neighbors (list): Neighbors of the currently visited oxygen atom.
            visited (set): Previously visited oxygen atoms.
            traj (set): Trajectory containing atom indexes of the currently discovered oxygen chain.
            length (int): Current trajectory length

        Returns:
            traj (set): Trajectory containing atom indexes of the currently discovered oxygen chain.
            visited (set): Previously visited oxygen atoms
            length (int): Current trajectory length
        """
        for neighbor in neighbors:
            if neighbor.GetAtomicNum() != 8 or neighbor.GetIdx() in visited:
                continue
            visited.add(neighbor.GetIdx())
            traj.add(neighbor.GetIdx())
            length += 1
            traj, visited, length = self.iterate_molecule(neighbor.GetNeighbors(), visited,traj,length)
        return traj, visited, length
    
    def okygen_chains(self, mol):
        visited = set()
        trajs = []
        lengths = {}
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() != 8 or atom.GetIdx() in visited:
                continue
            visited.add(atom.GetIdx())
            neighbors = atom.GetNeighbors()
            traj = {atom.GetIdx()}
            length = 1
            traj, visited, length  = self.iterate_molecule(neighbors,visited, traj, length)

            trajs.append(traj)
            if length in lengths.keys():
                lengths[length] += 1
            else:
                lengths[length] = 1
        return trajs,lengths
    
    def apply(self, mol: Mol) -> bool:
        trajs,lengths = self.okygen_chains(mol)
        return max(lengths.keys()) <= self.max_oxygen_chain_length
    
    @classmethod
    def get(cls, mol: Mol) -> float:
        print("Not implementet")
        return 0.0