from rdkit import Chem
from rdkit.Chem import Draw #MolsToGridImage,MolToImage

from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D
import cairosvg
import io
import sys
import json

def gen_mol_grid_image(savename, smiles_list, rewards, molsPerRow=2, subImgSize=(200*3, 120*3)):
    legends=[f'R(x) = {r:.2f}' for r in rewards]
    useSVG=False
    mols = [Chem.MolFromSmiles(s) for s in smiles_list]
    img = Draw.MolsToGridImage(mols, molsPerRow=2, subImgSize=subImgSize, legends=legends,useSVG=True)
    print(type(img))
    img.save(savename)
    return 1



def molecule_to_pdf(mol, full_path,bond_width, width=300, height=300):
    """Save substance structure as PDF"""

    # Define full path name
    #full_path = f"./figs/2Dstruct/{file_name}.pdf"

    # Render high resolution molecule
    drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
    drawer.drawOptions().bondLineWidth = bond_width 
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()

    # Export to pdf
    cairosvg.svg2pdf(bytestring=drawer.GetDrawingText().encode(), write_to=full_path)

# Example
#m = Chem.MolFromSmiles('Cn1cnc2n(C)c(=O)n(C)c(=O)c12')
#molecule_to_pdf(m, "myfav")

def gen_mol_images(savedir, smiles_list, rewards,vanilla_prob,db_match, size=(200*3, 120*3), bond_width=3):
    for i, s in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(s)
        # Customize drawing options
        #options = Draw.DrawingOptions()
        #options.bondLineWidth = bond_width  # Thicker bonds
        # Save the molecule to a PNG file
        #image = Draw.MolToImage(mol,size=size,useSVG=True,bondLineWidth= bond_width)
        #print(round(rewards[i],2))
        #image.save(f"images/{savedir}/mol{i}r{round(rewards[i],2)}.pdf")
        molecule_to_pdf(mol, f"C:\\Users\\paulj\Desktop\\Master\\images/{savedir}/mols/mol{i}r{round(rewards[i],2)}vp{round(vanilla_prob[i],2)}db{'y' if db_match[i] else 'n'}.pdf", bond_width, width=size[0], height=size[1])

    return 1

# from scent_gfn.image_gen import gen_mol_grid_image
# smiles_list = ['CC(=O)C1C(C2CCC(C)O2)OCC(c2cc[nH]c2)C1C1CCCCO1', 'COCOCCC1OC(CO)CC1c1cccs1', 'c1ccc(C2CCOCC2c2ccc(C3COCC(C4CCCC(C5CCCO5)C4)C3)s2)cc1', 'C1=C(C2CCCC(C3CCOCC3c3ccccc3-c3ccc(C4CCCO4)s3)C2)CCCC1']
# rewards = [0.8318676352500916,0.8210368156433105,0.7954882979393005,0.7778464555740356]
# gen_mol_grid_image("images/vanilla_1_mols.pdf", smiles_list, rewards, molsPerRow=2, subImgSize=(200*3, 120*3))  

if __name__ == "__main__":
    input_file = sys.argv[1]
    with open(input_file, 'r') as f:
        inputs = json.load(f)
    print(inputs)
    gen_mol_images(
        savedir=inputs['savedir'],
        smiles_list=inputs['smiles_list'],
        rewards=inputs['rewards'],
        vanilla_prob=inputs['vanilla_prob'],
        db_match=inputs['db_match'],
        size=tuple(inputs.get('size', (600, 360))),
        bond_width=inputs.get('bond_width', 3)
    )