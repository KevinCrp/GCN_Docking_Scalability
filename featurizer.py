import scipy.spatial
from openbabel import pybel

from utilities.redirect import stderr_redirected

MAX_BOND_ATOMIC_DISTANCE = 4.0

def open_pdb(filepath: str, hydrogens_removal: bool = True):
    with stderr_redirected():
        pymol = next(pybel.readfile('pdb', filepath))
        if hydrogens_removal:
            pymol.removeh()
        return pymol


def open_mol2(filepath: str, hydrogens_removal: bool = True):
    with stderr_redirected():
        pymol = next(pybel.readfile('mol2', filepath))
        if hydrogens_removal:
            pymol.removeh()
        return pymol


def atom_type_one_hot(atomic_num: int):
    one_hot = 8*[0]
    used_atom_num = [5, 6, 7, 8, 9, 15, 16]  # B, C, N, O, F, P, S, Others
    d_atm_num = {5: 0, 6: 1, 7: 2, 8: 3, 9: 4, 15: 5, 16: 6}
    if atomic_num in used_atom_num:
        one_hot[d_atm_num[atomic_num]] = 1
    return one_hot


# SMARTS definition for other properties
# From TFbio - Kalansanty (https://gitlab.com/cheminfIBB/tfbio/-/blob/master/tfbio/data.py)
smart_property_dict = {'hydrophobic': pybel.Smarts('[#6+0!$(*~[#7,#8,F]),SH0+0v2,s+0,S^3,Cl+0,Br+0,I+0]'),
                       'aromatic': pybel.Smarts('[a]'),
                       'acceptor': pybel.Smarts('[!$([#1,#6,F,Cl,Br,I,o,s,nX3,#7v5,#15v5,#16v4,#16v6,*+1,*+2,*+3])]'),
                       'donor': pybel.Smarts('[!$([#6,H0,-,-2,-3]),$([!H0;#7,#8,#9])]'),
                       'ring': pybel.Smarts('[r]')}


def generate_edges(pos_list):
    dist_mat = scipy.spatial.distance.cdist(pos_list, pos_list)
    adj_matrix = [[], []]
    for i in range(dist_mat.shape[0]):
        for j in range(dist_mat.shape[1]):
            if i != j:
                if dist_mat[i][j] <= MAX_BOND_ATOMIC_DISTANCE:
                    adj_matrix[0] += [i]
                    adj_matrix[1] += [j]
    return adj_matrix


def get_mol_properties(pybel_mol):
    list_hydrophobic = [p[0]
                        for p in smart_property_dict['hydrophobic'].findall(pybel_mol)]
    list_aromatic = [p[0]
                     for p in smart_property_dict['aromatic'].findall(pybel_mol)]
    list_acceptor = [p[0]
                     for p in smart_property_dict['acceptor'].findall(pybel_mol)]
    list_donor = [p[0]
                  for p in smart_property_dict['donor'].findall(pybel_mol)]
    list_ring = [p[0]
                 for p in smart_property_dict['ring'].findall(pybel_mol)]
    atom_feats_list = []
    one_hot_type_list = []
    mol_pos_list = []
    for atom in pybel_mol:
        one_hot_enc = atom_type_one_hot(atom.atomicnum)
        hydrophobic = 1 if atom.idx in list_hydrophobic else 0
        aromatic = 1 if atom.idx in list_aromatic else 0
        acceptor = 1 if atom.idx in list_acceptor else 0
        donor = 1 if atom.idx in list_donor else 0
        ring = 1 if atom.idx in list_ring else 0

        atom_feats_list.append([atom.hyb,
                                atom.heavydegree, atom.heterodegree,
                                atom.partialcharge, hydrophobic,
                                aromatic, acceptor, donor, ring,
                                ])
        one_hot_type_list.append(one_hot_enc)
        mol_pos_list.append(atom.coords)
    return atom_feats_list, one_hot_type_list, mol_pos_list


def featurize(protein_path: str, ligand_path: str):
    # protein_path can be protein or pocket path
    protein = open_pdb(protein_path, hydrogens_removal=True)
    ligand = open_mol2(ligand_path, hydrogens_removal=True)

    protein_atom_feats_list, protein_one_hot_type_list, protein_mol_pos_list = get_mol_properties(
        protein)
    protein_atom_feats_list_atom_protein_or_ligand = len(
        protein_atom_feats_list) * [[1, 0]]

    ligand_atom_feats_list, ligand_one_hot_type_list, ligand_mol_pos_list = get_mol_properties(
        ligand)
    ligand_atom_protein_or_ligand = len(ligand_atom_feats_list) * [[0, 1]]

    atom_feats_list = protein_atom_feats_list + ligand_atom_feats_list
    one_hot_type_list = protein_one_hot_type_list + ligand_one_hot_type_list
    mol_pos_list = protein_mol_pos_list + ligand_mol_pos_list
    protein_or_ligand = protein_atom_feats_list_atom_protein_or_ligand + \
        ligand_atom_protein_or_ligand

    adj_matrix = generate_edges(mol_pos_list)
    # print(len(atom_feats_list))
    # print(len(one_hot_type_list))
    # print(len(protein_or_ligand))
    return atom_feats_list, one_hot_type_list, adj_matrix, mol_pos_list, protein_or_ligand

# featurize("data/raw/1a0q/1a0q_pocket.pdb", "data/raw/1a0q/1a0q_ligand.mol2")