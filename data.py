import multiprocessing as mp
import os
import os.path as osp

import pandas as pd
import pytorch_lightning as pl
import torch
import torch_geometric as pyg
from biopandas.pdb import PandasPdb

import featurizer


def clean_pdb(pdb_path, out_filename):
    # Remove HETATM
    ppdb = PandasPdb()
    ppdb.read_pdb(pdb_path)
    ppdb.to_pdb(path=out_filename,
                records=['ATOM'],
                gz=False,
                append_newline=True)


def create_pyg_graph(protein_path: str,
                     ligand_path: str,
                     target: float = None):

    y = torch.tensor(target).float() if target is not None else None

    atom_feats_list, one_hot_type_list, adj_matrix, mol_pos_list, pocket_or_ligand = featurizer.featurize(
        protein_path, ligand_path)
    node_feats = torch.cat((torch.tensor(atom_feats_list), torch.tensor(
        one_hot_type_list), torch.tensor(pocket_or_ligand)), axis=1)

    edge_index = torch.tensor(adj_matrix)
    graph = pyg.data.Data(x=node_feats,
                          edge_index=edge_index,
                          y=y)

    return graph


def save_graph(raw_path: str, processed_path: str,
               target: float = None, only_pocket: bool = False):
    pdb_id = raw_path.split('/')[-1]
    protein_path = osp.join(
        raw_path, pdb_id+'_pocket.pdb') if only_pocket else osp.join(raw_path, pdb_id+'_protein.pdb')
    protein_path_clean = osp.join(
        raw_path, pdb_id+'_pocket_clean.pdb') if only_pocket else osp.join(raw_path, pdb_id+'_protein_clean.pdb')
    ligand_path = osp.join(raw_path, pdb_id+'_ligand.mol2')
    if not osp.isfile(protein_path_clean):
        clean_pdb(protein_path, protein_path_clean)
    g = create_pyg_graph(protein_path_clean, ligand_path, target)
    torch.save(g, processed_path)
    return processed_path


def combine_dicts(dico_1, dico_2):
    for pdb_id in dico_2.keys():
        if pdb_id not in dico_1.keys():
            dico_1[pdb_id] = dico_2[pdb_id]
    return dico_1


class PDBBindDataset(pyg.data.InMemoryDataset):
    def __init__(self, root: str, stage: str, only_pocket: bool = False,
                 transform=None,
                 pre_transform=None):
        self.stage = stage
        self.only_pocket = only_pocket
        self.prefix = 'pocket_' if only_pocket else 'protein_'
        self.df = pd.read_csv(
            osp.join(root, '{}.csv'.format(stage))).set_index('pdb_id')
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        filename_list = []
        for pdb_id in self.df.index:
            filename_list.append(pdb_id)
        return filename_list

    @property
    def processed_file_names(self):
        return [osp.join('{}_{}.pt'.format(self.prefix, self.stage))]

    def process(self):
        i = 0
        print('\t', self.stage)
        pool_args = []
        for raw_path in self.raw_paths:
            filename = osp.join(self.processed_dir,
                                'TMP_{}{}_data_{}.pt'.format(self.prefix,
                                                             self.stage, i))
            pdb_id = raw_path.split('/')[-1]
            pool_args.append((raw_path, filename,
                             self.df.loc[pdb_id]['target'], self.only_pocket))
            i += 1
        pool = mp.Pool(mp.cpu_count())
        data_path_list = list(pool.starmap(save_graph, pool_args))
        data_list = []
        for p in data_path_list:
            data_list.append(torch.load(p))
            os.remove(p)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class PDBBindDataModule(pl.LightningDataModule):

    def __init__(self, root, batch_size: int = 1, num_workers: int = 1,
                 only_pocket: bool = False):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.root = root
        self.only_pocket = only_pocket
        super().__init__()

    def prepare_data(self):
        pass

    def setup(self, stage):
        self.dt_train = PDBBindDataset(
            root=self.root, stage='train', only_pocket=self.only_pocket)
        # self.dt_val = PDBBindDataset(
        #     root=self.root, stage='val', only_pocket=self.only_pocket)

    def train_dataloader(self):
        return pyg.loader.DataLoader(self.dt_train,
                                     batch_size=self.batch_size,
                                     num_workers=self.num_workers,
                                     persistent_workers=True,
                                     shuffle=False)

    # def val_dataloader(self):
    #     return pyg.loader.DataLoader(self.dt_train,
    #                                  batch_size=self.batch_size,
    #                                  num_workers=self.num_workers,
    #                                  persistent_workers=True,
    #                                  shuffle=False)


if __name__ == '__main__':
    data_path = 'data'
    use_pocket = True
    dt_train = PDBBindDataset(root=data_path, stage='train',
                              only_pocket=use_pocket)
