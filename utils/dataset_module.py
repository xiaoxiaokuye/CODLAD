import torch
from torch.utils.data import Dataset, DataLoader, Sampler
import os
import numpy as np
from collections import defaultdict
from torch.nn.utils.rnn import pad_sequence
import copy
import pickle
import time
import sys

from utils.protein_module import traj_to_info, get_atomNum, build_split_dataset, CGDataset
import mdtraj as md
from utils.train_module import reparametrize, batch_to
import random
import re

def extract_number(file_path):
    # 使用正则表达式从文件路径中提取数字
    match = re.search(r'(?:train|val|test)_list_(\d+)\.pkl', file_path)
    if match:
        return int(match.group(1))
    return 0  # 如果没有找到匹配的数字，返回0


class MultiPKLDataset(Dataset):
    def __init__(self, directory_path, split="train", dataname="PED", max_data_pool_size=10000, extract_data=False):
        self.directory_path = directory_path
        self.max_data_pool_size = max_data_pool_size
        self.split = split
        self.dataname = dataname
        self.pkl_files = self._collect_pkl_files()
        self.pkl_files = sorted(self.pkl_files, key=lambda x: extract_number(x[0]))
        self.data_pool = []
        self.total_length = self._compute_total_length()
        self.file_idx = 0
        self.pool_start_idx = 0
        self.extract_data = extract_data
        if split == 'train' and self.extract_data == False:
            self._shuffle_files()

    def _collect_pkl_files(self):
        pkl_files = []

        if self.dataname == "PED":
            preprocess_dir = os.path.dirname(self.directory_path)
            split = os.path.basename(self.directory_path)
            if os.path.exists(f"{preprocess_dir}/{split}_pkl_files_list.pkl"):
                pkl_files = pickle.load(open(f"{preprocess_dir}/{split}_pkl_files_list.pkl", "rb"))
                return pkl_files

        for filename in os.listdir(self.directory_path):
            if filename.endswith('.pkl'):
                file_path = os.path.join(self.directory_path, filename)
                if self.dataname == "PED":
                    with open(file_path, 'rb') as f:
                        data_list = pickle.load(f)
                    pkl_files.append((file_path, len(data_list)))
                elif self.dataname == "Atlas":
                    pkl_files.append((file_path, 301))
                elif self.dataname == "PDB":
                    pkl_files.append((file_path, 1))


        if self.dataname == "PED":
            pickle.dump(pkl_files, open(f"{preprocess_dir}/{split}_pkl_files_list.pkl", "wb"))
        return pkl_files

    def _compute_total_length(self):
        return sum([length for _, length in self.pkl_files])

    def _shuffle_files(self):
        random.shuffle(self.pkl_files)

    def _load_file(self, file_idx):
        file_path, _ = self.pkl_files[file_idx]
        data_list = load_pkl_file(file_path)
        return data_list
    
    def _reset(self):
        """Reset the dataset to start a new epoch."""
        self.file_idx = 0
        self.pool_start_idx = 0
        self.data_pool = []
        if self.split == 'train' and self.extract_data == False:
            self._shuffle_files()

    def _load_next_file(self):
        if self.file_idx < len(self.pkl_files):
            data_list = list(self._load_file(self.file_idx))
            if self.split == 'train':
                random.shuffle(data_list)
            self.data_pool.extend(data_list)
            self.file_idx += 1

            # Keep the data pool size under control
            if len(self.data_pool) > self.max_data_pool_size:
                raw_len = len(self.data_pool)
                # self.data_pool = self.data_pool[-self.max_data_pool_size:]
                self.data_pool = self.data_pool[-(self.max_data_pool_size//2):]
                post_len = len(self.data_pool)
                self.pool_start_idx += raw_len - post_len


    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        if idx == 0:
            self._reset()
        pool_idx = idx - self.pool_start_idx

        while pool_idx >= len(self.data_pool):
            self._load_next_file()
            pool_idx = idx - self.pool_start_idx

        return self.data_pool[pool_idx]
    

def load_pkl_file(file_path):
    with open(file_path, 'rb') as f:
        data_list = pickle.load(f)
    return data_list

def load_dataset_vae(datasets_path, dataname="PED", debug=False):
    start_time = time.time()

    with open(f'{datasets_path}/preproccess_{dataname}/info_dict.pkl', 'rb') as f:
        info_dict = pickle.load(f)
    with open(f'{datasets_path}/preproccess_{dataname}/val_info_dict.pkl', 'rb') as f:
        val_info_dict = pickle.load(f)
    with open(f'{datasets_path}/preproccess_{dataname}/success_list.pkl', 'rb') as f:
        success_list = pickle.load(f)

    valset = MultiPKLDataset(f'{datasets_path}/preproccess_{dataname}/val',split="val", dataname=dataname)
    trainset = MultiPKLDataset(f'{datasets_path}/preproccess_{dataname}/train', split="train", dataname=dataname)

    print(f"load dataset {dataname} success")
    end_time = time.time()
    print(f"Time taken by pickle: {end_time - start_time} seconds")
    return info_dict, val_info_dict, trainset, valset, success_list


def load_dataset(data_path, params, single=True):
    batch_size=4

    if single:
        traj = md.load_pdb(f"{data_path}.pdb")
    else:
        pre_dir = os.path.split(data_path)[0]
        name = os.path.split(data_path)[1]
        traj_temp = md.load(f'{pre_dir}/{name}/{name}_prod_R1_fit.xtc', top=f'{pre_dir}/{name}/{name}.pdb') \
            + md.load(f'{pre_dir}/{name}/{name}_prod_R2_fit.xtc', top=f'{pre_dir}/{name}/{name}.pdb') \
            + md.load(f'{pre_dir}/{name}/{name}_prod_R3_fit.xtc', top=f'{pre_dir}/{name}/{name}.pdb')
        ref = md.load(f'{pre_dir}/{name}/{name}.pdb')
        traj_temp = ref + traj_temp

        # frame_indices = list(range(0, len(traj_temp), 100))
        frame_indices = list(range(0, len(traj_temp), 10000))
        traj = traj_temp[frame_indices]

    heavy_idxs = traj.top.select("mass > 1.1")
    if len(heavy_idxs) != traj.n_atoms and len(heavy_idxs) >  0 :
        traj = traj.atom_slice(heavy_idxs)


    file_names = [ 
        'PED00151e000', 'PED00151e001', 'PED00151e002',
        'PED00011e001', 'PED00143e001', 'PED00145e000',
        'PED00145e001', 'PED00148e001', 'PED00148e002',
        'PED00150e000', 'PED00150e001', 'PED00150e002',
        'PED00145e002',
    ] # some PED data need to be processed
    if data_path.split("/")[-1] in file_names:
        topology = traj.topology
        residues = list(topology.residues)
        residues_to_keep = residues[1:-1]
        atoms_to_keep = [atom.index for residue in residues_to_keep for atom in residue.atoms]
        traj = traj.atom_slice(atoms_to_keep)


    _top = traj.topology
    residues = list(_top.residues)
    residues_to_keep = residues[1:-1]
    atoms_to_keep = [atom.index for residue in residues_to_keep for atom in residue.atoms]
    _traj = traj.atom_slice(atoms_to_keep)
    _top = _traj.top

    
    info, n_cg = traj_to_info(traj)
    n_cg_list, traj_list, info_dict = [n_cg], [traj], {0: info}


    # build testset
    testset_list = []
    for i, traj in enumerate(traj_list):
        atomic_nums, protein_index = get_atomNum(traj)
        table, _ = traj.top.to_dataframe()
        # multiple chain
        table['newSeq'] = table['resSeq'] + 5000*table['chainID']

        nfirst = len(table.loc[table.newSeq==table.newSeq.min()])
        nlast = len(table.loc[table.newSeq==table.newSeq.max()])

        n_atoms = atomic_nums.shape[0]
        n_atoms = n_atoms - (nfirst+nlast)
        atomic_nums = atomic_nums[nfirst:-nlast]

        all_idx = np.arange(len(traj))

        # ndata = len(all_idx)-len(all_idx) % batch_size
        # all_idx = all_idx[:ndata]
        # all_idx = all_idx[:100]

        n_cgs = n_cg_list[i]
        
        testset, mapping = build_split_dataset(traj[all_idx], params, mapping=None, prot_idx=i)
        testset_list.append(testset)

    if single:
        batch_size = min(len(testset), 96)
    else:
        batch_size = min(len(testset), 24)

    testset = torch.utils.data.ConcatDataset(testset_list)
    testloader = DataLoader(testset, batch_size=batch_size, collate_fn=CG_collate, shuffle=False, pin_memory=True, drop_last=False)
    return testloader, info_dict, n_atoms, n_cgs, atomic_nums, _top


def get_norm_feature(feature, feature_type, norm_channel=True, norm_single=False, norm_in=True, dataname="PED"):
    miu_and_sigma_path = "./datasets/miu_and_sigma"

    tail_name = ""
    if norm_single:
        tail_name = "_single"

    feature_name = f"{feature_type}_x"
    
    scale_for_mean = torch.load(f"{miu_and_sigma_path}/{dataname}_{feature_name}_mean{tail_name}.pt")
    scale_for_std = torch.load(f"{miu_and_sigma_path}/{dataname}_{feature_name}_std{tail_name}.pt")
    if norm_in:
        print(f"{feature.shape} {scale_for_mean.shape} {scale_for_std.shape}")
        feature = (feature - scale_for_mean.to(feature.device)) / scale_for_std.to(feature.device)
    else:
        feature = feature * scale_for_std.to(feature.device) + scale_for_mean.to(feature.device)
        
    return feature


def CG_collate(dicts):
    # new indices for the batch: the first one is zero and the
    # last does not matter

    cumulative_atoms = np.cumsum([0] + [d['num_atoms'].item() for d in dicts])[:-1]
    cumulative_CGs = np.cumsum([0] + [d['num_CGs'].item() for d in dicts])[:-1]

    for n, d in zip(cumulative_atoms, dicts):
        d['nbr_list'] = d['nbr_list'] + int(n)
        d['bond_edge_list'] = d['bond_edge_list'] + int(n)
        d['mask_xyz_list'] = d['mask_xyz_list'] + int(n)
        d['bb_NO_list'] = d['bb_NO_list'] + int(n)
        d['interaction_list'] = d['interaction_list'] + int(n)
        d['pi_pi_list'] = d['pi_pi_list'] + int(n)
            
    for n, d in zip(cumulative_CGs, dicts):
        d['CG_mapping'] = d['CG_mapping'] + int(n)
        d['CG_nbr_list'] = d['CG_nbr_list'] + int(n)

    # batching the data
    batch = {}
    for key, val in dicts[0].items():
        if hasattr(val, 'shape') and len(val.shape) > 0:
            batch[key] = torch.cat([
                data[key]
                for data in dicts
            ], dim=0)

        elif type(val) == str: 
            batch[key] = [data[key] for data in dicts]
        else:
            batch[key] = torch.stack(
                [data[key] for data in dicts],
                dim=0
            )

    return batch


def latent_collate_fn(batch):
    # batch: (features, labels, prot_idx)
    # features.shape = labels.shape = (1, seqlen, 36)
    features, labels, prot_idx, ic, batch_raw = zip(*batch)
    batch_raw = [item.item() for item in batch_raw]

    # pad_sequence add 0 in begin or end. ('padding_first'==False)
    # tuple to list [max_len, 36]
    features = list(features)
    labels = list(labels)
    ic = list(ic)

    for i in range(len(ic)):
        # 提取当前 tensor
        tensor = ic[i]

        # 计算 sin 和 cos
        sin_key_angle = torch.sin(tensor[:, :, 1])
        cos_key_angle = torch.cos(tensor[:, :, 1])
        sin_torsion_angle = torch.sin(tensor[:, :, 2])
        cos_torsion_angle = torch.cos(tensor[:, :, 2])

        # 提取边长
        lengths = tensor[:, :, 0]

        # 组合新的 tensor
        new_tensor = torch.stack([lengths, sin_key_angle, cos_key_angle, sin_torsion_angle, cos_torsion_angle], dim=-1)

        # 更新 ic 列表中的 tensor
        ic[i] = new_tensor

    # get max length of features
    feature_lengths = [f.size(0) for f in features]

    features = pad_sequence(features, batch_first=True)
    if labels[0] is not None:
        labels = pad_sequence(labels, batch_first=True)
    else:
        labels = None
    ic = pad_sequence(ic, batch_first=True)

    # create mask
    mask = torch.zeros(features.size(0), max(feature_lengths), dtype=torch.bool)
    for i, fl in enumerate(feature_lengths):
        mask[i, :fl] = True


    # process batch
    batch_raw = CG_collate(batch_raw)


    return features, labels, prot_idx, mask, ic, batch_raw

class ShuffleSampler(Sampler):
    def __init__(self, data_source):
        self.data_source = data_source
        label_to_indexes = defaultdict(list)
        for idx, item in enumerate(self.data_source):
            if len(item) >10:
                label = item["prot_idx"].item()
            else:
                label = item[-3].item()
            label_to_indexes[label].append(idx)     
        self.label_to_indexes = label_to_indexes

    def shuffle_by_class(self):
        label_to_indexes = copy.deepcopy(self.label_to_indexes)

        # Shuffle indexes within each class
        for indexes in label_to_indexes.values():
            np.random.shuffle(indexes)

        # Shuffle classes and combine indexes
        labels = list(label_to_indexes.keys())
        np.random.shuffle(labels)

        return [idx for label in labels for idx in label_to_indexes[label]]

    def __iter__(self):
        return iter(self.shuffle_by_class())

    def __len__(self):
        return len(self.data_source)

class CustomDataset(Dataset):
    def __init__(self, features_dir, labels_dir, id_dir, raw_dir, feature_miu_dir, feature_sigma_dir, label_miu_dir, label_sigma_dir, learn_sigma=False):
        self.features_dir = features_dir # true
        self.labels_dir = labels_dir
        self.id_dir = id_dir
        self.raw_dir = raw_dir # true

        self.features_files = self._sorted_files(features_dir)
        self.raw_files = self._sorted_files(raw_dir)
        # self.id_files = self._sorted_files(id_dir)
        if labels_dir is not None:
            self.labels_files = self._sorted_files(labels_dir)

        self.reparametrize = feature_miu_dir is not None
        self.learn_sigma = learn_sigma
        
        if self.reparametrize:
            self.feature_miu_dir = feature_miu_dir
            self.feature_sigma_dir = feature_sigma_dir
            self.label_miu_dir = label_miu_dir
            self.label_sigma_dir = label_sigma_dir

            if label_miu_dir is not None:
                self.feature_miu_files = self._sorted_files(feature_miu_dir)
                self.feature_sigma_files = self._sorted_files(feature_sigma_dir)
                self.label_miu_files = self._sorted_files(label_miu_dir)
                self.label_sigma_files = self._sorted_files(label_sigma_dir)

    def _sorted_files(self, directory):
        files = os.listdir(directory)
        return sorted(files, key=lambda s: int(s.split('.')[0]))

    def __len__(self):
        # assert len(self.features_files) == len(self.labels_files), \
        #     "Number of feature files and label files should be same"
        return len(self.features_files)

    def __getitem__(self, idx):
        raw_file = self.raw_files[idx]

        prot_idx = None
        batch_raw = np.load(os.path.join(self.raw_dir, raw_file), allow_pickle=True)
        raw = batch_raw.item()['ic'] # (seq,13,3)

        if not self.reparametrize:
            feature_file = self.features_files[idx]
            features = torch.from_numpy(np.load(os.path.join(self.features_dir, feature_file)).squeeze(0)) # (1, seq, 36) → (seq, 36)

            if self.labels_dir is not None:
                label_file = self.labels_files[idx]
                labels = torch.from_numpy(np.load(os.path.join(self.labels_dir, label_file)).squeeze(0)) # (1, seq, 36) → (seq, 36)
            else:
                labels = None
        else:
            feature_miu_file = self.feature_miu_files[idx]
            feature_sigma_file = self.feature_sigma_files[idx]
            label_miu_file = self.label_miu_files[idx]
            label_sigma_file = self.label_sigma_files[idx]

            feature_miu = torch.from_numpy(np.load(os.path.join(self.feature_miu_dir, feature_miu_file))) # (seq, 36)
            feature_sigma = torch.from_numpy(np.load(os.path.join(self.feature_sigma_dir, feature_sigma_file)))
            label_miu = torch.from_numpy(np.load(os.path.join(self.label_miu_dir, label_miu_file)))
            label_sigma = torch.from_numpy(np.load(os.path.join(self.label_sigma_dir, label_sigma_file)))
            
            if self.learn_sigma:
                features = torch.cat([feature_miu, feature_sigma], dim=-1)
                labels = torch.cat([label_miu, label_sigma], dim=-1)

            else:
                features = reparametrize(feature_miu, feature_sigma)
                labels = reparametrize(label_miu, label_sigma)

        return features, labels, prot_idx, raw, batch_raw
    

def get_protein_dataset(feature_path, split, dataname="PED", reparametrize=False, learn_sigma=False):
    features_dir = f"{feature_path}/{dataname}_features_{split}" # true
    raw_dir = f"{feature_path}/{dataname}_raw_{split}" # true
    if dataname == "PED":
        labels_dir = f"{feature_path}/{dataname}_labels_{split}"
        id_dir = f"{feature_path}/{dataname}_id_{split}"
    else:
        labels_dir = None
        id_dir = None

    feature_miu_dir, feature_sigma_dir, label_miu_dir, label_sigma_dir = None, None, None, None
    if reparametrize == True:
        if dataname == "PED":
            feature_miu_dir = f"{feature_path}/{dataname}_feature_miu_{split}" # true
            feature_sigma_dir = f"{feature_path}/{dataname}_feature_sigma_{split}" # true
            label_miu_dir = f"{feature_path}/{dataname}_labels_miu_{split}"
            label_sigma_dir = f"{feature_path}/{dataname}_labels_sigma_{split}"

    dataset = CustomDataset(features_dir, labels_dir, id_dir, raw_dir, feature_miu_dir, feature_sigma_dir, label_miu_dir, label_sigma_dir, learn_sigma)
    return dataset

def get_protein_dataloader(feature_path, split, batch_size, num_workers, dataname="PED", reparametrize=False, learn_sigma=False):
    dataset = get_protein_dataset(feature_path, split, dataname, reparametrize, learn_sigma)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True if split == 'train' else False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=latent_collate_fn,
    )
    return dataloader