# the first flag below was False when we tested this script but True makes A100 training a lot faster:
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torch.utils.data import DataLoader
import numpy as np
import argparse
import os
from collections import Counter
import pickle
import pandas as pd
import sys
sys.path.append('./utils')

from utils.model_module import get_vae_model
from utils.train_module import set_random_seed, batch_to
from utils.dataset_module import CG_collate, MultiPKLDataset
from utils.protein_module import create_info_dict, build_split_dataset

#################################################################################
#                             Training Helper Functions                         #
#################################################################################
def statistics_codebook(updated_histogram, args, max_x=4096):
    import csv
    with open(f'{args.features_path}/counter_data.csv', 'w', newline='') as file:
        writer = csv.writer(file)

        # Write the header
        writer.writerow(['Value', 'Count'])

        # Write the data
        for value, count in updated_histogram.items():
            writer.writerow([value, count])


    import matplotlib.pyplot as plt
    values = list(updated_histogram.keys())
    counts = list(updated_histogram.values())

    # Creating the histogram
    plt.figure(figsize=(10, 6))
    plt.bar(values, counts)

    # Adding labels and title
    plt.xlabel('Values')
    plt.ylabel('Counts')
    plt.title('Histogram from Counter')

    # Showing the plot
    plt.xlim(0, max_x)  # 设置x轴的范围为0到8192
    plt.grid(axis='y')
    plt.show()
    plt.savefig(f'{args.features_path}/value_count_histogram.png')

def update_histogram(existing_lists, new_list):
    """
    Update the histogram of index occurrences with a new list of indices.
    
    :param existing_lists: List of lists containing previously received indices.
    :param new_list: New list of indices to be added.
    :return: Updated histogram of index occurrences.
    """
    # Append the new list to the existing lists
    existing_lists.append(new_list)

    # Flatten the list of lists and count occurrences
    flattened_indices = [index for sublist in existing_lists for index in sublist]
    index_counts = Counter(flattened_indices)

    return index_counts
existing_lists = []

def get_file_list(dataname, data_path):
    if dataname.split('.')[-1] == "txt":
        with open(f'{dataname}','r') as fh:
            # filename_list = fh.readlines()
            # filename_list = [label.strip('\n') for label in filename_list]
            filename_list = [label.strip() for label in fh]
            filename_list = [data_path + label for label in filename_list]
    elif dataname.split('.')[-1] == "csv":
        df = pd.read_csv(dataname, index_col='name')
        filename_list = df.index.tolist()
        filename_list = [data_path + label for label in filename_list]
    else:
        raise ValueError(f"Unsupported file extension: {dataname.split('.')[-1]}")
    
    return filename_list

#################################################################################
#                                  main Loop                                #
#################################################################################

def process_pdbdata_to_icdataset(dataset_path="./datasets/protein", dataname="PED"):

    params = {
        'atom_cutoff': 9.0,
        'cg_cutoff': 21.0,
        'edgeorder': 2
    }

    success_list = []

    if dataname == "PED":
        train_id = "PED_train_id.txt"
        valid_id = "PED_val_id.txt"
        test_id = "PED_test_id.txt"
        subdir = "pedfiles_genzprot"
    elif dataname == "Atlas": 
        train_id = "new_atlas_train.csv"
        valid_id = "new_atlas_val.csv"
        test_id = "new_atlas_test.csv"
        subdir = "atlas_data"
    elif dataname == "PDB": 
        train_id = "new_PDB_train_id.txt"
        valid_id = "new_PDB_val_id.txt"
        test_id = "new_PDB_test_id.txt"
        subdir = "PDB_Diamondback"

    data_path = f"{dataset_path}/{dataname}/"
    subdir = f"{dataset_path}/{dataname}/{subdir}/"

    #################################### get pdb file list
    test_label_list = get_file_list(f'{data_path}/{test_id}', subdir)
    val_label_list = get_file_list(f'{data_path}/{valid_id}', subdir)
    train_label_list = get_file_list(f'{data_path}/{train_id}', subdir)
    print("num training data entries", len(train_label_list))
    print("num validation data entries", len(val_label_list))
    print("num test data entries", len(test_label_list))

    os.makedirs(f'{dataset_path}/preproccess_{dataname}', exist_ok=True)
    os.makedirs(f'{dataset_path}/preproccess_{dataname}/test', exist_ok=True)
    os.makedirs(f'{dataset_path}/preproccess_{dataname}/val', exist_ok=True)
    os.makedirs(f'{dataset_path}/preproccess_{dataname}/train', exist_ok=True)
    ################################### load PED traj and get info
    single = False if dataname == "Atlas" else True

    def process_data(label_list, subset, offset=0):
        n_cg_list, traj_list, info_dict, ped_name = create_info_dict(label_list, single)
        for i, traj in enumerate(traj_list):
            prot_idx = i + offset
            label = label_list[i]
            n_cgs = n_cg_list[i]
            dataset, mapping = build_split_dataset(traj, params, mapping=None, n_cgs=n_cgs, prot_idx=prot_idx)
            print(f"done dataset:id {prot_idx}, {label}")
            with open(f'{dataset_path}/preproccess_{dataname}/{subset}/{subset}_list_{prot_idx}.pkl', 'wb') as f:
                pickle.dump(dataset, f)
            success_list.append(label)
        return info_dict
    
    print("TRAINING DATA")
    info_dict = process_data(train_label_list, 'train')

    print("VALID DATA")
    val_info_dict = process_data(val_label_list, 'val', offset=len(train_label_list))

    print("TEST DATA")
    test_info_dict = process_data(test_label_list, 'test', offset=len(train_label_list) + len(val_label_list))


    val_info_dict = {k+len(train_label_list): val_info_dict[k] for k in val_info_dict.keys()}
    test_info_dict = {k+len(train_label_list)+len(val_label_list): test_info_dict[k] for k in test_info_dict.keys()}
    info_dict.update(val_info_dict)
    info_dict.update(test_info_dict)

    not_success_list = [label for label in (train_label_list + val_label_list + test_label_list) if label not in success_list]
    print('not success: ', not_success_list)

    ################################### save the data
    with open(f'{dataset_path}/preproccess_{dataname}/info_dict.pkl', 'wb') as f:
        pickle.dump(info_dict, f)
    with open(f'{dataset_path}/preproccess_{dataname}/val_info_dict.pkl', 'wb') as f:
        pickle.dump(val_info_dict, f)
    with open(f'{dataset_path}/preproccess_{dataname}/test_info_dict.pkl', 'wb') as f:
        pickle.dump(test_info_dict, f)
    with open(f'{dataset_path}/preproccess_{dataname}/success_list.pkl', 'wb') as f:
        pickle.dump(success_list, f)
    with open(f'{dataset_path}/preproccess_{dataname}/not_success_list.pkl', 'wb') as f:
        pickle.dump(not_success_list, f)


def extract_from_vae(args):
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_random_seed(args.global_seed)
    print(f"Starting training on device={device}.")

    # Setup a feature folder:
    os.makedirs(args.features_path, exist_ok=True)
    for split in ["train", "valid"]:
        os.makedirs(os.path.join(args.features_path, f'{args.dataname}_features_{split}'), exist_ok=True)
        os.makedirs(os.path.join(args.features_path, f'{args.dataname}_labels_{split}'), exist_ok=True)
        os.makedirs(os.path.join(args.features_path, f'{args.dataname}_raw_{split}'), exist_ok=True)
        os.makedirs(os.path.join(args.features_path, f'{args.dataname}_feature_miu_sigma_{split}'), exist_ok=True)

    # ------------------------------------------------------------
    # load vae model
    vae_model, params = get_vae_model(f"{args.vae_type}",device=device)
    vae_model.eval()
    print(f"Finish loading vae model {args.vae_type}")


    datasets_path = "./datasets/protein"
    with open(f'{datasets_path}/preproccess_{args.dataname}/success_list.pkl', 'rb') as f:
        success_list = pickle.load(f)
    valset = MultiPKLDataset(f'{datasets_path}/preproccess_{args.dataname}/val',split="val", dataname=args.dataname, extract_data=True)
    trainset = MultiPKLDataset(f'{datasets_path}/preproccess_{args.dataname}/train', split="train", dataname=args.dataname, extract_data=True)
    trainloader = DataLoader(trainset, batch_size=1, collate_fn=CG_collate, shuffle=False, pin_memory=True)
    valloader = DataLoader(valset, batch_size=1, collate_fn=CG_collate, shuffle=False, pin_memory=True)

    # ------------------------------------------------------------------------   
    # extract features 
    train_steps = 0
    max_seqlen = 0
    prot_index = 0
    prot_idx = 0 
    updated_histogram = None
    train_x_list, train_y_list = [], []


    ########################################################################################
    # train
    ########################################################################################
    for batch in trainloader:
        # ------------------------------------------------------------------------
        # PED_raw_train and PED_id_train
        np.save(f'{args.features_path}/{args.dataname}_raw_train/{train_steps}.npy', batch)  

        # ------------------------------------------------------------------------
        # PED_features_train
        batch = batch_to(batch, device)
        with torch.no_grad():

            x, index, _, _, _, _, _ = vae_model.get_latent_wovq(batch) # wovq features
            # x, index, _, _, _, _, _ = vae_model.get_latent(batch)
            np.save(f'{args.features_path}/{args.dataname}_features_train/{train_steps}.npy', x.detach().cpu().numpy())
            train_x_list.append(x.detach().cpu())

            # # ------------------------------------------------------------------------
            # for vq index check
            if index != None:
                index_list = index.tolist()[0]
                updated_histogram = update_histogram(existing_lists, index_list)

        train_steps += 1
        if train_steps % 100 == 0:
            print(f"train_dataset:{train_steps}")
        


    ########################################################################################
    # Validation
    ########################################################################################
    valid_steps = 0
    for batch in valloader:
        # ------------------------------------------------------------------------
        # PED_raw_valid and PED_id_valid
        np.save(f'{args.features_path}/{args.dataname}_raw_valid/{valid_steps}.npy', batch)  

        # ------------------------------------------------------------------------
        # PED_features_valid
        batch = batch_to(batch, device)
        with torch.no_grad():
            x, index, _, _, _, _, _ = vae_model.get_latent_wovq(batch)
            # x, index, _, _, _, _, _ = vae_model.get_latent(batch)
            np.save(f'{args.features_path}/{args.dataname}_features_valid/{valid_steps}.npy', x.detach().cpu().numpy())

            # for vq index check
            if index != None:
                index_list = index.tolist()[0]
                updated_histogram = update_histogram(existing_lists, index_list)
        
        valid_steps += 1

    


    # # ------------------------------------------------------------------------
    # raw data norm

    all_train_x = torch.cat(train_x_list, dim=1)
    torch.save(train_x_list, f'./datasets/miu_and_sigma/{args.dataname}_{args.vae_type}_x_list.pt')

    all_train_x_mean, all_train_x_std = all_train_x.mean(dim=[0,1]), all_train_x.std(dim=[0,1])
    torch.save(all_train_x_mean, f'./datasets/miu_and_sigma/{args.dataname}_{args.vae_type}_x_mean.pt')
    torch.save(all_train_x_std, f'./datasets/miu_and_sigma/{args.dataname}_{args.vae_type}_x_std.pt')

    # ------------------------------------------------------------------------
    # statistics the codebook usage
    if updated_histogram != None:
        statistics_codebook(updated_histogram, args, max_x=4096)



if __name__ == "__main__":
    # Default args here will train DiT-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--features-path", type=str, default="./features")  # need to change
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--vae_type", type=str, default="N6")  # need to change
    parser.add_argument("--dataname", type=str, default="PED", choices=["PED", "Atlas", "PDB"])  # need to change
    parser.add_argument("--process_data", type=bool, default=False)  # need to change
    parser.add_argument("--extract_features", type=bool, default=False)  # need to change
    args = parser.parse_args()

    if args.process_data:
        process_pdbdata_to_icdataset(dataname=args.dataname)
        print("Finish processing data")
    
    if args.extract_features:
        extract_from_vae(args)
        print("Finish extracting features")

    print("Finish all")
