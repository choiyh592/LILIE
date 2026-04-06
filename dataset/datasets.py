import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import random
import os
import re

class LongitudinalEEGDataset(Dataset):
    def __init__(self, pairs_df, metadata_csv_path, embeddings_npy_path, n_draws=5):
        """
        Args:
            pairs_df: Dataframe defining before/after splits.
            metadata_csv_path: Path to the CSV describing the embeddings.
            embeddings_npy_path: Path to the pre-computed .npy embeddings.
            n_draws: Number of random instance pairs to draw per patient date per epoch.
        """
        self.n_draws = n_draws
        
        # 1. Load embeddings using memory mapping
        # mmap_mode='r' prevents loading the entire dataset into RAM at once, 
        # which is crucial for large EEG embedding files.
        self.embeddings = np.load(embeddings_npy_path, mmap_mode='r')

        # 2. Dynamically parse the metadata
        meta_df = pd.read_csv(metadata_csv_path)
        
        # Extract ID and date elements dynamically from "ID_YYYY_MM_DD_..."
        parsed_meta = meta_df['group_name'].str.split('_', expand=True)
        meta_df['ID'] = parsed_meta[0].astype(int)
        meta_df['Year'] = parsed_meta[1].astype(int)
        meta_df['Month'] = parsed_meta[2].astype(int)
        meta_df['Day'] = parsed_meta[3].astype(int)

        # Create a dictionary mapping the tuple (ID, Year, Month, Day) -> list of dataset_idxs
        self.date_to_indices = meta_df.groupby(['ID', 'Year', 'Month', 'Day'])['dataset_idx'].apply(list).to_dict()

        # 3. Process pairs and filter for valid ones
        self.valid_pairs = []

        for _, row in pairs_df.iterrows():
            pid = int(row['ID'])
            # Cast to int to ensure "04" and "4" map to the same key
            before_key = (pid, int(row['Year_Before']), int(row['Month_Before']), int(row['Day_Before']))
            after_key = (pid, int(row['Year_After']), int(row['Month_After']), int(row['Day_After']))

            # Only append to our active set if BOTH dates have corresponding instances in the metadata
            if before_key in self.date_to_indices and after_key in self.date_to_indices:
                self.valid_pairs.append((before_key, after_key))

    def __len__(self):
        # Epoch length equals the number of unique valid pairs multiplied by 'n' draws
        return len(self.valid_pairs) * self.n_draws

    def __getitem__(self, idx):
        # 1. Map the global index to a specific patient pair
        pair_idx = idx // self.n_draws
        before_key, after_key = self.valid_pairs[pair_idx]

        # 2. Randomly select one instance for 'before' and one for 'after'
        before_instance_idx = random.choice(self.date_to_indices[before_key])
        after_instance_idx = random.choice(self.date_to_indices[after_key])

        # Convert numpy arrays to PyTorch tensors
        embed_before = torch.tensor(self.embeddings[before_instance_idx], dtype=torch.float32)
        embed_after = torch.tensor(self.embeddings[after_instance_idx], dtype=torch.float32)

        # 3. Randomly assign label and output order
        label = random.randint(0, 1)

        if label == 1:
            # Flipped time order
            seq_1, seq_2 = embed_after, embed_before
        else:
            # Correct time order
            seq_1, seq_2 = embed_before, embed_after

        return seq_1, seq_2, torch.tensor(label, dtype=torch.long)

def get_fold_splits(directory, test_idx):
    """
    Merges all longitudinal_pairs_fold_n.csv files except for test_idx.
    
    Args:
        directory (str): Path to the folder containing the CSVs.
        test_idx (int): The fold number to be used as the test set.
        
    Returns:
        tuple: (train_df, test_df)
    """
    all_files = os.listdir(directory)
    
    # Pattern to match the specific naming convention and extract the fold number
    pattern = re.compile(r'longitudinal_pairs_fold_(\d+)\.csv')
    
    train_dfs = []
    test_df = None
    
    for filename in all_files:
        match = pattern.match(filename)
        if match:
            current_fold_idx = int(match.group(1))
            file_path = os.path.join(directory, filename)
            
            if current_fold_idx == test_idx:
                test_df = pd.read_csv(file_path)
            else:
                train_dfs.append(pd.read_csv(file_path))
    
    # Check if we actually found the test index
    if test_df is None:
        raise ValueError(f"Fold index {test_idx} not found in {directory}")

    # Merge all other folds into one training dataframe
    train_df = pd.concat(train_dfs, ignore_index=True)
    
    return train_df, test_df

def create_train_test_splits(split_csv_dir, metadata_csv_path, embeddings_npy_path, batch_size=32, num_workers=4, test_idx=0, n_draws=5):

    train_df, test_df = get_fold_splits(split_csv_dir, test_idx)

    train_dataset = LongitudinalEEGDataset(
        pairs_df=train_df,
        metadata_csv_path=metadata_csv_path,
        embeddings_npy_path=embeddings_npy_path,
        n_draws=n_draws
    )

    test_dataset = LongitudinalEEGDataset(
        pairs_df=test_df,
        metadata_csv_path=metadata_csv_path,
        embeddings_npy_path=embeddings_npy_path,
        n_draws=n_draws
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,       
        pin_memory=True      
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,       
        pin_memory=True      
    )

    #     Example iteration
    #     for batch_idx, (seq_1, seq_2, labels) in enumerate(train_loader):
    #         print(f"Batch {batch_idx}: Seq1 shape {seq_1.shape}, Seq2 shape {seq_2.shape}, Labels shape {labels.shape}")
    #         break

    return train_loader, test_loader