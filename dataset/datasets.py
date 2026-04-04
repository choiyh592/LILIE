import torch
import numpy as np
from torch.utils.data import Dataset
import random

#TODO : Refine dataset for multiple calls of random assignments!!

class LongitudinalDataset(Dataset):
    def __init__(self):
        super().__init__()

        # Placeholders
        self.all_pairs = []
        self.before_embs = []
        self.after_embs = []

    @classmethod
    def create_instance_ids(num_patients, num_samples_per_patient):
        pass

    def __len__(self):
        # The total number of combinations across all subjects
        return len(self.all_pairs)
    
    def __getitem__(self, idx):
        before_idx, after_idx = self.all_pairs[idx]
        label = self.labels[idx]
        
        # Determine order based on the fixed label
        if label == 1:
            x_0, x_1 = self.before_embs[before_idx], self.after_embs[after_idx]
        else:
            x_0, x_1 = self.after_embs[after_idx], self.before_embs[before_idx]
            
        return (
            torch.from_numpy(x_0).float(), 
            torch.from_numpy(x_1).float(), 
            torch.tensor(label, dtype=torch.long)
        )

class RandomLongitudinalDataset(LongitudinalDataset):
    def __init__(self, before_path, after_path, subject_ids, len_multiplier=5):
        """
        Args:
            before_path (str): Path to 'before_embeddings.npy'
            after_path (str): Path to 'after_embeddings.npy'
            subject_ids (np.array or list): A list of IDs identifying the subject 
                                            for each row (e.g., [S1, S1, S2, S2...])
        """
        self.before_embs = np.load(before_path)
        self.after_embs = np.load(after_path)
        self.subject_ids = np.array(subject_ids)
        
        # Create a mapping: subject_id -> list of indices belonging to them
        self.subject_map = {}
        for idx, s_id in enumerate(self.subject_ids):
            if s_id not in self.subject_map:
                self.subject_map[s_id] = []
            self.subject_map[s_id].append(idx)
            
        self.unique_ids = list(self.subject_map.keys())

    # Override for stochastic draws
    def __getitem__(self, idx):
        # 1. Identify the subject for the current index
        current_subject = self.subject_ids[idx]
        
        # 2. Get all possible indices for this specific subject
        subject_indices = self.subject_map[current_subject]
        
        # 3. Randomly select an index from the 'after' set for this same subject
        # This provides the "longitudinal" variety
        random_after_idx = random.choice(subject_indices)
        
        emb_before = self.before_embs[idx]
        emb_after = self.after_embs[random_after_idx]
        
        # 4. Determine order (1: Before->After, 0: After->Before)
        label = np.random.randint(0, 2)
        
        if label == 1:
            x_0, x_1 = emb_before, emb_after
        else:
            x_0, x_1 = emb_after, emb_before
            
        return (
            torch.from_numpy(x_0).float(), 
            torch.from_numpy(x_1).float(), 
            torch.tensor(label, dtype=torch.long)
        )

class AllPairsLongitudinalDataset(LongitudinalDataset):
    def __init__(self, before_path, after_path, subject_ids):
        """
        Args:
            before_path (str): Path to 'before_embeddings.npy'
            after_path (str): Path to 'after_embeddings.npy'
            subject_ids (np.array or list): IDs identifying the subject for each row
        """
        self.before_embs = np.load(before_path)
        self.after_embs = np.load(after_path)
        self.subject_ids = np.array(subject_ids)
        
        # 1. Map each subject ID to their list of indices
        self.subject_map = {}
        for idx, s_id in enumerate(self.subject_ids):
            if s_id not in self.subject_map:
                self.subject_map[s_id] = []
            self.subject_map[s_id].append(idx)
            
        # 2. Pre-compute all possible (before_idx, after_idx) pairs for each subject
        self.all_pairs = []
        self.subject_id_map = []
        for s_id, indices in self.subject_map.items():
            # This creates a Cartesian product of the subject's indices
            # Result: every 'before' sample paired with every 'after' sample for this ID
            for b_idx in indices:
                for a_idx in indices:
                    self.all_pairs.append((b_idx, a_idx))
                    self.subject_id_map.append(s_id)

class KPairsLongitudinalDataset(LongitudinalDataset):
    def __init__(self, before_path, after_path, subject_ids, k_pairs=8, seed=42):
        """
        Args:
            before_path (str): Path to 'before_embeddings.npy'
            after_path (str): Path to 'after_embeddings.npy'
            subject_ids (list): IDs for each row
            k_pairs (int): Number of random pairs to draw per subject
        """
        self.before_embs = np.load(before_path)
        self.after_embs = np.load(after_path)
        self.subject_ids = np.array(subject_ids)
        self.k_pairs = k_pairs
        
        # 1. Map each subject ID to their indices
        self.subject_map = {}
        for idx, s_id in enumerate(self.subject_ids):
            if s_id not in self.subject_map:
                self.subject_map[s_id] = []
            self.subject_map[s_id].append(idx)
            
        # 2. Generate Fixed-K pairs per subject
        self.all_pairs = []
        self.subject_id_map = []
        random.seed(seed)
        
        for s_id, indices in self.subject_map.items():
            # Generate all possible combinations for this subject
            possible_pairs = [(b, a) for b in indices for a in indices]
            
            # If they have fewer than k possible pairs, take all (or oversample)
            if len(possible_pairs) <= self.k_pairs:
                sampled = possible_pairs
            else:
                sampled = random.sample(possible_pairs, self.k_pairs)
            
            self.all_pairs.extend(sampled)
            self.subject_id_map.extend([s_id] * len(sampled))

        # 3. Pre-assign deterministic labels to avoid "moving target" validation
        np.random.seed(seed)
        self.labels = np.random.randint(0, 2, size=len(self.all_pairs))

