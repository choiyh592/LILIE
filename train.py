from dataset.datasets import RandomLongitudinalDataset, AllPairsLongitudinalDataset, KPairsLongitudinalDataset
from models.models import LILIE

import torch.utils
import lightning as L
from torch.utils.data import DataLoader

from lightning.pytorch.loggers import CSVLogger

import warnings

from lightning.pytorch.callbacks import ModelCheckpoint
from sklearn.model_selection import GroupShuffleSplit

class DelayedCheckpoint(ModelCheckpoint):
    def __init__(self, start_epoch, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_epoch = start_epoch

    def on_validation_end(self, trainer, pl_module):
        if trainer.current_epoch >= self.start_epoch:
            super().on_validation_end(trainer, pl_module)

def check_patient_overlap(dataset, train_idx, test_idx):
    train_patients = set([dataset.subject_id_map[i] for i in train_idx])
    val_patients = set([dataset.subject_id_map[i] for i in test_idx])

    overlap = train_patients.intersection(val_patients)

    if len(overlap) > 0:
        print(f"CRITICAL LEAK: {len(overlap)} patients are in both Train and Val!")
        print(f"Patient IDs: {overlap}")
        return False
    else:
        print("Split is clean: No patient overlap.")
        return True

if __name__ == "__main__":
    warnings.filterwarnings("ignore", message="pkg_resources is deprecated*")
    warnings.filterwarnings("ignore", message=".*No negative samples in targets*")
    
    samples_per_patient = 4
    pair_ids = [i // samples_per_patient for i in range(48 * 4)]

    subject_ids_4 = [1290972, 1290972, 1290972, 1290972, 1391736, 1391736, 1391736, 1391736, 1621015, 1621015, 1621015, 1621015, 1925754, 1925754, 1925754, 1925754, 20046947, 20046947, 20046947, 20046947, 20046947, 20046947, 20046947, 20046947, 20078516, 20078516, 20078516, 20078516, 20118630, 20118630, 20118630, 20118630, 20134679, 20134679, 20134679, 20134679, 2256166, 2256166, 2256166, 2256166, 2676587, 2676587, 2676587, 2676587, 2950427, 2950427, 2950427, 2950427, 2958837, 2958837, 2958837, 2958837, 309480, 309480, 309480, 309480, 3450576, 3450576, 3450576, 3450576, 3753849, 3753849, 3753849, 3753849, 382566, 382566, 382566, 382566, 475077, 475077, 475077, 475077, 520029, 520029, 520029, 520029, 5328122, 5328122, 5328122, 5328122, 5328122, 5328122, 5328122, 5328122, 6066494, 6066494, 6066494, 6066494, 6207136, 6207136, 6207136, 6207136, 6207136, 6207136, 6207136, 6207136, 622588, 622588, 622588, 622588, 6264731, 6264731, 6264731, 6264731, 6434045, 6434045, 6434045, 6434045, 6537579, 6537579, 6537579, 6537579, 6603372, 6603372, 6603372, 6603372, 6804555, 6804555, 6804555, 6804555, 6869574, 6869574, 6869574, 6869574, 6869574, 6869574, 6869574, 6869574, 6894993, 6894993, 6894993, 6894993, 6894993, 6894993, 6894993, 6894993, 6908730, 6908730, 6908730, 6908730, 6908730, 6908730, 6908730, 6908730, 830639, 830639, 830639, 830639, 8596341, 8596341, 8596341, 8596341, 9107603, 9107603, 9107603, 9107603, 9168177, 9168177, 9168177, 9168177, 9244082, 9244082, 9244082, 9244082, 9244082, 9244082, 9244082, 9244082, 9246548, 9246548, 9246548, 9246548, 9385345, 9385345, 9385345, 9385345, 9418325, 9418325, 9418325, 9418325, 9469406, 9469406, 9469406, 9469406, 947920, 947920, 947920, 947920, 947920, 947920, 947920, 947920]
    subject_ids = []
    for i, id in enumerate(subject_ids_4):
        if i % 4 == 0:
            subject_ids = subject_ids + [id] * 8

    dataset = KPairsLongitudinalDataset('/home/yhchoi/EEG_Data/Create_L_NL_Control/experiment_2/before_embeddings_lecanemab.npy', 
                                          '/home/yhchoi/EEG_Data/Create_L_NL_Control/experiment_2/after_embeddings_lecanemab.npy', 
                                          subject_ids=pair_ids)

    gss = GroupShuffleSplit(n_splits=1, train_size=0.6, random_state=42)
    train_idx, test_idx = next(gss.split(X=range(len(dataset)), groups=subject_ids))

    if check_patient_overlap(dataset, train_idx, test_idx):
        train_dataset = torch.utils.data.Subset(dataset, train_idx)
        test_dataset = torch.utils.data.Subset(dataset, test_idx)

        train_loader = DataLoader(
            train_dataset, 
            batch_size=32, 
            shuffle=True, 
            num_workers=4, 
            persistent_workers=True
        )

        test_loader = DataLoader(
            test_dataset, 
            batch_size=32, 
            shuffle=False, 
            num_workers=4
        )

        model = LILIE(input_dim=256, embedding_size=256, num_classes=2, pool_method="NN", clf_method="NN")
        logger = CSVLogger("/home/yhchoi/EEG_Data/Create_L_NL_Control/exp_pl", name="eeg_experiment")

        # Usage: Start saving only after epoch 10
        checkpoint_callback = DelayedCheckpoint(
            start_epoch=10,
            monitor="val_auroc",
            mode="max",
            save_top_k=1,
            filename="best-eeg-{epoch:02d}-{val_auroc:.2f}"
        )

        trainer = L.Trainer(logger=logger, callbacks=[checkpoint_callback], strategy="ddp_find_unused_parameters_true", max_epochs=100)
        trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=test_loader)