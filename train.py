from dataset.datasets import create_train_test_splits
from models.models import LILIE

import torch.utils
import lightning as L
from torch.utils.data import DataLoader

from lightning.pytorch.loggers import CSVLogger

import warnings

from lightning.pytorch.callbacks import ModelCheckpoint

import argparse

class DelayedCheckpoint(ModelCheckpoint):
    def __init__(self, start_epoch, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_epoch = start_epoch

    def on_validation_end(self, trainer, pl_module):
        if trainer.current_epoch >= self.start_epoch:
            super().on_validation_end(trainer, pl_module)

def parse_args():
    parser = argparse.ArgumentParser(description="LILIE Model Training Script")

    # --- Data Paths ---
    parser.add_argument("--split_csv_dir", type=str, required=True, help="Path to split CSVs")
    parser.add_argument("--metadata_csv", type=str, required=True, help="Path to metadata CSV")
    parser.add_argument("--embeddings_npy", type=str, required=True, help="Path to embeddings .npy file")
    parser.add_argument("--log_dir", type=str, default="./exp_pl", help="Directory for logs")
    parser.add_argument("--exp_name", type=str, default="eeg_experiment", help="Experiment name")

    # --- DataLoader Args ---
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--test_idx", type=int, default=1, help="Index for cross-validation split. Starts from 1")
    parser.add_argument("--n_draws", type=int, default=5)

    # --- Model Hyperparameters ---
    parser.add_argument("--input_dim", type=int, default=256)
    parser.add_argument("--embedding_size", type=int, default=256)
    parser.add_argument("--pool_method", type=str, default="Attentive", choices=["Attentive", "Mean", "Max"])
    parser.add_argument("--clf_method", type=str, default="NN", choices=["NN", "Linear"])

    # --- Trainer Args ---
    parser.add_argument("--max_epochs", type=int, default=1000)
    parser.add_argument("--start_saving_epoch", type=int, default=20, help="Epoch to start checkpointing")
    parser.add_argument("--accelerator", type=str, default="gpu", help="cpu, gpu, or auto")
    parser.add_argument("--devices", type=str, default="auto", help="Number of devices or 'auto'")

    return parser.parse_args()

if __name__ == "__main__":
    warnings.filterwarnings("ignore", message="pkg_resources is deprecated*")
    warnings.filterwarnings("ignore", message=".*No negative samples in targets*")

    args = parse_args()
    
    train_loader, test_loader = create_train_test_splits(
        split_csv_dir=args.split_csv_dir,
        metadata_csv_path=args.metadata_csv,
        embeddings_npy_path=args.embeddings_npy,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        test_idx=args.test_idx,
        n_draws=args.n_draws
    )

    model = LILIE(
        input_dim=args.input_dim, 
        embedding_size=args.embedding_size, 
        num_classes=2, 
        pool_method=args.pool_method, 
        clf_method=args.clf_method
    )

    logger = CSVLogger(args.log_dir, name=args.exp_name)

    # Note: 'devices' is passed as an int if numeric, else as string
    devs = int(args.devices) if args.devices.isdigit() else args.devices

    # Usage: Start saving only after epoch 10
    checkpoint_callback = DelayedCheckpoint(
        start_epoch=args.start_saving_epoch,
        monitor="val_auroc",
        mode="max",
        save_top_k=1,
        filename="best-eeg-{epoch:02d}-{val_auroc:.4f}-{val_acc:.4f}"
    )

    trainer = L.Trainer(
            logger=logger, 
            callbacks=[checkpoint_callback], 
            strategy="ddp" if args.accelerator == "gpu" else "auto",
            max_epochs=args.max_epochs,
            accelerator=args.accelerator,
            devices=devs
        )
    
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=test_loader)