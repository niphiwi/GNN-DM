from models.model import GNNModule
from data.synthetic import SyntheticDataset, SyntheticDataModule
import pytorch_lightning as pl
import torch
from torch.utils.data import ConcatDataset
from torch_geometric.loader import DataLoader
import argparse

# Define argparse arguments
parser = argparse.ArgumentParser(description='Train a model with PyTorch Lightning')
parser.add_argument('--gpu', type=int, default=0, help='GPU')
parser.add_argument('--name', type=str, default='train', help='Name of experiment (for logging)')
parser.add_argument('--epochs', type=int, default=20, help='Epochs to train')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
args = parser.parse_args()

torch.set_float32_matmul_precision('medium')

hparams = {
    "learning_rate": 1e-4,
    "weight_decay": 1e-5,
    "num_nodes": 750,
    "p_sensor": None,
    "hidden_dim": 64, 
    "heads": 2, 
    "use_residual": True,
    "dropout": 0.2,
    "dropout_edge": 0.1,
}


model = GNNModule(hparams=hparams)

# datamodule = SyntheticDataModule(seq_len=10, batch_size=args.batch_size, num_workers=0, shuffle=True, p_sensor=hparams.get("p_sensor"))
train_dataset = SyntheticDataset(root="data/30x25", type="train")  
val_dataset = SyntheticDataset(root="data/30x25", type="valid")  


tb_logger = pl.loggers.TensorBoardLogger(save_dir="logs", name=args.name, default_hp_metric=True)
trainer = pl.Trainer(
    logger=tb_logger,
    max_epochs=args.epochs,
    enable_progress_bar=True,
    accelerator="auto" if torch.cuda.is_available() else "cpu",
    devices=[args.gpu],
    fast_dev_run=False,
)

combined_dataset = ConcatDataset([train_dataset, val_dataset])
combined_loader = DataLoader(combined_dataset, 
                            batch_size=args.batch_size, 
                            shuffle=True)

trainer.fit(model, combined_loader)