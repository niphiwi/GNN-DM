import torch
import torch.nn as nn
from torch.optim import Adam
from torch_geometric.nn import GATv2Conv
from torch_geometric.nn.pool import global_mean_pool
import pytorch_lightning as pl
from torch_geometric.utils import subgraph

def filter_sensors(data):
    sensors = data.known[:,-1]
    sensors_idx = torch.where(sensors)[0]
    edge_index, edge_attr = subgraph(sensors_idx, data.edge_index, data.edge_attr)

    return edge_index, edge_attr

def filter_sensors_to_virtual(data):
    sensor = data.known[:, -1].bool()
    sensors_idx = torch.where(sensor)[0]
    virtual_idx = torch.where(~sensor)[0]

    # Create a mask for edges that connect sensors to non-sensors
    edge_mask = torch.isin(data.edge_index[0], sensors_idx) & torch.isin(data.edge_index[1], virtual_idx) | \
                torch.isin(data.edge_index[1], sensors_idx) & torch.isin(data.edge_index[0], virtual_idx)

    # Apply the mask to filter edges and edge attributes
    edge_index = data.edge_index[:, edge_mask]
    edge_attr = data.edge_attr[edge_mask] if data.edge_attr is not None else None

    return edge_index, edge_attr

class GNN(nn.Module):
    def __init__(
        self,
        input_dim=10,
        hidden_dim=64,
        output_dim=1,
        heads=1,
        p_dropout=0.3,
        p_dropout_edge=0.3,
        use_residual=True,
    ):
        super(GNN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.heads = heads
        self.p_dropout = p_dropout
        self.p_dropout_edge = p_dropout_edge
        self.use_residual = use_residual

        # [first_dim] = x + mask
        first_dim = input_dim + input_dim

        self.elu = nn.ELU()
        self.dropout = nn.Dropout(p=self.p_dropout)

        # MLP for global representation
        self.global_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU()
        )

        # Convolution layers
        self.conv1 = GATv2Conv(first_dim + hidden_dim, hidden_dim, edge_dim=1, heads=heads, add_self_loops=False)
        self.conv2 = GATv2Conv(hidden_dim*heads, hidden_dim, heads=heads, edge_dim=1, add_self_loops=False)
        self.conv3 = GATv2Conv(hidden_dim*heads, hidden_dim, heads=heads, edge_dim=1, add_self_loops=False)

        if self.use_residual:
            self.residual_ln = torch.nn.Linear(first_dim + hidden_dim, hidden_dim*heads)
        self.readout = torch.nn.Linear(hidden_dim*heads, output_dim)


    def forward(self, data):
        # Pool 
        is_sensor = data.known[:,-1]
        try:
            batch = data.batch
            batch_sensors = batch[is_sensor == True]
        except TypeError:
            batch = torch.zeros(data.num_nodes, device=data.x.device, dtype=int)
            batch_sensors = batch[is_sensor == True]
        g = global_mean_pool(x=data.x[is_sensor == True], batch=batch_sensors)
        g_transformed = self.global_mlp(g)
        g_broadcasted = g_transformed[batch]

        x = torch.cat([data.x, data.known, g_broadcasted], dim=-1)

        x = torch.cat([data.x, data.known, g_broadcasted], dim=-1)
        edge_index = data.edge_index
        edge_attr = data.edge_attr

        # First layer (only sensors to sensors)
        edge_index, edge_attr = filter_sensors(data)
        out1 = self.conv1(x, edge_index, edge_attr)
        out1 = self.elu(out1)
        out1 = self.dropout(out1)
        if self.use_residual:
            out1 = out1 + self.residual_ln(x)

        # Second Layer (only sensors to virtual)
        edge_index, edge_attr = filter_sensors_to_virtual(data)
        out2 = self.conv2(out1, edge_index, edge_attr)
        out2 = self.elu(out2)
        out2 = self.dropout(out2)
        if self.use_residual:
            out2 = out2 + out1

        # Third Layer (all nodes)
        out3 = self.conv3(out2, data.edge_index, data.edge_attr)

        # Output
        pred = self.readout(out3)

        return pred


class GNNModule(pl.LightningModule):
    def __init__(self, hparams: dict = None):
        super().__init__()
        if hparams is None:
            hparams = {}
        self.save_hyperparameters()

        self.lr = hparams.get("learning_rate", 1e-4)
        self.weight_decay = hparams.get("weight_decay", 0.0005)
        self.fine_tuning = hparams.get("fine_tuning", False)
        input_dim = hparams.get("seq_len", 10)
        self.model = GNN(
            input_dim=input_dim,
            hidden_dim=hparams.get("hidden_dim", 64),
            output_dim=1,
            heads=hparams.get("heads", 2),
            p_dropout=hparams.get("dropout", 0.2),
            p_dropout_edge=hparams.get("dropout_edge", 0),
            use_residual=hparams.get("use_residual", True),
        )
        self.loss_fn = nn.MSELoss()

    def forward(self, data):
        return self.model(data)
    
    def configure_optimizers(self):
        return Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def training_step(self, batch, batch_idx):
        y = batch.y[:, -1].unsqueeze(1)
        y_hat = self.forward(batch)

        if self.fine_tuning:
            sensor_mask = batch.sensor
            # Only care for nodes that are sensors
            y_hat = y_hat[sensor_mask]
            y = y[sensor_mask]
        else:
            # Pretraining: don't filter anything
            pass

        loss = self.loss_fn(y_hat, y)

        self.log("train_loss", loss, on_epoch=True, sync_dist=True, prog_bar=True, batch_size=len(batch.batch))
        return loss

    def validation_step(self, batch, batch_idx):
        y = batch.y[:, -1].unsqueeze(1)
        y_hat = self.forward(batch)

        if self.fine_tuning:
            sensor_mask = batch.sensor
            # sensor_mask = (sensor > 0).any(dim=-1)
            # Only care for nodes that are sensors
            y_hat = y_hat[sensor_mask]
            y = y[sensor_mask]
        else:
            # Pretraining: don't filter anything
            pass

        loss = self.loss_fn(y_hat, y)
        self.log("val_loss", loss, on_epoch=True, sync_dist=True, prog_bar=True, batch_size=len(batch.batch))

    def freeze(self):
        for name, param in self.named_parameters():
            if "readout" in name:
                param.requires_grad = True 
            elif "conv3" in name:
                param.requires_grad = True
            # elif "conv2" in name:
            #     param.requires_grad = True
            else:
                param.requires_grad = False