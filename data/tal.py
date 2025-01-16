import pickle
import joblib
import torch
from tqdm import tqdm

from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T

import pytorch_lightning as pl
import os.path as osp
from typing import Literal
from datetime import datetime

from utils.transforms import Add2DMask, Apply2DMask, AddVirtualNodes
from utils.data_processing import sequentialize_data, sequentialize_timestamps

def read_tal_data(root: str, param: str, seq_len: int, min_nodes_per_graph: int, pre_transform):
    d = joblib.load(root)

    node_id_list = d["id"]

    # Create a mapping from node IDs to integers
    id_mapping = {'robot': 0}   # <-- reserved for robot - not in use for this dataset
    id_mapping.update({f'TAL-{i:02d}': i for i in range(1, 21)})

    # Apply the mapping to the node_id_list
    node_id_mapped = [id_mapping[node_id] for node_id in node_id_list]

    raw_x = d[param]
    pos_x = d["x"]
    pos_y = d["y"]
    timestamps = d["time"]

    x = sequentialize_data(raw_x, seq_len)
    timestamps = sequentialize_timestamps(timestamps, seq_len)
    pos_x = pos_x.repeat(x.shape[0], seq_len, 1) #sequentialize_data(pos_x, seq_len) # pos_x.repeat(x.shape[0], seq_len, 1)
    pos_y = pos_y.repeat(x.shape[0], seq_len, 1)# sequentialize_data(pos_y, seq_len)
    node_id = torch.tensor(node_id_mapped).repeat(x.shape[0], seq_len, 1)

    data_list = []
    for i in (pbar := tqdm(range(x.shape[0]))):
        pbar.set_description(f"Processing {root}")
        sample_x = x[i]
        sample_pos_x = pos_x[i]
        sample_pos_y = pos_y[i]

        # Prepare sensing node data
        snodes_x = sample_x[:,0:]
        snodes_pos_x = sample_pos_x[:,0:][0]
        snodes_pos_y = sample_pos_y[:,0:][0]
        snodes_pos = torch.stack([snodes_pos_x, snodes_pos_y]).float().t()

        snode_ids = node_id[i][0][0:]
        data_timestamps = timestamps[i]

        # Detect sensor that are nan and filter them out
        nan_mask = torch.isnan(sample_x).any(dim=0)
        snodes_x = snodes_x[:,~nan_mask]
        snodes_pos = snodes_pos[~nan_mask,:]
        snode_ids = snode_ids[~nan_mask]

        # Skip this sample, if nodes < min_nodes_per_graph
        if snodes_x.shape[1] < min_nodes_per_graph:
            pass
        # Skip this sample, if time between first and last timestamp is longer than 1 hour
        elif (data_timestamps[-1] - data_timestamps[0]).seconds > 3600:
            print(f"Skipping sample {i} due to time difference > 1 hour")
            pass

        graph = Data(y=snodes_x.t(), id=snode_ids, pos=snodes_pos, orig_pos=snodes_pos, datetime=data_timestamps)
        data_list.append(graph)

    return data_list

# ~~~~~~~~~~~
# DATASET
# ~~~~~~~~~~~

class TALDataset(InMemoryDataset):
    """
    Creates torch_geometric dataset.
    Init args:
        
    """
    def __init__(self, 
                 root: str, 
                 environmental_param: Literal["gas"]="gas", 
                 seq_len: int=10,
                 p_sensor: float=0.7,
                 type: Literal[None, "train", "valid", "test"]=None,
                 transform=None,
                 ):
        self.environmental_param = environmental_param
        self.root = root
        self.seq_len = seq_len
        self.min_nodes_per_graph = 5  
        self.p_sensor = p_sensor 
        self.type = type
        self._load_timestamps()


        self.pre_transform = None

        if transform is None:
            self.transform = T.Compose([
                Add2DMask(seq_len=self.seq_len, percentage=self.p_sensor),
                Apply2DMask(),
                AddVirtualNodes(cell_size=2),
                T.NormalizeScale(),
                T.RadiusGraph(r=0.5, loop=True, max_num_neighbors=300),
                T.Distance(norm=False),
            ])
        else:
            self.transform = transform

        super().__init__(root, transform=self.transform)
        self.load(self.processed_paths[0])
    
    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, "raw")
    
    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, "processed", self.environmental_param)
        
    @property
    def processed_dir(self) -> str:
        if self.type == None:
            return osp.join(self.root, f"processed")
        else:
            return osp.join(self.root, f"processed", self.type)

    @property
    def raw_file_names(self):
        if self.type == None:
            return ["data.pkl"]
        else:
            return [f"{self.type}.pkl"]
    
    @property
    def raw_file_names(self):
        if self.type == None:
            return ["data.pkl"]
        else:
            return [f"{self.type}.pkl"]

    @property
    def processed_file_names(self):
        return ["data.pt"]
    
    def download(self):
        pass
    
    def process(self):
        data_list = read_tal_data(
                        root=osp.join(self.raw_paths[0]), 
                        param=self.environmental_param, 
                        seq_len=self.seq_len,
                        min_nodes_per_graph=self.min_nodes_per_graph,
                        pre_transform=self.pre_transform
                    )

        for data in data_list:
            self.timestamps.append(data.datetime[0])
        
        # Save timestamps list
        with open(osp.join(self.processed_dir, 'timestamps.pkl'), 'wb') as f:
            pickle.dump(self.timestamps, f)

        self.save(data_list, self.processed_paths[0])
    
    def get_by_timestamp(self, query_timestamp: datetime):
        """
        Returns the Data object for the specified timestamp.

        Args:
            query_timestamp (datetime): The datetime object to search for.

        Returns:
            Data: The Data object corresponding to the query_timestamp, or None if not found.
        """
        # Check if the timestamp exists in the dataset
        if query_timestamp in self.timestamps:
            idx = self.timestamps.index(query_timestamp)
            return self.__getitem__(idx)
        else:
            print("Timestamp not found in the dataset.")
            return None
        
    def _load_timestamps(self):
        # Load timestamps list
        timestamps_path = osp.join(self.processed_dir, 'timestamps.pkl')
        if osp.exists(timestamps_path):
            with open(timestamps_path, 'rb') as f:
                self.timestamps = pickle.load(f)
        else:
            self.timestamps = []

# ~~~~~~~~~~~
# DATAMODULE
# ~~~~~~~~~~~
class TALDataModule(pl.LightningDataModule):
    r"""DataModule that loads the TAL campaign dataset.
    """
    def __init__(self, environmental_param: str="gas", seq_len: int=10, batch_size: int=32, num_workers: int=0, shuffle: bool=False, p_sensor: float=0.7, transform=None):
        super().__init__()
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.environmental_param = environmental_param
        self.p_sensor = p_sensor
        self.transform = transform

    def setup(self, stage: str):
        # dataset
        self.train_dataset = TALDataset(root="data/tal", type="train", p_sensor=self.p_sensor, transform=self.transform)     
        self.val_dataset = TALDataset(root="data/tal", type="test", p_sensor=self.p_sensor, transform=self.transform)   
        self.test_dataset = TALDataset(root="data/tal", type="test", p_sensor=self.p_sensor, transform=self.transform)   
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers)
    
    def val_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)