import torch
from torch_geometric.transforms import BaseTransform, KNNGraph, VirtualNode
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected, subgraph
import random

class AddMask:
    r"""Add a mask to mask random nodes. True denotes known nodes.""" 
    def __init__(self, masked_percentage):
        self.masked_percentage = masked_percentage
        
    def __call__(self, data):
        num_nodes = data.num_nodes
        num_masked_nodes = int(num_nodes * self.masked_percentage)
        mask = torch.zeros(num_nodes, dtype=torch.bool)
        masked_indices = torch.randperm(num_nodes)[:num_masked_nodes]
        mask[masked_indices] = True
        data.known = mask
        return data

class Add2DMask:
    r"""Add a mask to mask random nodes. True denotes known nodes.
    Mask is two dimensional, meaning that the temporal dimension is also taken into account.""" 
    def __init__(self, seq_len, percentage=None):
        self.seq_len = seq_len
        self.percentage = percentage

    def __call__(self, data):
        num_nodes = data.num_nodes
        if self.percentage == None:
            masked_percentage = random.uniform(0.04, 0.15)
        else:
            masked_percentage = self.percentage
        num_masked_nodes = int(num_nodes * masked_percentage)
        mask = torch.zeros(num_nodes, dtype=torch.bool)
        masked_indices = torch.randperm(num_nodes)[:num_masked_nodes]
        mask[masked_indices] = True
        data.known = mask.repeat(self.seq_len,1).permute(1,0)
        return data    
        
class AddGlobalNode(BaseTransform):
    def __init__(self):
        self.global_node = VirtualNode()

    def __call__(self, data):
        # add global node
        data = self.global_node(data)

        global_flag = torch.zeros((data.num_nodes, 1), device=data.x.device, dtype=torch.bool)
        global_flag[-1] = True
        # data.x = torch.cat([data.x, global_flag], dim=1)
        data.global_flag = global_flag

        # remove global node from ground truth
        if hasattr(data, 'ground_truth'):
            data.ground_truth = data.ground_truth[:-1,:]

        return data

class AddVirtualNodes(BaseTransform):
    def __init__(self, cell_size):
        """
        Transform to add virtual nodes based on a grid with cell size.
        Positions can be optionally normalized.
        """
        self.cell_size = cell_size

    def __call__(self, data: Data) -> Data:
        # Original nodes of the subgraph
        x = data.x                  # (num_sensor, feature_dim)
        y = data.y                  # (num_sensor, target_dim)
        pos = data.pos              # (num_sensor, 2) - normalized coordinates
        orig_pos = data.orig_pos    # (num_sensor, 2) - real-world meter coordinates
        known = data.known          # (num_sensor, feature_dim) or (num_sensor, 1)
        sensor = torch.ones(data.num_nodes, dtype=torch.bool)
        node_id = data.id if hasattr(data, 'id') else None
        datetime = data.datetime if hasattr(data, 'datetime') else None

        # --- Compute the bounding box ---
        min_xy = orig_pos.min(dim=0).values
        max_xy = orig_pos.max(dim=0).values

        # Create a grid from min to max with steps of cell_size
        xs = torch.arange(min_xy[0], max_xy[0] + self.cell_size, step=self.cell_size, device=data.x.device)
        ys = torch.arange(min_xy[1], max_xy[1] + self.cell_size, step=self.cell_size, device=data.x.device)
        mesh = torch.stack(torch.meshgrid(xs, ys, indexing='xy'), dim=-1).view(-1, 2)  # (M,2)

        # Number of new virtual nodes
        num_virtual = mesh.size(0)

        # --- Features for virtual nodes ---
        # known=0, x=0, y=0
        new_known = torch.zeros(num_virtual, known.size(1), device=data.x.device)
        new_x = torch.zeros(num_virtual, x.size(1), device=data.x.device)
        new_y = torch.zeros(num_virtual, y.size(1), device=data.x.device) if y.numel() > 0 else None
        new_id = -1 * torch.ones(num_virtual, device=data.x.device) if node_id is not None else None

        # orig_pos = mesh (meter coordinates)
        new_orig_pos = mesh
        new_pos = mesh
        
        # --- Combine original and virtual nodes ---
        x_combined = torch.cat([x, new_x], dim=0)
        y_combined = torch.cat([y, new_y], dim=0) if new_y is not None else y
        id_combined = torch.cat([node_id, new_id]) if node_id is not None else None
        pos_combined = torch.cat([pos, new_pos], dim=0)
        orig_pos_combined = torch.cat([orig_pos, new_orig_pos], dim=0)
        known_combined = torch.cat([known, new_known], dim=0)
        sensor_combined = torch.cat([sensor, torch.zeros(num_virtual, dtype=torch.bool, device=data.x.device)], dim=0)

        # --- Create the augmented graph ---
        augmented_data = Data(
            x=x_combined,
            y=y_combined,
            pos=pos_combined,
            orig_pos=orig_pos_combined,
            known=known_combined,
            sensor=sensor_combined,
            datetime=datetime,
        )

        if id_combined is not None:
            augmented_data.id = id_combined

        return augmented_data


class RandomSubgraphTransform(BaseTransform):
    def __init__(self, num_nodes: int):
        self.num_nodes = num_nodes

    def __call__(self, data):
        if self.num_nodes < 750:
            # Randomly select node indices
            node_idx = torch.randperm(data.num_nodes)[:self.num_nodes]

            # Select the corresponding node features
            y_sub = data.y[node_idx]
            pos_sub = data.pos[node_idx]
            orig_pos_sub = data.orig_pos[node_idx]

            edge_index, edge_attr = subgraph(node_idx, data.edge_index, data.edge_attr, relabel_nodes=True)

            # Create the new data object
            sub_data = Data(
                y=y_sub,
                pos=pos_sub,
                edge_index=edge_index,
                edge_attr=edge_attr,
                orig_pos=orig_pos_sub,
                ground_truth=getattr(data, 'ground_truth', None),
            )

            return sub_data
        else:
            return data

    def __repr__(self):
        return f"{self.__class__.__name__}(num_nodes={self.num_nodes})"

class AddGridMask:
    r"""Add a mask to mask a regular grid of nodes. True denotes known nodes. 
    Args:
        n (int): The number of cells between a 'sensor' in x and y directions.
    """ 
    def __init__(self, n: int=5):
        n = n
        cell_size = 1
        x_min = 0
        y_min = 0

        # Create an empty image-like representation
        grid = torch.zeros((30, 25))

        # Check the cells, where grid sampling position is located
        for row in range(int(n/2), grid.shape[0], n): 
            for col in range(int(n/2), grid.shape[1], n):
                grid[row, col] = 1

        # Find positions of sensor grid
        marked_cells = torch.nonzero(grid)
        pos_x_grid = marked_cells[:,0] * cell_size + x_min
        pos_y_grid = marked_cells[:,1] * cell_size + y_min
        pos_grid = torch.stack([pos_x_grid, pos_y_grid], dim=1)
        self.sensor_positions = pos_grid

    def __call__(self, data):
        # Find matching positions of sensor grid in array from data graph 
        equality_matrix = (data.orig_pos[:, None] == self.sensor_positions).all(dim=2)
        matching_mask = equality_matrix.any(dim=1)
        data.ground_truth = data.y
        #data.x = data.x * matching_mask.view(-1,1)
        data.known = (torch.ones(750) * matching_mask).bool()
        data.known = data.known.repeat(data.y.shape[1],1).permute(1,0)

        return data

class ApplyMask(BaseTransform):
    r"""Apply mask to simulate sensor positions."""    
    def __call__(self, data: Data) -> Data:
        data.x = data.y * data.known.unsqueeze(1)
        return data
    
class Apply2DMask(BaseTransform):
    r"""Apply mask to simulate sensor positions."""    
    def __call__(self, data: Data) -> Data:
        data.x = data.y * data.known
        return data  

class SampleSubgraph(BaseTransform):
    r"""Creates a subgraph.
    Args:
        n (int): The number of nodes in the subgraph.    
    """
    def __init__(self, n):
        self.n = n
    
    def __call__(self, data: Data) -> Data:
        # randomly select nodes from the original graph
        selected_mask = random.sample(range(750), self.n)

        # extract the corresponding nodes and edges from the original graph
        selected_nodes = data.y[selected_mask,:]

        # create a new Data object from the selected nodes and edges
        data = Data(y=selected_nodes, ground_truth=data.y, pos=data.pos[selected_mask], orig_pos=data.orig_pos[selected_mask])
        return data
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(n={self.n})'
    
class ConnectMissingToKnown(BaseTransform):
    r"""Create edge_index by connecting the missing nodes to all known nodes."""
    def __init__(self, undirected=True):
        self.undirected = undirected

    def __call__(self, data: Data) -> Data:
        known_indices = torch.nonzero(data.known, as_tuple=True)[0]
        missing_indices = torch.nonzero(~data.known, as_tuple=True)[0]
        if self.undirected:
            edge_indices = to_undirected(torch.cartesian_prod(missing_indices, known_indices).t())
        else:
            edge_indices = torch.cartesian_prod(known_indices, missing_indices).t()

        data.edge_index = edge_indices
        return data

class KNNtoMissingNodes(BaseTransform):
    r"""Create edge_index by connecting the missing nodes to all known nodes."""
    def __init__(self, k: int=10):
        self.k = k

    def filter_list_of_lists(self, l1, l2):
        """ Filter a list of list and only keep the elements that don't contain any elements from another list."""
        l3 = [[element for element in pair if element not in l2] for pair in l1]
        l3 = [pair for pair in l3 if len(pair)==2]
        return l3

    def create_edges_missing(self, data: Data) -> Data:
        """
        Create missing edges based on a KNNGraph.

        Args:
            data (Tensor): Input data tensor representing the graph.

        Returns:
            Tensor: Tensor representing the missing edges.

        """  
        data_knn = data.clone()
        edges_knn = KNNGraph(k=4, loop=False)(data_knn).edge_index
        edges_knn = torch.transpose(edges_knn, 0, 1)
        idx_known = torch.nonzero(data.known).flatten()
        edges_missing = torch.tensor(self.filter_list_of_lists(edges_knn.tolist(), idx_known.tolist()))
        edges_missing = torch.transpose(edges_missing, 0, 1)
        return edges_missing

    def __call__(self, data: Data) -> Data:
        edge_index_missing = self.create_edges_missing(data)
        edge_index = torch.cat([data.edge_index, edge_index_missing], dim=1)
        data.edge_index = edge_index

        return data