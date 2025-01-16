import torch
# from tqdm import tqdm
# from torch_geometric.data import Data


def sequentialize_data(x, seq_len):
    r""" Splits the input data into torch.tensor of torch.Size([<number of sampels>, seq_len, 8])
    Init args:
        x (torch.tensor): The data of the synthetic dataset of torch.Size([<total readings>, 8])
        seq_len (int): The length of the sequences.
    """
    nth_frame = 1
    sliding_step = 1
    
    # Select only the images that we care about (ruled by nth frame)
    idxs = torch.tensor(range(x.shape[1])[::nth_frame])
    x = torch.index_select(x, 1, idxs)
    x = x.unfold(dimension=1, size=seq_len, step=sliding_step)
    x = torch.permute(x,(1,2,0))
    
    return x

def sequentialize_timestamps(datetime_str_list, seq_len):
    """ Splits the input datetime data into sequences.
    
    Args:
        datetime_str_list (list): List of datetime strings.
        seq_len (int): The length of the sequences.
    """
    # Initialize the list to hold the sequences
    sequences = []

    # Iterate over the datetime list to create subsequences
    for i in range(len(datetime_str_list) - seq_len + 1):
        # Extract the subsequence
        subsequence = datetime_str_list[i:i + seq_len]
        sequences.append(subsequence)

    return sequences