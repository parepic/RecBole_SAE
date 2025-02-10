# -*- coding: utf-8 -*-
# @Time   : 2020/7/17
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn

# UPDATE
# @Time   : 2021/3/8, 2022/7/12, 2023/2/11
# @Author : Jiawei Guan, Lei Wang, Gaowei Zhang
# @Email  : guanjw@ruc.edu.cn, zxcptss@gmail.com, zgw2022101006@ruc.edu.cn

"""
recbole.utils.utils
################################
"""

import h5py
import datetime
import importlib
import os
import random
import pandas as pd

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from texttable import Texttable


from recbole.utils.enum_type import ModelType


def get_local_time():
    r"""Get current time

    Returns:
        str: current time
    """
    cur = datetime.datetime.now()
    cur = cur.strftime("%b-%d-%Y_%H-%M-%S")

    return cur


def ensure_dir(dir_path):
    r"""Make sure the directory exists, if it does not exist, create it

    Args:
        dir_path (str): directory path

    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def get_model(model_name):
    r"""Automatically select model class based on model name

    Args:
        model_name (str): model name

    Returns:
        Recommender: model class
    """
    model_submodule = [
        "general_recommender",
        "context_aware_recommender",
        "sequential_recommender",
        "knowledge_aware_recommender",
        "exlib_recommender",
    ]

    model_file_name = model_name.lower()
    model_module = None
    for submodule in model_submodule:
        module_path = ".".join(["recbole.model", submodule, model_file_name])
        if importlib.util.find_spec(module_path, __name__):
            model_module = importlib.import_module(module_path, __name__)
            break

    if model_module is None:
        raise ValueError(
            "`model_name` [{}] is not the name of an existing model.".format(model_name)
        )
    model_class = getattr(model_module, model_name)
    return model_class


def get_trainer(model_type, model_name):
    r"""Automatically select trainer class based on model type and model name

    Args:
        model_type (ModelType): model type
        model_name (str): model name

    Returns:
        Trainer: trainer class
    """
    try:
        return getattr(
            importlib.import_module("recbole.trainer"), model_name + "Trainer"
        )
    except AttributeError:
        if model_type == ModelType.KNOWLEDGE:
            return getattr(importlib.import_module("recbole.trainer"), "KGTrainer")
        elif model_type == ModelType.TRADITIONAL:
            return getattr(
                importlib.import_module("recbole.trainer"), "TraditionalTrainer"
            )
        else:
            return getattr(importlib.import_module("recbole.trainer"), "Trainer")


def early_stopping(value, best, cur_step, max_step, bigger=True):
    r"""validation-based early stopping

    Args:
        value (float): current result
        best (float): best result
        cur_step (int): the number of consecutive steps that did not exceed the best result
        max_step (int): threshold steps for stopping
        bigger (bool, optional): whether the bigger the better

    Returns:
        tuple:
        - float,
          best result after this step
        - int,
          the number of consecutive steps that did not exceed the best result after this step
        - bool,
          whether to stop
        - bool,
          whether to update
    """
    stop_flag = False
    update_flag = False
    if bigger:
        if value >= best:
            cur_step = 0
            best = value
            update_flag = True
        else:
            cur_step += 1
            if cur_step > max_step:
                stop_flag = True
    else:
        if value <= best:
            cur_step = 0
            best = value
            update_flag = True
        else:
            cur_step += 1
            if cur_step > max_step:
                stop_flag = True
    return best, cur_step, stop_flag, update_flag


def calculate_valid_score(valid_result, valid_metric=None):
    r"""return valid score from valid result

    Args:
        valid_result (dict): valid result
        valid_metric (str, optional): the selected metric in valid result for valid score

    Returns:
        float: valid score
    """
    if valid_metric:
        return valid_result[valid_metric]
    else:
        return valid_result["Recall@10"]
    


def dict2str(result_dict):
    r"""convert result dict to str

    Args:
        result_dict (dict): result dict

    Returns:
        str: result str
    """

    return "    ".join(
        [str(metric) + " : " + str(value) for metric, value in result_dict.items()]
    )


def init_seed(seed, reproducibility):
    r"""init random seed for random functions in numpy, torch, cuda and cudnn

    Args:
        seed (int): random seed
        reproducibility (bool): Whether to require reproducibility
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if reproducibility:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False


def get_tensorboard(logger):
    r"""Creates a SummaryWriter of Tensorboard that can log PyTorch models and metrics into a directory for
    visualization within the TensorBoard UI.
    For the convenience of the user, the naming rule of the SummaryWriter's log_dir is the same as the logger.

    Args:
        logger: its output filename is used to name the SummaryWriter's log_dir.
                If the filename is not available, we will name the log_dir according to the current time.

    Returns:
        SummaryWriter: it will write out events and summaries to the event file.
    """
    base_path = "log_tensorboard"

    dir_name = None
    for handler in logger.handlers:
        if hasattr(handler, "baseFilename"):
            dir_name = os.path.basename(getattr(handler, "baseFilename")).split(".")[0]
            break
    if dir_name is None:
        dir_name = "{}-{}".format("model", get_local_time())

    dir_path = os.path.join(base_path, dir_name)
    writer = SummaryWriter(dir_path)
    return writer


def get_gpu_usage(device=None):
    r"""Return the reserved memory and total memory of given device in a string.
    Args:
        device: cuda.device. It is the device that the model run on.

    Returns:
        str: it contains the info about reserved memory and total memory of given device.
    """

    reserved = torch.cuda.max_memory_reserved(device) / 1024**3
    total = torch.cuda.get_device_properties(device).total_memory / 1024**3

    return "{:.2f} G/{:.2f} G".format(reserved, total)


def get_flops(model, dataset, device, logger, transform, verbose=False):
    r"""Given a model and dataset to the model, compute the per-operator flops
    of the given model.
    Args:
        model: the model to compute flop counts.
        dataset: dataset that are passed to `model` to count flops.
        device: cuda.device. It is the device that the model run on.
        verbose: whether to print information of modules.

    Returns:
        total_ops: the number of flops for each operation.
    """
    if model.type == ModelType.DECISIONTREE:
        return 1
    if model.__class__.__name__ == "Pop":
        return 1

    import copy

    model = copy.deepcopy(model)

    def count_normalization(m, x, y):
        x = x[0]
        flops = torch.DoubleTensor([2 * x.numel()])
        m.total_ops += flops

    def count_embedding(m, x, y):
        x = x[0]
        nelements = x.numel()
        hiddensize = y.shape[-1]
        m.total_ops += nelements * hiddensize

    class TracingAdapter(torch.nn.Module):
        def __init__(self, rec_model):
            super().__init__()
            self.model = rec_model

        def forward(self, interaction):
            return self.model.predict(interaction)

    custom_ops = {
        torch.nn.Embedding: count_embedding,
        torch.nn.LayerNorm: count_normalization,
    }
    wrapper = TracingAdapter(model)
    inter = dataset[torch.tensor([1])].to(device)
    inter = transform(dataset, inter)
    inputs = (inter,)
    from thop.profile import register_hooks
    from thop.vision.basic_hooks import count_parameters

    handler_collection = {}
    fn_handles = []
    params_handles = []
    types_collection = set()
    if custom_ops is None:
        custom_ops = {}

    def add_hooks(m: nn.Module):
        m.register_buffer("total_ops", torch.zeros(1, dtype=torch.float64))
        m.register_buffer("total_params", torch.zeros(1, dtype=torch.float64))

        m_type = type(m)

        fn = None
        if m_type in custom_ops:
            fn = custom_ops[m_type]
            if m_type not in types_collection and verbose:
                logger.info("Customize rule %s() %s." % (fn.__qualname__, m_type))
        elif m_type in register_hooks:
            fn = register_hooks[m_type]
            if m_type not in types_collection and verbose:
                logger.info("Register %s() for %s." % (fn.__qualname__, m_type))
        else:
            if m_type not in types_collection and verbose:
                logger.warning(
                    "[WARN] Cannot find rule for %s. Treat it as zero Macs and zero Params."
                    % m_type
                )

        if fn is not None:
            handle_fn = m.register_forward_hook(fn)
            handle_paras = m.register_forward_hook(count_parameters)
            handler_collection[m] = (
                handle_fn,
                handle_paras,
            )
            fn_handles.append(handle_fn)
            params_handles.append(handle_paras)
        types_collection.add(m_type)

    prev_training_status = wrapper.training

    wrapper.eval()
    wrapper.apply(add_hooks)

    with torch.no_grad():
        wrapper(*inputs)

    def dfs_count(module: nn.Module, prefix="\t"):
        total_ops, total_params = module.total_ops.item(), 0
        ret_dict = {}
        for n, m in module.named_children():
            next_dict = {}
            if m in handler_collection and not isinstance(
                m, (nn.Sequential, nn.ModuleList)
            ):
                m_ops, m_params = m.total_ops.item(), m.total_params.item()
            else:
                m_ops, m_params, next_dict = dfs_count(m, prefix=prefix + "\t")
            ret_dict[n] = (m_ops, m_params, next_dict)
            total_ops += m_ops
            total_params += m_params

        return total_ops, total_params, ret_dict

    total_ops, total_params, ret_dict = dfs_count(wrapper)

    # reset wrapper to original status
    wrapper.train(prev_training_status)
    for m, (op_handler, params_handler) in handler_collection.items():
        m._buffers.pop("total_ops")
        m._buffers.pop("total_params")
    for i in range(len(fn_handles)):
        fn_handles[i].remove()
        params_handles[i].remove()

    return total_ops


def list_to_latex(convert_list, bigger_flag=True, subset_columns=[]):
    result = {}
    for d in convert_list:
        for key, value in d.items():
            if key in result:
                result[key].append(value)
            else:
                result[key] = [value]

    df = pd.DataFrame.from_dict(result, orient="index").T

    if len(subset_columns) == 0:
        tex = df.to_latex(index=False)
        return df, tex

    def bold_func(x, bigger_flag):
        if bigger_flag:
            return np.where(x == np.max(x.to_numpy()), "font-weight:bold", None)
        else:
            return np.where(x == np.min(x.to_numpy()), "font-weight:bold", None)

    style = df.style
    style.apply(bold_func, bigger_flag=bigger_flag, subset=subset_columns)
    style.format(precision=4)

    num_column = len(df.columns)
    column_format = "c" * num_column
    tex = style.hide(axis="index").to_latex(
        caption="Result Table",
        label="Result Table",
        convert_css=True,
        hrules=True,
        column_format=column_format,
    )

    return df, tex


def get_environment(config):
    gpu_usage = (
        get_gpu_usage(config["device"])
        if torch.cuda.is_available() and config["use_gpu"]
        else "0.0 / 0.0"
    )

    import psutil

    memory_used = psutil.Process(os.getpid()).memory_info().rss / 1024**3
    memory_total = psutil.virtual_memory()[0] / 1024**3
    memory_usage = "{:.2f} G/{:.2f} G".format(memory_used, memory_total)
    cpu_usage = "{:.2f} %".format(psutil.cpu_percent(interval=1))
    """environment_data = [
        {"Environment": "CPU", "Usage": cpu_usage,},
        {"Environment": "GPU", "Usage": gpu_usage, },
        {"Environment": "Memory", "Usage": memory_usage, },
    ]"""

    table = Texttable()
    table.set_cols_align(["l", "c"])
    table.set_cols_valign(["m", "m"])
    table.add_rows(
        [
            ["Environment", "Usage"],
            ["CPU", cpu_usage],
            ["GPU", gpu_usage],
            ["Memory", memory_usage],
        ]
    )

    return table


def get_item_titles(tensor, df):
    """
    Given a tensor of any shape, return the titles corresponding to the IDs
    in 'tensor', skipping any elements that are 0. For IDs not found in the 
    DataFrame, return "undefined". The shape of the returned list of lists matches
    the shape of the input tensor.
    
    :param tensor: The tensor (PyTorch, NumPy array, etc.) containing item IDs.
    :param df: The DataFrame containing the 'item_id:token' and 'movie_title:token' columns.
    :return: A nested list of movie titles corresponding to all IDs in 'tensor'.
    """
    # Convert tensor to list of lists to preserve shape
    tensor_as_list = tensor.tolist()

    # Create a dictionary for quick lookup of item titles
    id_to_title = dict(zip(df['item_id:token'], df['movie_title:token']))

    # Replace IDs with movie titles, preserving the shape
    result = []
    for row in tensor_as_list:
        # Use `.get(item_id, "undefined")` to handle missing IDs
        titles_row = [id_to_title.get(item_id, "undefined") for item_id in row if item_id != 0]
        result.append(titles_row)
    
    return result


def label_popular_items():
    # Load the data
    data = pd.read_csv(r'./dataset/steam/interactions_remapped.csv', encoding='latin1')  # Replace with your actual file name
    titles_data = pd.read_csv(r'./dataset/steam/items_remapped.csv', encoding='latin1')  # Replace with your file containing titles and item IDs

    # Calculate interaction counts per item
    item_interactions = data['item_id:token'].value_counts().reset_index()
    item_interactions.columns = ['item_id:token', 'interaction_count']

    # Calculate the thresholds for top 10% and bottom 20%
    top_threshold = item_interactions['interaction_count'].quantile(0.8)
    bottom_threshold = item_interactions['interaction_count'].quantile(0.2)

    # Label items as 'popular' (1), 'unpopular' (-1), or 'neutral' (0)
    def label_popularity(count):
        if count >= top_threshold:
            return 1
        elif count <= bottom_threshold:
            return -1
        else:
            return 0

    item_interactions['popularity_label'] = item_interactions['interaction_count'].apply(label_popularity)

    # Merge with movie titles data to add titles
    output_df = pd.merge(item_interactions, titles_data, how='left', left_on='item_id:token', right_on='item_id:token')

    # Select relevant columns
    output_df = output_df[['item_id:token', 'app_name:token', 'popularity_label', 'interaction_count']]

    # Sort by popularity label and interaction count
    output_df = output_df.sort_values(by=['popularity_label', 'interaction_count'], ascending=[False, False])

    # Save the output to a CSV file
    output_df.to_csv(r"./dataset/steam/item_popularity_labels_with_titles.csv", index=False)

    print("Popularity labels with titles saved to 'item_popularity_labels_with_titles.csv'")

import math
def save_user_popularity_score(alpha, user_ids, sequences):
    sequences = [seq.cpu().numpy().tolist() for seq in sequences]
    user_ids = [id.item() for id in user_ids]
    
    """
    Updates an HDF5 file with new user IDs and sequences.
    
    Args:
        file_path (str): Path to the HDF5 file.
        user_ids (list): List of user IDs (length [N]).
        sequences (list or numpy array): List or array of sequences (shape [N, 50]).
    """
    # Ensure sequences are lists if they are tensors (PyTorch or NumPy)
    
    item_labels = pd.read_csv("./dataset/ml-1m/item_popularity_labels_with_titles.csv")
    # Open the HDF5 file in append mode (create if it doesn't exist)
    with h5py.File('user_popularity_scores.h5', "a") as f:
        for user_id, user_sequences in zip(user_ids, sequences):
            # Ensure the group for the user exists
            if str(user_id) not in f:
                f.create_group(str(user_id))
            user_group = f[str(user_id)]
            filtered_seq = [x for x in user_sequences if x != 0]
            seq_key = str(filtered_seq)  # Convert sequence to a string key for checking
            total_weight = 0
            if seq_key not in user_group:                    
                # Calculate total_score
                total_score = 0
                for idx, item in enumerate(reversed(filtered_seq)):
                    # Get the popularity label for the item
                    popularity_label = item_labels.loc[item_labels['item_id:token'] == item, 'popularity_label']
                    weight = math.pow(alpha, idx)
                    total_weight += weight
                    total_score += int(popularity_label.iloc[0]) * weight
                # Save the sequence and total_score as a dataset and attribute
                dataset_name = f"seq_{len(user_group)}"
                dataset = user_group.create_dataset(dataset_name, data=filtered_seq)
                dataset.attrs["total_score"] = total_score / total_weight



def fetch_user_popularity_score(user_ids, sequences):
    """
    Fetches the total scores for the given user IDs and their sequences from the HDF5 file.

    Args:
        user_ids (list): List of user IDs to fetch.
        sequences (list): List of sequences corresponding to the user IDs.

    Returns:
        list: A list of total scores for the provided sequences.
    """
    # Convert sequences and user_ids from tensor to native Python types.
    sequences = [seq.cpu().numpy() for seq in sequences]
    user_ids = [uid.item() for uid in user_ids]

    total_scores = []
    file_path = r"./dataset/ml-1m/user_popularity_scores.h5"

    with h5py.File(file_path, "r") as f:
        for uid, seq in zip(user_ids, sequences):
            uid_str = str(uid)
            if uid_str not in f:
                print(f"User ID {uid_str} not found in the HDF5 file.")
                continue

            user_group = f[uid_str]
            # Compute the filtered sequence once for the user (removing zeros)
            filtered_seq = np.array([x for x in seq if x != 0])

            # Option 1: Iterate through datasets and compare using numpy
            found = False
            for ds_name in user_group:
                stored_seq = user_group[ds_name][:]
                # First check the shape before doing an element-wise comparison
                if stored_seq.shape == filtered_seq.shape and np.array_equal(stored_seq, filtered_seq):
                    total_score = user_group[ds_name].attrs.get("total_score", None)
                    total_scores.append(total_score)
                    found = True
                    break

            if not found:
                print(f"Sequence {filtered_seq.tolist()} not found for user {uid_str}.")
                
    print(len(total_scores))
    return total_scores

def save_batch_activations(bulk_data):
    """
    Saves a bulk of data (shape 4096 x 4096) to the HDF5 file, appending it to each row.

    Args:
        file_path (str): Path to the HDF5 file.
        bulk_data (numpy.ndarray): A 2D NumPy array of shape (4096, 4096) to append.
    """
    print(bulk_data.shape)
    bulk_data = bulk_data.permute(1, 0)
    file_path = r"./dataset/ml-1m/neuron_activations.h5"
    with h5py.File(file_path, "a") as f:
        if "dataset" not in f:
            # If the dataset doesn't exist, create it with unlimited columns
            max_shape = (4096, 1100000)  # Unlimited columns
            f.create_dataset(
                "dataset",
                data=bulk_data,
                maxshape=max_shape,
                chunks=(4096, 2048),  # Optimize chunk size for appending
                dtype="float32",
            )
        else:
            # Resize the dataset to accommodate the new data
            dataset = f["dataset"]
            current_cols = dataset.shape[1]
            print(current_cols)
            new_cols = current_cols + bulk_data.shape[1]
            dataset.resize((4096, new_cols))
            
            # Write the new data at the end
            dataset[:, current_cols:new_cols] = bulk_data
            


def save_batch_user_popularities(bulk_data):
    """
    Saves a bulk of data to an HDF5 file, appending it to a 1D dataset.

    Args:
        file_path (str): Path to the HDF5 file.
        bulk_data (list or numpy.ndarray): A 1D list or array of values to append.
    """
    bulk_data = np.array(bulk_data, dtype=np.float32)  # Ensure data is in NumPy array format
    file_path = r"./dataset/ml-1m/user_scores.h5"
    with h5py.File(file_path, "a") as f:
        if "dataset" not in f:
            # Create the dataset if it doesn't exist
            max_shape = (1100000,)  # Unlimited length
            f.create_dataset(
                "dataset",
                data=bulk_data,
                maxshape=max_shape,
                chunks=(len(bulk_data),),  # Chunk size is the size of the bulk
                dtype="float32"
            )
        else:

            # Resize and append to the existing dataset
            dataset = f["dataset"]
            current_size = dataset.shape[0]
            new_size = current_size + len(bulk_data)
            dataset.resize((new_size,))
            dataset[current_size:new_size] = bulk_data


def calculate_pearson_correlation():
    """
    Calculates the Pearson correlation of each row in a dataset with another dataset and saves the result to a CSV file.

    Args:
        file1_path (str): Path to the first HDF5 file (shape (N, F)).
        file2_path (str): Path to the second HDF5 file (shape (F,)).
        output_csv_path (str): Path to save the resultant CSV file (shape (N,)).
    """
    
    file1_path = r"./dataset/ml-1m/neuron_activations.h5"
    file2_path = r"./dataset/ml-1m/user_scores.h5"
    output_csv_path = r"./dataset/ml-1m/correlations.csv"
    # Load the data from the HDF5 files
    with h5py.File(file1_path, "r") as f1, h5py.File(file2_path, "r") as f2:
        dataset2 = f2["dataset"][:]  # Shape (F,)
        dataset1 = f1["dataset"][:]  # Shape (F

    # Validate the shapes
    if dataset1.shape[1] != dataset2.shape[0]:
        raise ValueError("The number of features (F) in file1 must match the length of the dataset in file2.")
    # Calculate Pearson correlation for each row in dataset1 with dataset2
    correlations = []
    for row in dataset1:
        correlation = np.corrcoef(row, dataset2)[0, 1]  # Pearson correlation
        correlations.append(correlation)

    # Save the results to a CSV file
    df = pd.DataFrame({"Pearson_Correlation": correlations})
    df.to_csv(output_csv_path, index=False)
    print(f"Pearson correlations saved to {output_csv_path}")
    return correlations


import matplotlib.pyplot as plt

def get_extreme_correlations(file_name: str, n: int, unpopular_only: bool):
    """
    Retrieves the highest and lowest correlation indexes and their values.
    
    Parameters:
    file_name (str): CSV file name containing correlation values.
    n (int): Number of extreme values to retrieve.
    unpopular_only (bool): Whether to return only the lowest values.
    
    Returns:
    list or tuple: If unpopular_only is True, returns a list of lowest indexes and their values.
                   Otherwise, returns a tuple of (highest_indexes, highest_values, lowest_indexes, lowest_values).
    """
    file_path = r"./dataset/ml-1m/" + file_name
    df = pd.read_csv(file_path)
    
    # Assuming the column name is unknown, take the first column
    column_name = df.columns[0]
    values = df[column_name]
    
    # Get indexes and values of highest and lowest n/2 values
    highest = values.nlargest(n)
    lowest = values.nsmallest(n)
    
    highest_indexes = highest.index.tolist()
    highest_values = highest.tolist()
    lowest_indexes = lowest.index.tolist()
    lowest_values = lowest.tolist()
    
    if unpopular_only:
        return list(zip(lowest_indexes, lowest_values))
    
    return (list(zip(highest_indexes, highest_values)), list(zip(lowest_indexes, lowest_values)))




def count():
    # Define file paths
    inter_file_path =  r'./dataset/steam/steam.inter'  # Replace with actual path
    item_file_path = r'./dataset/steam/steam.item'  # Replace with actual path

    # Define output file paths
    filtered_inter_file_path = r'./dataset/steam/steam.inter'
    filtered_item_file_path = r'./dataset/steam/steam.item'

    # Load and filter the .inter file (assuming it's tab-separated; modify sep if needed)
    df_inter = pd.read_csv(inter_file_path, sep="\t", usecols=["user_id:token", "item_id:token", "timestamp:float"])
    df_inter.to_csv(filtered_inter_file_path, sep="\t", index=False)

    # Load and filter the .item file (assuming it's tab-separated; modify sep if needed)
    df_item = pd.read_csv(item_file_path, sep="\t", usecols=["item_id:token", "app_name:token"])
    df_item.to_csv(filtered_item_file_path, sep="\t", index=False)

    print("Filtered .inter and .item files have been saved.")


def compute_averages(output_file="output_averages.csv"): 
    """
    Computes the average values for all 4096 elements in A based on labels in C,
    and appends the results to a CSV file. If the file does not exist, it is created.
    
    Parameters:
    output_file (str): Path to save the resulting CSV file.
    """
    file1_path = r"./dataset/ml-1m/neuron_activations.h5"
    file2_path = r"./dataset/ml-1m/user_scores.h5"
    
    with h5py.File(file1_path, "r") as f1, h5py.File(file2_path, "r") as f2:
        dataset1 = f1["dataset"][:]  # Shape (4096, N)
        dataset2 = f2["dataset"][:]  # Shape (N,)
    
    # Step 1: Create list D based on conditions
    D = np.where(dataset2 < 0.3, -1, np.where(dataset2 > 0.7, 1, 0))
    
    # Step 2: Compute averages for each row in dataset1 based on D values
    valid_neg = D == -1
    valid_pos = D == 1
    
    avg_neg = np.where(np.any(valid_neg, axis=0), np.mean(dataset1[:, valid_neg], axis=1), 0)
    avg_pos = np.where(np.any(valid_pos, axis=0), np.mean(dataset1[:, valid_pos], axis=1), 0)
    diff = avg_pos - avg_neg
    
    # Combine results
    Y = np.column_stack((avg_neg, avg_pos, diff))
    
    # Convert to DataFrame
    df_Y = pd.DataFrame(Y, columns=["Avg (-1)", "Avg (1)", "Difference"])
    
    # Append to CSV file, creating it if it does not exist
    df_Y.to_csv(output_file, mode='a', header=not os.path.exists(output_file), index=False)
    
    print(f"Results appended to {output_file}")


def get_difference_values(indexes, csv_file="output_averages.csv"):
    """
    Given an array of indexes, returns an array with 'Difference' values for the given indexes.
    
    Parameters:
    indexes (np.ndarray): Array of indexes to retrieve values for.
    csv_file (str): Path to the CSV file containing precomputed values.
    
    Returns:
    np.ndarray: Array of 'Difference' values corresponding to the given indexes.
    """
    df = pd.read_csv(csv_file)
    return df.iloc[indexes]["Difference"].values


def remove_sparse_users_items():
    interactions_file = r"./dataset/lfm1b-artists/lfm1b-artists.inter"
    items_file = r"./dataset/lfm1b-artists/lfm1b-artists.item"

    # Load interactions and items data
    df_inter = pd.read_csv(interactions_file, sep='\t')
    df_item = pd.read_csv(items_file, sep='\t')

    # -------------------------------
    # 2. Iterative Filtering
    # -------------------------------
    min_interactions = 420

    # We'll iterate until the number of interactions/users doesn't change.
    prev_interactions_count = -1
    prev_users_count = -1

    while True:
        # --- Filter Items ---
        # Count interactions per item
        item_counts = df_inter['item_id:token'].value_counts()
        # Identify items with at least min_interactions
        valid_items = item_counts[item_counts >= min_interactions].index
        # Filter interactions DataFrame to keep only valid items
        df_inter = df_inter[df_inter['item_id:token'].isin(valid_items)]

        # --- Filter Users ---
        # Count interactions per user
        user_counts = df_inter['user_id:token'].value_counts()
        # Identify users with at least min_interactions
        valid_users = user_counts[user_counts >= min_interactions].index
        # Filter interactions DataFrame to keep only valid users
        df_inter = df_inter[df_inter['user_id:token'].isin(valid_users)]

        # Check for convergence
        current_interactions_count = df_inter.shape[0]
        current_users_count = df_inter['user_id:token'].nunique()

        if (current_interactions_count == prev_interactions_count) and (current_users_count == prev_users_count):
            break  # No further changes
        else:
            prev_interactions_count = current_interactions_count
            prev_users_count = current_users_count

    # -------------------------------
    # 3. Update the Items DataFrame
    # -------------------------------
    # Keep only items that still appear in the interactions DataFrame
    df_item = df_item[df_item['item_id:token'].isin(df_inter['item_id:token'].unique())]

    # -------------------------------
    # 4. (Optional) Save the Filtered Data
    # -------------------------------
    df_inter.to_csv(r'./dataset/lfm1b-artists/lfm1b-artists_new.inter', index=False, sep='\t')
    df_item.to_csv(r'./dataset/lfm1b-artists/lfm1b-artists_new.item', index=False, sep='\t')

    print("Filtering complete:")
    print(f" - Interactions: {df_inter.shape[0]} records")
    print(f" - Users: {df_inter['user_id:token'].nunique()} unique users")
    print(f" - Items: {df_item.shape[0]} records")