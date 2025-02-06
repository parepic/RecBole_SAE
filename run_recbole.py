# @Time   : 2020/7/20
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn

# UPDATE
# @Time   : 2022/7/8, 2020/10/3, 2020/10/1
# @Author : Zhen Tian, Yupeng Hou, Zihan Lin
# @Email  : chenyuwuxinn@gmail.com, houyupeng@ruc.edu.cn, zhlin@ruc.edu.cn

import matplotlib.pyplot as plt
import numpy as np
import argparse
import torch
from recbole.quick_start import run_recbole, load_data_and_model, run
from recbole.utils import (
    init_logger,
    get_model,
    get_trainer,
    label_popular_items,
    init_seed,
    set_color,
    get_flops,
    get_environment,
    count,
    calculate_pearson_correlation
)


def plot_graphs(ndcgs, hits, coverages, lt_coverages, dampen_percs):
    bar_width = 0.2  # Width of each bar
    # dampen_percs = [x + 1 for x in dampen_percs]
    index = np.arange(len(dampen_percs) + 1)  # X-axis positions for groups, adding 'sasrec'
    
    plt.figure(figsize=(10, 6))
    # Hardcoded first set of bars for 'sasrec'
    
    
    ndcgs = [0.1573] + ndcgs
    hits = [0.2805] + hits
    coverages = [0.6180] + coverages
    lt_coverages = [0.5228] + lt_coverages
    dampen_labels = ['sasrec'] + [f'{dp}' for dp in dampen_percs]
    
    plt.bar(index, ndcgs, bar_width, label='NDCG@10')
    plt.bar(index + bar_width, hits, bar_width, label='Hit@10')
    plt.bar(index + 2 * bar_width, coverages, bar_width, label='Coverage@10')
    plt.bar(index + 3 * bar_width, lt_coverages, bar_width, label='LT Coverage@10')
    
    # Add trend lines starting from the top of each respective bar
    plt.plot(index, ndcgs, marker='o', linestyle='--', color='blue', linewidth=2, label='NDCG@10 Trend')
    plt.plot(index + bar_width, hits, marker='o', linestyle='--', color='orange', linewidth=2, label='Hit@10 Trend')
    plt.plot(index + 2 * bar_width, coverages, marker='o', linestyle='--', color='green', linewidth=2, label='Coverage@10 Trend')
    plt.plot(index + 3 * bar_width, lt_coverages, marker='o', linestyle='--', color='red', linewidth=2, label='LT Coverage@10 Trend')
    
    plt.xlabel('Dampen Percentage')
    plt.ylabel('Values')
    plt.title('Performance Metrics by Dampen Percentage')
    
    plt.xticks(index + 1.5 * bar_width, dampen_labels)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()


import pandas as pd
from IPython.display import display


def calculate_percentage_change(new_values, base_value):
    print(new_values)
    return [f"{new:.4f} ({((new - base_value) / base_value) * 100:.2f}%)" for new in new_values]

def display_metrics_table(dampen_percs, ndcgs, hits, coverages, lt_coverages):
    # Hardcoded first row for 'sasrec'
    base_values = {
        'NDCG@10': 0.1573,
        'Hit@10': 0.2805,
        'Coverage@10': 0.6180,
        'LT Coverage@10': 0.5228
    }
    
    dampen_labels = ['sasrec'] + [f'{dp}' for dp in dampen_percs]
    
    data = {
        'Damped neurons': dampen_labels,
        'NDCG@10': [f"{base_values['NDCG@10']:.4f} (-)" ] + calculate_percentage_change(ndcgs, base_values['NDCG@10']),
        'Hit@10': [f"{base_values['Hit@10']:.4f} (-)" ] + calculate_percentage_change(hits, base_values['Hit@10']),
        'Coverage@10': [f"{base_values['Coverage@10']:.4f} (-)" ] + calculate_percentage_change(coverages, base_values['Coverage@10']),
        'LT Coverage@10': [f"{base_values['LT Coverage@10']:.4f} (-)" ] + calculate_percentage_change(lt_coverages, base_values['LT Coverage@10'])
    }
    df = pd.DataFrame(data)
    
    # Display table
    display(df)

def create_visualizations():
    config, model, dataset, train_data, valid_data, test_data = load_data_and_model(
        model_file=args.path, sae=(args.model=='SASRec_SAE'), device=device
    )  
    ndcgs = []
    hits = []
    coverages = []
    lt_coverages = []
    dampen_percs = []
    dampen_perc = 0
    while dampen_perc <= 1.5:
        test_result = trainer.evaluate(
            test_data, model_file=args.path, show_progress=config["show_progress"], dampen_perc = dampen_perc
        )
        ndcgs.append(test_result['ndcg@10'])
        hits.append(test_result['hit@10'])
        coverages.append(test_result['coverage@10'])
        lt_coverages.append(test_result['LT_coverage@10'])
        dampen_percs.append(dampen_perc)
        print(test_result['ndcg@10'])
        print(test_result['coverage@10'])
        dampen_perc += 0.2
    plot_graphs(ndcgs, hits, coverages, lt_coverages, dampen_percs)
    display_metrics_table(dampen_percs, ndcgs, hits, coverages, lt_coverages)
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", type=str, default="BPR", help="name of models")
    parser.add_argument(
        "--dataset", "-d", type=str, default="ml-1m", help="name of datasets"
    )
    parser.add_argument("--config_files", type=str, default=None, help="config files")
    parser.add_argument(
        "--nproc", type=int, default=1, help="the number of process in this group"
    )
    parser.add_argument(
        "--ip", type=str, default="localhost", help="the ip of master node"
    )
    parser.add_argument(
        "--port", type=str, default="5678", help="the port of master node"
    )
    parser.add_argument(
        "--world_size", type=int, default=-1, help="total number of jobs"
    )
    parser.add_argument(
        "--group_offset",
        type=int,
        default=0,
        help="the global rank offset of this group",
    )
    
    # Add arguments
    parser.add_argument('--path', '-p', type=str, required=False, help="Path to the dataset or configuration file (e.g., 'blablabla').")
    parser.add_argument('--train', action='store_true', help="Flag to indicate whether to train the model.")
    parser.add_argument('--test', action='store_true', help="Flag to indicate whether to test the model.")
    parser.add_argument('--eval_data', action='store_true', help="Flag to indicate whether to test the model.")
    parser.add_argument('--corr_file', '-c', type=str, required=False, help="Name of csv file containing correlation values")
    parser.add_argument('--neuron_count', '-n', type=int, required=False, help="Number of neurons to dampen")
    parser.add_argument('--damp_percent', '-dp', type=float, required=False, help="Damping percentage for popular/unpopular neurons")
    parser.add_argument('--unpopular_only', '-u', action='store_true', help="Flag to indicate whether to dampen only unpopular neurons.")
    parser.add_argument('--save_neurons', '-s', action='store_true', help="Flag to indicate whether to save SAE activations.")

    # Parse the arguments
    args = parser.parse_args()

    args, _ = parser.parse_known_args()
    
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    
    
    if(args.model == "SASRec" and args.train):
        config_file_list = (
                args.config_files.strip().split(" ") if args.config_files else None
            )
        parameter_dict = {
            'train_neg_sample_args': None,
             
            # 'sae_k': 8,
            # 'sae_scale_size': 32,
            # 'sae_lr':1e-3
        }   
        run(
            'SASRec',
            'gowalla',
            config_file_list=config_file_list,
            config_dict=parameter_dict,
            nproc=args.nproc,
            world_size=args.world_size,
            ip=args.ip,
            port=args.port,
            group_offset=args.group_offset,
        )
    else:
        config, model, dataset, train_data, valid_data, test_data = load_data_and_model(
            model_file=args.path, sae=(args.model=='SASRec_SAE'), device=device
        )  
        trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)
        if(args.test):
            # if(args.corr_file):
            #     test_result = trainer.dampen_neurons(
            #         train_data, model_file=args.path, show_progress=config["show_progress"], eval_data=args.eval_data,
            #         corr_file=args.corr_file, neuron_count=args.neuron_count,
            #         damp_percent=args.damp_percent, unpopular_only = args.unpopular_only
            #     )            
            # create_visualizations()
            test_result = trainer.evaluate(
                test_data, model_file=args.path, show_progress=config["show_progress"]
            )
            print(test_result)
            
        elif(args.model == "SASRec_SAE" and args.save_neurons):
            data = test_data if args.eval_data else train_data
            trainer.save_neuron_activations(data,  model_file=args.path, eval_data=args.eval_data)
        elif(args.model == "SASRec_SAE" and args.train):
            trainer.fit_SAE(config, 
                args.path,
                train_data,
                dataset,
                valid_data=valid_data,
                show_progress=True
                )
            