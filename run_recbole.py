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
    calculate_pearson_correlation,
    compute_averages,
    remove_sparse_users_items,
    plot_binned_bar_chart,
    make_items_unpopular,
    save_mean_SD,
    save_cohens_d,
    find_diff
)


def plot_graphs(ndcgs, hits, coverages, lt_coverages, dampen_percs):
    bar_width = 0.2  # Width of each bar
    # dampen_percs = [x + 1 for x in dampen_percs]
    index = np.arange(len(dampen_percs))  # X-axis positions for groups, adding 'sasrec'
    
    plt.figure(figsize=(10, 6))
    # Hardcoded first set of bars for 'sasrec'
    
    dampen_labels =  [f'{dp}' for dp in dampen_percs]
    
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
    return [f"{new:.4f} ({((new - base_value) / base_value) * 100:.2f}%)" for new in new_values]

def display_metrics_table(dampen_percs, ndcgs, hits, coverages, lt_coverages, deep_lt_coverages, ginis, ips_ndcgs, arps, ndcg_heads, ndcg_mids, ndcg_tails):
    # Hardcoded first row for 'sasrec'
    dampen_labels = [f'{dp}' for dp in dampen_percs]
    
    data = {
        'Alpha': dampen_labels,
        'NDCG@10': calculate_percentage_change(ndcgs, ndcgs[0]),
        'NDCG-HEAD@10': calculate_percentage_change(ndcg_heads, ndcg_heads[0]),
        'NDCG-MID@10': calculate_percentage_change(ndcg_mids, ndcg_mids[0]),
        'NDCG-TAIL@10': calculate_percentage_change(ndcg_tails, ndcg_tails[0]),
        # 'Hit@10': calculate_percentage_change(hits, hits[0]),
        # 'Coverage@10': calculate_percentage_change(coverages, coverages[0]),
        'LT Coverage@10': calculate_percentage_change(lt_coverages, lt_coverages[0]),
        'Deep LT Coverage@10': calculate_percentage_change(deep_lt_coverages, deep_lt_coverages[0]),
        # 'Gini coefficient@10': calculate_percentage_change(ginis, ginis[0]),
        'IPS NDCG@10': calculate_percentage_change(ips_ndcgs, ips_ndcgs[0]),
        'ARP@10': calculate_percentage_change(arps, arps[0])
        
    }
    df = pd.DataFrame(data)
    
    # Display table
    print(df.to_string(index=False))  # Print the entire table without truncation

    

def create_visualizations():
    config, model, dataset, train_data, valid_data, test_data = load_data_and_model(
        model_file=args.path, sae=(args.model=='SASRec_SAE'), device=device
    )  
    trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)
    arps = []
    ndcgs = []
    hits = []
    coverages = []
    lt_coverages = []
    deep_lt_coverages = []
    dampen_percs = []
    ginis = []
    ips_ndcgs = []
    ndcg_heads = []
    ndcg_mids = []
    ndcg_tails = []
    dampen_perc = 0
    neuron_count = 0
    for i in range(17):
        test_result = trainer.evaluate(
            valid_data, model_file=args.path, show_progress=config["show_progress"], dampen_perc = dampen_perc
        )
        ndcgs.append(test_result['ndcg@10'])
        ndcg_heads.append(test_result['ndcg-head@10'])
        ndcg_mids.append(test_result['ndcg-mid@10'])
        ndcg_tails.append(test_result['ndcg-tail@10'])
        hits.append(test_result['hit@10'])
        coverages.append(test_result['coverage@10'])
        lt_coverages.append(test_result['LT_coverage@10'])
        deep_lt_coverages.append(test_result['Deep_LT_coverage@10'])
        ginis.append(test_result['Gini_coef@10'])
        ips_ndcgs.append(test_result['ips_ndcg@10'])
        dampen_percs.append(dampen_perc)
        arps.append(test_result['ARP@10'])
        print(test_result['ndcg@10'])
        print(test_result['ips_ndcg@10'])
        print(test_result['coverage@10'])
        dampen_perc += 0.1
    display_metrics_table(dampen_percs, ndcgs, hits, coverages, lt_coverages, deep_lt_coverages, ginis,
                          ips_ndcgs, arps, ndcg_heads, ndcg_mids, ndcg_tails)




def tune_hyperparam():
    config, model, dataset, train_data, valid_data, test_data = load_data_and_model(
        model_file=args.path, sae=(args.model=='SASRec_SAE'), device=device
    )  
    trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)
    Ns = np.linspace(0, 32, 17).tolist()
    betas = np.linspace(-1.5, 1.5, 7).tolist()
    gammas = np.linspace(1, 3, 5).tolist()
    baseline_ndcg = -1
    baseline_arp = -1
    best_triplet = []
    best_metric = []
    best_diff = 9999
    it_num = 0
    for n in Ns:
        if it_num == 0:
            test_result = trainer.evaluate(
                valid_data, model_file=args.path, show_progress=config["show_progress"]
            )
            baseline_ndcg = test_result['ndcg@10']
            baseline_arp = test_result['ARP@10']
            it_num += 1
            continue
        for beta in betas:
            for gamma in gammas:
                test_result = trainer.evaluate(
                    valid_data, model_file=args.path, show_progress=config["show_progress"], N=n, beta=beta, gamma=gamma
                )
                perc_change_ndcg = (test_result['ndcg@10'] - baseline_ndcg) / baseline_ndcg
                perc_change_arp = (test_result['ARP@10'] - baseline_arp) / baseline_arp
                if perc_change_ndcg >= -0.15:
                    if perc_change_arp<= -0.1:
                        if(perc_change_arp - perc_change_ndcg < best_diff):
                            best_diff = perc_change_arp - perc_change_ndcg
                            best_triplet = [n, beta, gamma]
                            best_metric = [test_result['ndcg@10'], test_result['ARP@10']]
                print(f"Iteration number: {it_num} N: {n} Beta: {beta} Gamma: {gamma} ")
                print(f"Current Ndcg: {test_result['ndcg@10']} Current Arp {test_result['ARP@10']} " )
                if len(best_metric) > 0:
                    print(f"Best metric so far Ndcg: {best_metric[0]} Arp {best_metric[1]} " )

                it_num +=1
    print(f"Best ever triplet: {best_triplet}, with results {best_metric}")
    for best in best_triplet:
        print("blyat ", best)
    return best_triplet, best_metric
    

def create_visualizations_neurons():
    config, model, dataset, train_data, valid_data, test_data = load_data_and_model(
        model_file=args.path, sae=(args.model=='SASRec_SAE'), device=device
    )  
    trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)
    arps = []
    ndcgs = []
    hits = []
    coverages = []
    lt_coverages = []
    deep_lt_coverages = []
    dampen_percs = []
    ginis = []
    ips_ndcgs = []
    ndcg_heads = []
    ndcg_mids = []
    ndcg_tails = []
    dampen_perc = 0
    neuron_count = 0
    for i in range(17):
        test_result = trainer.evaluate(
            valid_data, model_file=args.path, show_progress=config["show_progress"], dampen_perc = neuron_count
        )
        ndcgs.append(test_result['ndcg@10'])
        ndcg_heads.append(test_result['ndcg-head@10'])
        ndcg_mids.append(test_result['ndcg-mid@10'])
        ndcg_tails.append(test_result['ndcg-tail@10'])
        hits.append(test_result['hit@10'])
        coverages.append(test_result['coverage@10'])
        lt_coverages.append(test_result['LT_coverage@10'])
        deep_lt_coverages.append(test_result['Deep_LT_coverage@10'])
        ginis.append(test_result['Gini_coef@10'])
        ips_ndcgs.append(test_result['ips_ndcg@10'])
        dampen_percs.append(neuron_count)
        arps.append(test_result['ARP@10'])
        print(test_result['ndcg@10'])
        print(test_result['ips_ndcg@10'])
        print(test_result['coverage@10'])
        neuron_count += 2
    display_metrics_table(dampen_percs, ndcgs, hits, coverages, lt_coverages, deep_lt_coverages, ginis,
                          ips_ndcgs, arps, ndcg_heads, ndcg_mids, ndcg_tails)


from scipy.stats import pearsonr
import h5py
from itertools import combinations
from multiprocessing import Pool, cpu_count



# file_path = r"./dataset/ml-1m/neuron_activations_popular_sasrec.h5"
# dataset_name = "dataset"  # Replace with actual dataset key inside the HDF5
# output_csv = "correlation_pairs_popular.csv"
# num_workers = 6  # You said 6 cores!
# with h5py.File(file_path, "r") as f:
#     data = f[dataset_name][...]  # shape: (64, X)

# assert data.shape[0] == 64, "Expected 64 rows (neurons)."

# === PREPARE PAIRS (i < j only) ===
# row_pairs = list(combinations(range(64), 2))  # Only unique pairs

def compute_corr(pair):
    i, j = pair
    r, _ = pearsonr(data[i], data[j])
    return (i, j, r)



if __name__ == "__main__":
   
    # with Pool(num_workers) as pool:
    #     results = pool.map(compute_corr, row_pairs)

    # # Save to CSV
    # df = pd.DataFrame(results, columns=["row_i", "row_j", "correlation"])
    # df["abs_corr"] = df["correlation"].abs()
    # # df.sort_values("abs_corr", ascending=False, inplace=True)
    # df.drop(columns="abs_corr", inplace=True)
    # df.to_csv(output_csv, index=False)

    # print(f"Saved correlation results to '{output_csv}' using {num_workers} cores.")
    
    
    
    # save_cohens_d()
    # exit()
    # remove_sparse_users_items()
    # label_popular_items()
    # label_popular_items()
    # plot_binned_bar_chart('./dataset/ml-1m/correlations_pop.csv')
    # save_mean_SD()
    # exit()
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
            'ml-1m',
            config_file_list=config_file_list,
            config_dict=parameter_dict,
            nproc=args.nproc,
            world_size=args.world_size,
            ip=args.ip,
            port=args.port,
            group_offset=args.group_offset,
        )
    else:
        # config, model, dataset, train_data, valid_data, test_data = load_data_and_model(
        #     model_file=args.path, sae=(args.model=='SASRec_SAE'), device=device
        # )  
        
        # trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)
        # trainer.fit_gate( 
        #     train_data,
        #     valid_data=test_data,
        #     show_progress=True,
        #     device=config["device"],
        #     path=args.path
        #     )
        
        # exit()
        if(args.test):
            # if(args.corr_file):
            #     test_result = trainer.dampen_neurons(
            #         train_data, model_file=args.path, show_progress=config["show_progress"], eval_data=args.eval_data,
            #         corr_file=args.corr_file, neuron_count=args.neuron_count,
            #         damp_percent=args.damp_percent, unpopular_only = args.unpopular_only
            #     )            
            tune_hyperparam() 
            # test_result = trainer.evaluate(
            #     test_data, model_file=args.path, show_progress=config["show_progress"], dampen_perc=1
            # )
            # print(test_result)
        elif(args.model == "SASRec_SAE" and args.save_neurons):
            data = test_data if args.eval_data else train_data
            trainer.save_neuron_activations2(data,  model_file=args.path, eval_data=args.eval_data, sae=True)
        elif(args.model == "SASRec" and args.save_neurons):
            data = test_data if args.eval_data else train_data
            trainer.save_neuron_activations2(data,  model_file=args.path, eval_data=args.eval_data, sae=False)
        elif(args.model == "SASRec_SAE" and args.train):
            trainer.fit_SAE(config, 
                args.path,
                train_data,
                dataset,
                valid_data=valid_data,
                show_progress=True
                )