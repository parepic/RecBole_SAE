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
    find_diff, 
    rank_neurons_by_chi2,
    plot_h5_columns,
    compute_and_save_correlations,
    create_unbiased_set,
    create_item_popularity_csv,
    search_movies,
    process_and_save_movies,
    sample_users_interactions,
    plot_interaction_distribution
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
    result = []
    for new in new_values:
        if base_value == 0:
            change = "inf" if new != 0 else "0.00%"
        else:
            change = f"{((new - base_value) / base_value) * 100:.2f}%"
        result.append(f"{new:.4f} ({change})")
    return result



def display_metrics_table(dampen_percs, ndcgs, hits, coverages, lt_coverages, deep_lt_coverages, ginis, arps, ndcg_heads, ndcg_mids, ndcg_tails):
    # Hardcoded first row for 'sasrec'
    dampen_labels = [f'{dp}' for dp in dampen_percs]
    
    data = {
        'gamma': dampen_labels,
        'NDCG@10': calculate_percentage_change(ndcgs, ndcgs[0]),
        'NDCG-HEAD@10': calculate_percentage_change(ndcg_heads, ndcg_heads[0]),
        'NDCG-MID@10': calculate_percentage_change(ndcg_mids, ndcg_mids[0]),
        'NDCG-TAIL@10': calculate_percentage_change(ndcg_tails, ndcg_tails[0]),
        # 'Hit@10': calculate_percentage_change(hits, hits[0]),
        # 'Coverage@10': calculate_percentage_change(coverages, coverages[0]),
        # 'LT Coverage@10': calculate_percentage_change(lt_coverages, lt_coverages[0]),
        'Deep LT Coverage@10': calculate_percentage_change(deep_lt_coverages, deep_lt_coverages[0]),
        'Gini coefficient@10': calculate_percentage_change(ginis, ginis[0]),
        # 'IPS NDCG@10': calculate_percentage_change(ips_ndcgs, ips_ndcgs[0]),
        'ARP@10': calculate_percentage_change(arps, arps[0])
        
    }
    df = pd.DataFrame(data)
    
    # Display table
    print(df.to_string(index=False))  # Print the entire table without truncation

    

def tune_hyperparam():
    config, model, dataset, train_data, valid_data, test_data = load_data_and_model(
        model_file=args.path, sae=(args.model=='SASRec_SAE'), device=device
    )  
    trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)
    Ns = np.linspace(2, 64, 33)
    betas = [[0.0, 1.0], [0.5, 1.0], [0.0, 0.5], [0.5, 1.5], [0, 1.5], [0.5, 2.0], [1.0, 2.0], [1.0, 2.0], [1.5, 2.0]]
    gammas = [[0.0, 1.0], [0.5, 1.0], [0.0, 0.5], [0.5, 1.5], [0, 1.5], [0.5, 2.0], [1.0, 2.0], [1.0, 2.0], [1.5, 2.0]]
    
    best_triplet = []
    best_ndcg = -1
    it_num = 0
    baseline_ndcg = -1
    for n in Ns:
        if it_num == 0:
            test_result = trainer.evaluate(
                valid_data, model_file=args.path, show_progress=config["show_progress"]
            )
            baseline_ndcg = test_result['ndcg@10']
            it_num += 1
            continue
        for beta in betas:
            for gamma in gammas:
                test_result = trainer.evaluate(
                    valid_data, model_file=args.path, show_progress=config["show_progress"], N=n, beta=beta, gamma=gamma
                )
                if test_result['ndcg@10'] >= best_ndcg:
                    best_triplet = [n, beta, gamma]
                    best_ndcg = test_result['ndcg@10']
                print(f"Iteration number: {it_num} N: {n} Beta: {beta}, Gamma: {gamma} ")
                print(f"Current ndcg {test_result['ndcg@10']}, best ndcg:  {best_ndcg}, best triplet: {best_triplet}" )
                it_num +=1
    print(f"Best ever triplet: {best_triplet}, with results {best_ndcg}, baseline was {baseline_ndcg}")
    for best in best_triplet:
        print("blyat ", best)
    return best_triplet, best_ndcg
    



def create_visualizations_neurons():
    config, model, dataset, train_data, valid_data, test_data = load_data_and_model(
        model_file=args.path, sae=(args.model=='SASRec_SAE'), device=device
    )  
    
    trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)
    arps = [0.00038130350295296784]
    ndcgs = [0.6105]
    hits = [0.6667]
    coverages = [0.8194957983193277]
    lt_coverages = [0.8169110126150699]
    deep_lt_coverages = [0.7359234234234234]
    dampen_percs = [0.0]  # still no value provided
    ginis = [0.6642737940991374]
    ndcg_heads = [0.6935]
    ndcg_mids = [0.5673]
    ndcg_tails = [0.6359]
    neuron_count = 0
    count = 0
    # tochange = np.linspace(0, 4096, 17).tolist()
    tochange = np.linspace(-5, 5, 1)
    # tochange = [[0.0, 1.0],  [0.0, 0.25], [0.5, 1.0], [0.0, 0.5], [0.5, 1.5], [0, 1.5], [0.5, 2.0], [1.0, 2.0], [1.0, 2.0], [1.5, 2.0]]
    toc = [[0.0, 1.0], [0.5, 1.0], [0.0, 0.5], [0.5, 1.5], [0, 1.5], [0.5, 2.0], [1.0, 2.0], [1.0, 2.0], [1.5, 2.0], [1.5, 2.5]]
    
    # tochange = np.linspace(0, 64, 17).tolist()
    for change in tochange:
        if count==0:
            test_result = trainer.evaluate(
                valid_data, model_file=args.path, show_progress=config["show_progress"]
            )     
            print(test_result) 
        else:
            test_result = trainer.evaluate(
                valid_data, model_file=args.path, show_progress=config["show_progress"], N=4096, beta=-4, gamma=4
            )
        count += 1
        ndcgs.append(test_result['ndcg@10'])
        ndcg_heads.append(test_result['ndcg-head@10'])
        ndcg_mids.append(test_result['ndcg-mid@10'])
        ndcg_tails.append(test_result['ndcg-tail@10'])
        hits.append(test_result['hit@10'])
        coverages.append(test_result['coverage@10'])
        lt_coverages.append(test_result['LT_coverage@10'])
        deep_lt_coverages.append(test_result['Deep_LT_coverage@10'])
        ginis.append(test_result['Gini_coef@10'])
        dampen_percs.append(change)
        arps.append(test_result['ARP@10'])
        print(test_result['ndcg@10'])
        print(test_result['Deep_LT_coverage@10'])
        print(test_result['Gini_coef@10'])
        
        neuron_count += 2
    display_metrics_table(dampen_percs, ndcgs, hits, coverages, lt_coverages, deep_lt_coverages, ginis,
                         arps, ndcg_heads, ndcg_mids, ndcg_tails)


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
   
    # exit()
    # create_item_popularity_csv(0.2)
    # exit()
    # remove_sparse_users_items()
    # remove_sparse_users_items(10)
    # exit()
    # sample_users_interactions(15000)
    # exit()
    # save_cohens_d()
    # exit()
    # with open(r"./dataset/Amazon_Electronics/Amazon_Electronics.inter", 'r', encoding='utf-8') as f:
    #     for i, line in enumerate(f):
    #         if i >= 10000000:
    #             break
    #         # Strip newline and split
    #         parts = line.rstrip('\n').split('\t')
    #         if len(parts) != 4:
    #             raise ValueError(
    #                 f"Line {i+1} in hu has {len(parts)} columns (expected {4}): {parts!r}"
    #             )
    #     print(
    #         f"Validation passed: first {min(1000, i+1)} lines ofgg each have {4} columns."
    #     )


    # exit()
    # df = pd.read_csv(r"./dataset/yelp2018/yelp2018.inter", sep='\t', nrows=5)
    # print(df.columns.tolist())
    # sample_users_interactions(20000)
    # exit()

    # df = pd.read_csv(r"./dataset/mind_small_train/mind_small_train.inter", sep='\t')

    # # 2. Extract unique item IDs
    # unique_items = df['item_id:token'].drop_duplicates()

    # # 3. Write them out to the new file with the correct header
    # unique_items.to_frame().to_csv(
    #     r"./dataset/mind_small_train/mind_small_train.item",     # output filename
    #     sep='\t',                     # same separator
    #     index=False,                  # no index column
    #     header=['item_id:token']      # ensure the column name is exactly this
    # )

    # exit()
    # plot_interaction_distribution(r"./dataset/lastfm/item_popularity_labels_with_titles.csv")
    # exit()
    # save_cohens_d()
    # exit()
    # create_item_popularity_csv(0.2)
    # exit()
    # save_cohens_d()
    # exit()
    # save_cohens_d()
    # exit()
    # remove_sparse_users_items(5)
    # exit()
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
        }   

        # config_file_list = [r'./recbole/recbole/properties/overall.yaml',
        #             r'./recbole/recbole/properties/model/SASRec.yaml',
        #             r'./recbole/recbole/properties/dataset/ml-1m.yaml'
        #             ]
        
        run(
            'SASRec',
            'lastfm',
            # config_file_list=config_file_list,
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
        # trainer.save_neuron_activations3(model_file=args.path)
        # exit()
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
            # tune_hyperparam()
            create_visualizations_neurons()
            # create_visualizations_neurons()
            # test_result = trainer.evaluate(
            #     valid_data, model_file=args.path, show_progress=config["show_progress"]
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
                show_progress=True,
                sasrec_sae_file=r"./recbole/saved/SASRec_SAE-Apr-26-k32-64-lastfm.pth"
                )