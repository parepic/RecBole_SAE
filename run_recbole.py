# @Time   : 2020/7/20
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn

# UPDATE
# @Time   : 2022/7/8, 2020/10/3, 2020/10/1
# @Author : Zhen Tian, Yupeng Hou, Zihan Lin
# @Email  : chenyuwuxinn@gmail.com, houyupeng@ruc.edu.cn, zhlin@ruc.edu.cn

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
    count
)



if __name__ == "__main__":
    label_popular_items()
    exit()
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
            'train_neg_sample_args': None
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
        config, model, dataset, train_data, valid_data, test_data = load_data_and_model(
            model_file=args.path, sae=(args.model=='SASRec_SAE'), device=device
        )  
        trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)
        if(args.test):
            test_result = trainer.evaluate(
            test_data, model_file=args.path, show_progress=config["show_progress"]
            )
            print(test_result)
        elif(args.model == "SASRec_SAE" and args.save_neurons):
            data = test_data if args.eval_data else train_data
            trainer.save_neuron_activations(valid_data,  model_file=args.path, eval_data=args.eval_data)
        elif(args.model == "SASRec_SAE" and args.train):
            trainer.fit_SAE(config, 
                args.path,
                train_data,
                dataset,
                valid_data=valid_data,
                show_progress=True
                )


