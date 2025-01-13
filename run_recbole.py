# @Time   : 2020/7/20
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn

# UPDATE
# @Time   : 2022/7/8, 2020/10/3, 2020/10/1
# @Author : Zhen Tian, Yupeng Hou, Zihan Lin
# @Email  : chenyuwuxinn@gmail.com, houyupeng@ruc.edu.cn, zhlin@ruc.edu.cn

import argparse
from recbole.quick_start import run_recbole, load_data_and_model, run
from recbole.utils import (
    init_logger,
    get_model,
    get_trainer,
    init_seed,
    set_color,
    get_flops,
    get_environment,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", type=str, default="BPR", help="name of models")
    parser.add_argument(
        "--dataset", "-d", type=str, default="ml-100k", help="name of datasets"
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

    args, _ = parser.parse_known_args()

    config_file_list = (
        args.config_files.strip().split(" ") if args.config_files else None
    )

    parameter_dict = {
        'train_neg_sample_args': None,
    }   
    
    # run(
    #     args.model,
    #     args.dataset,
    #     config_file_list=config_file_list,
    #     config_dict=parameter_dict,
    #     nproc=args.nproc,
    #     world_size=args.world_size,
    #     ip=args.ip,
    #     port=args.port,
    #     group_offset=args.group_offset,
    # )
    
    
    # config, model, dataset, train_data, valid_data, test_data = load_data_and_model(
    #     model_file='./saved/SASRec-Jan-12-2025_16-43-54.pth',
    # )  # Here you can replace it by your model path.

    
    config, model, dataset, train_data, valid_data, test_data = load_data_and_model(
        model_file='./saved/SASRec-Jan-13-2025_20-40-37.pth',
    )  # Here you can replace it by your model path.

    trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)


    # trainer.fit_SAE(config, 
    #                 './saved/SASRec-Jan-12-2025_16-43-54.pth',
    #                 train_data,
    #                 dataset,
    #                 valid_data=valid_data,
    #                 show_progress=True
    #                 )
                    
    test_result = trainer.evaluate(
        test_data, model_file='./saved/SASRec-Jan-13-2025_20-40-37.pth', show_progress=config["show_progress"], SAE = True, config=config, dataset=dataset 
    )

    print(test_result)