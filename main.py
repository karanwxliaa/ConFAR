import os
import sys
import argparse
import torch
import numpy as np
from random import shuffle
from collections import OrderedDict
import agents
from dataloaders.base import load_ConFAR,load_ConFAR_UB #,load_ConFAR_AF2
import warnings
from torch.utils.data import Subset

warnings.filterwarnings("ignore")

import logging


def run(args):
    if not os.path.exists('outputs'):
        os.mkdir('outputs')

    # if args.bound == 'UB':
    #     train_dataset_splits, val_dataset_splits, task_output_space = load_ConFAR_UB(image_size=args.image_size, aug=args.train_aug)
    # else:
    #     if args.dataset == 'single':
    #         train_dataset_splits, val_dataset_splits, task_output_space = load_ConFAR_UB(image_size=args.image_size, aug=args.train_aug)
    #     else:
    #         train_dataset_splits, val_dataset_splits, task_output_space = load_ConFAR(image_size=args.image_size, aug=args.train_aug)

    train_dataset_splits, val_dataset_splits, task_output_space = load_ConFAR_UB(image_size=args.image_size, aug=args.train_aug)

    if args.datasize:
        for task in train_dataset_splits:
            train_dataset_splits[task] = Subset(train_dataset_splits[task], range(min(len(train_dataset_splits[task]), args.datasize)))
            val_dataset_splits[task] = Subset(val_dataset_splits[task], range(min(len(val_dataset_splits[task]), args.datasize)))
    logging.info("------------------------------------------------------------")
    print("------------------------------------------------------------")
    # # Print the shape of a sample from the training datasets
    # for key, dataset in train_dataset_splits.items():
    # 	logging.info(f"Training dataset shape for task {key}: {len(dataset)}")
    # 	print(f"Training dataset shape for task {key}: {len(dataset)}")
    # 	sample_x, sample_y, _ = dataset[0][0]  # Assuming the first sample is representative
    # 	logging.info(f"Training dataset sample shape for task {key} - x: {sample_x.shape}, y: {sample_y.shape}")
    # 	print(f"Training dataset sample shape for task {key} - x: {sample_x.shape}, y: {sample_y.shape}")
    
    # # Print the shape of a sample from the validation datasets
    # for key, dataset in val_dataset_splits.items():
    # 	logging.info(f"Validation dataset shape for task {key}: {len(dataset)}")
    # 	print(f"Validation dataset shape for task {key}: {len(dataset)}")
    # 	sample_x, sample_y, _ = dataset[0][0]  # Assuming the first sample is representative
    # 	logging.info(f"Validation dataset sample shape for task {key} - x: {sample_x.shape}, y: {sample_y.shape}")
    # 	print(f"Validation dataset sample shape for task {key} - x: {sample_x.shape}, y: {sample_y.shape}")
    # logging.info("------------------------------------------------------------")
    # print("------------------------------------------------------------")
    
    # Decide split ordering
    
    task_names = list(task_output_space.keys())[::]  # SORT THIS HERE SUXYHDGBFN
    # task_names = list(task_output_space.keys()) #SORT THIS HERE SUXYHDGBFN
    print('Task order:', task_names)
    logging.info('Task order: %s', task_names)
    
    if 'All' in task_names:
        out_dim = {'All': 15}
    else:
        out_dim = {'FER': args.force_out_dim_fer, 'AU': args.force_out_dim_au, 'AV': args.force_out_dim_av}

    
    # ADD OPTIMIZER HERE
    agent_config = {'tasks': task_names, 'lr': args.lr, 'momentum': args.momentum, 'weight_decay': args.weight_decay, 'schedule': args.schedule,
                    'model_type': args.model_type, 'model_name': args.model_name, 'model_weights': args.model_weights,
                    'out_dim': out_dim,
                    'optimizer': args.optimizer,
                    'print_freq': args.print_freq, 'gpuid': args.gpuid,
                    'reg_coef': reg_coef,
                    'img_size':args.image_size}
    #breakpoint()
    agent = agents.__dict__[args.agent_type].__dict__[args.agent_name](agent_config)

    print(agent.model)
    logging.info(agent.model)

    # print('# of parameters:',agent.count_parameter())
    acc_table = OrderedDict()
    f1_table = OrderedDict()
    
    for i in range(len(task_names)):
        
        logging.info(len(train_dataset_splits[task_names[i]]))
        print(len(train_dataset_splits[task_names[i]]))
        
        # adjusted_train_dataset = CustomDataset(train_dataset_splits[train_name], train_name)
        # adjusted_val_dataset = CustomDataset(val_dataset_splits[train_name], train_name)

        train_name = task_names[i]

        # for task_name, dataset in train_dataset_splits.items():
        # 	dataset.set_transform(lambda label: custom_target_transform(task_name, task_output_space, train_name, label))
        logging.info('====================== %s =======================',train_name)
        print('======================', train_name, '=======================')

        if args.bound == 'UB':
            for j in range(i+1):
                
                current_task = task_names[j]

                print("\nTraining ",current_task,"\n")

                train_loader = torch.utils.data.DataLoader(train_dataset_splits[current_task],
                                                            batch_size=args.batch_size, shuffle=True, num_workers=0)
                val_loader = torch.utils.data.DataLoader(val_dataset_splits[current_task],
                                                            batch_size=args.batch_size, shuffle=False, num_workers=0)
                if j == i:
                    agent.add_valid_output_dim(task_output_space[current_task])

                agent.learn_batch(train_loader, val_loader, j)
        else:
            train_loader = torch.utils.data.DataLoader(train_dataset_splits[train_name],
                                                            batch_size=args.batch_size, shuffle=True, num_workers=0)#,collate_fn=custom_collate_fn)
            val_loader = torch.utils.data.DataLoader(val_dataset_splits[train_name],
                                                        batch_size=args.batch_size, shuffle=False, num_workers=0)#,collate_fn=custom_collate_fn)
            
            agent.add_valid_output_dim(task_output_space[train_name])
            #breakpoint()
            agent.learn_batch(train_loader, val_loader, train_name)

            
        # Evaluate
        acc_table[train_name] = {}  # OrderedDict()
        f1_table[train_name] = {}  # OrderedDict()
        for j in range(i + 1):
            val_name = task_names[j]

            logging.info('validation split name: %s', val_name)
            print('validation split name:', val_name)
            val_data = val_dataset_splits[val_name] if not args.eval_on_train_set else train_dataset_splits[val_name]
            val_loader = torch.utils.data.DataLoader(val_data,
                                                     batch_size=args.batch_size, shuffle=False,
                                                     num_workers=0)
            acc_table[train_name][val_name], f1_table[train_name][val_name] = agent.validation(val_loader, val_name)
            
            """ table's format :
            [FER: FER
            AU:  FER,  AU]
            """
    
    return acc_table, task_names, f1_table

def get_args(argv):
    # This function prepares the variables shared across demo.py
    parser = argparse.ArgumentParser(description="Argument Parser for ConFAR.")
    parser.add_argument('--gpuid', nargs="+", type=int, default=[-1], help="The list of gpuid, ex:--gpuid 3 1. Negative value means cpu-only")
    parser.add_argument('--model_type', type=str, default='custom_cnn', help="The type (mlp|lenet|vgg|resnet) of backbone network")
    parser.add_argument('--model_name', type=str, default='Net', help="The name of actual model for the backbone")
    parser.add_argument('--force_out_dim_fer', type=int, default=7, help="Set 0 to let the task decide the required output dimension")
    parser.add_argument('--force_out_dim_av', type=int, default=2, help="Set 0 to let the task decide the required output dimension")
    parser.add_argument('--force_out_dim_au', type=int, default=12, help="Set 0 to let the task decide the required output dimension")
    parser.add_argument('--agent_type', type=str, default='customization', help="The type (filename) of agent")
    parser.add_argument('--agent_name', type=str, default='NormalNN', help="The class name of agent")
    parser.add_argument('--optimizer', type=str, default='Adam', help="SGD|Adam|RMSprop|amsgrad|Adadelta|Adagrad|Adamax ...")
    parser.add_argument('--dataroot', type=str, default='data', help="The root folder of dataset or downloaded data")
    parser.add_argument('--dataset', type=str, default='multiple', help="Multiple: RAFDB|BP4D|DISFA|AffWild2 , Single: AffWild2")
    parser.add_argument('--n_permutation', type=int, default=0, help="Enable permuted tests when >0")
    parser.add_argument('--first_split_size', type=int, default=2)
    parser.add_argument('--other_split_size', type=int, default=2)
    parser.add_argument('--no_class_remap', dest='no_class_remap', default=False, action='store_true',
                        help="Avoid the dataset with a subset of classes doing the remapping. Ex: [2,5,6 ...] -> [0,1,2 ...]")
    parser.add_argument('--train_aug', dest='train_aug', default=False, action='store_true',
                        help="Allow data augmentation during training")
    parser.add_argument('--rand_split', dest='rand_split', default=False, action='store_true',
                        help="Randomize the classes in splits")
    parser.add_argument('--rand_split_order', dest='rand_split_order', default=False, action='store_true',
                        help="Randomize the order of splits")
    parser.add_argument('--workers', type=int, default=3, help="#Thread for dataloader")
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.0001, help="Learning rate")
    parser.add_argument('--momentum', type=float, default=0)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--schedule', nargs="+", type=int, default=[1],
                        help="The list of epoch numbers to reduce learning rate by factor of 0.1. Last number is the end epoch")
    parser.add_argument('--print_freq', type=float, default=100, help="Print the log at every x iteration")
    parser.add_argument('--model_weights', type=str, default=None,
                        help="The path to the file for the model weights (*.pth).")
    parser.add_argument('--reg_coef', nargs="+", type=float, default=[0.],
                        help="The coefficient for regularization. Larger means less plasilicity. Give a list for hyperparameter search.")
    parser.add_argument('--eval_on_train_set', dest='eval_on_train_set', default=False, action='store_true',
                        help="Force the evaluation on train set")
    parser.add_argument('--offline_training', dest='offline_training', default=False, action='store_true',
                        help="Non-incremental learning by make all data available in one batch. For measuring the upperbound performance.")
    parser.add_argument('--repeat', type=int, default=1, help="Repeat the experiment N times")
    parser.add_argument('--incremental_class', dest='incremental_class', default=False, action='store_true',
                        help="The number of output node in the single-headed model increases along with new categories.")
    parser.add_argument('--category', type=str, default='ConFAR', help="The Category (FER, AU, AV, ConFAR)")
    parser.add_argument('--image_size', type=int, default=224, help="Image Size(M) such that MxMx3")
    parser.add_argument('--datasize', type=int, default=None, help="Maximum number of images to use from the dataset")
    parser.add_argument('--bound', type=str, default="lower", help="to calc LB / UB")
    args = parser.parse_args(argv)
    return args


if __name__ == '__main__':
    
    input_args = sys.argv[1:]
    args = get_args(input_args)
    # Configure logging
    
    logging.basicConfig(filename='temp.log', level=logging.DEBUG, format='%(asctime)s %(levelname)s:%(message)s')

    logging.info("got args: %s",input_args)
    
    print("got args, entered main")
    avg_final_acc = {}
    # The for loops over hyper-paramerters or repeats
    for reg_coef in args.reg_coef:
        avg_final_acc[reg_coef] = np.zeros(args.repeat)
        for r in range(args.repeat):
            # Run the experiment
            acc_table, task_names, f1_table = run(args=args)

            print("\n------------------------------------------------------------")
            logging.info("\n------------------------------------------------------------")
            
            #logging.info("The Accuracy table is %s", acc_table)
            #print("The Accuracy table is ", acc_table)
            # Assuming acc_table, f1_table, and some CCC scores dictionary are available

            # Header
            header = ["Train/Test"] + list(acc_table.keys())
            header_line = "{:<12}" + "{:>12}" * len(acc_table)

            logging.info("%s",header_line.format(*header))
            print(header_line.format(*header))

            # Divider
            logging.info("------------------------------------------------")
            print("-" * 12 * (len(acc_table) + 1))

            # Rows
            for train_task in acc_table:
                row = [train_task]
                for test_task in acc_table.keys():
                    acc = acc_table[train_task].get(test_task, "N/A")  # Retrieve accuracy or default to "N/A"
                    row.append(f"{acc:>12.6f}" if isinstance(acc, float) else "{:>12}".format(acc))
                print("{:<12}".format(row[0]) + "".join(row[1:]))
                logging.info("{:<12}".format(row[0]) + "".join(row[1:]))
            
            name = args.category if args.train_aug == False else args.category + "_augmented"
            if not os.path.exists(f'results'):
                os.makedirs(f'results')
            with open('results' + '/' + name + '.txt', 'a') as f:
                f.write(
                    "\n" + str(acc_table) + "f1: " + str(f1_table) + " repeat:" + str(r) + " reg_coef: " + str(reg_coef) + " " + str(args.agent_name))
            # Calculate average performance across tasks
            # Customize this part for a different performance metric
            
            avg_acc_history = [0] * len(task_names)
            cls_acc_sum_fer,cls_acc_sum_au,cls_acc_sum_av = [],[],[]
            for i in range(len(task_names)):
                train_name = task_names[i]
                
                for j in range(i + 1):

                    val_name = task_names[j]

                    if val_name == "AU":
                        cls_acc_sum_au.append(acc_table[train_name][val_name])
                    elif val_name == "AV":
                        cls_acc_sum_av.append(acc_table[train_name][val_name])
                    elif val_name == "FER":
                        cls_acc_sum_fer.append(acc_table[train_name][val_name])
                
                # avg_acc_history[i] = cls_acc_sum / (i + 1)
                        
            for i in range(len(task_names)):
                train_name = task_names[i]
                if train_name == "AU":
                    li = cls_acc_sum_au
                    logging.info('\nTask %s average f1: %s',train_name,sum(li)/len(li))
                    print('\nTask', train_name, 'average f1:', sum(li)/len(li))
                elif train_name == "AV":
                    li = cls_acc_sum_av
                    logging.info('\nTask %s average ccc: %s',train_name,sum(li)/len(li))
                    print('\nTask', train_name, 'average ccc:', sum(li)/len(li))
                elif train_name == "FER":
                    li = cls_acc_sum_fer
                    logging.info('\nTask %s average acc: %s',train_name,sum(li)/len(li))
                    print('\nTask', train_name, 'average acc:', sum(li)/len(li))
                
                
            # Gather the final avg accuracy
            avg_final_acc[reg_coef][r] = avg_acc_history[-1]
            
            # Print the summary so far
            logging.info('\n===Summary of experiment repeats: %s / %s ===', r + 1, args.repeat)
            logging.info('The regularization coefficient: %s', reg_coef)
            logging.info('The last avg acc of all repeats: %s', avg_final_acc[reg_coef])
            logging.info('mean: %s std: %s', avg_final_acc[reg_coef].mean(),avg_final_acc[reg_coef].std())

            print('\n===Summary of experiment repeats:', r + 1, '/', args.repeat, '===')
            print('The regularization coefficient:', reg_coef)
            print('The last avg acc of all repeats:', avg_final_acc[reg_coef])
            print('mean:', avg_final_acc[reg_coef].mean(), 'std:', avg_final_acc[reg_coef].std())
    for reg_coef, v in avg_final_acc.items():
        logging.info('reg_coef: %s / mean: %s / std: %s', reg_coef, avg_final_acc[reg_coef].mean(), avg_final_acc[reg_coef].std())
        print('reg_coef:', reg_coef, 'mean:', avg_final_acc[reg_coef].mean(), 'std:', avg_final_acc[reg_coef].std())
