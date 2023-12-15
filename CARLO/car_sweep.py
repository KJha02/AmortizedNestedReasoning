import car_utils.slurm                                                                                                                                              
import train_car_action_pred
import train_car_nnL1
import train_belief_nn
import torch
                                                             
                        
def get_run_argss():                                                                                                      
    # sweep of params
    possible_networks = ["state_recog", "nnL0_goal_recog", "nnL1_goal_recog", "nnL1_action_recog"]
    network = possible_networks[0]  # choose from one of the above options
    for num_samples_L2 in [3]:
        for num_samples in [3]:
            experiment_name = ""
            for lookAheadDepth in [10]:
                for num_train_dat in [1000]:                                                                                                    
                    for beta in [0.01]:                                                                                               
                            for hidDim in [128]:                                                                                                                                                                 
                                for lr in [1e-4]:
                                    for bSize in [64]:      
                                        if network == "state_recog":
                                            args = train_belief_nn.get_args_parser().parse_args([])
                                        elif network == "nnL0_goal_recog":
                                            args = train_car_action_pred.get_args_parser().parse_args([])
                                        else:
                                            args = train_car_nnL1.get_args_parser().parse_args([])
                                            if network == "nnL1_action_recog":
                                                args.predictActions = True
                                                                                           
                                        args.beta = beta   
                                        args.lookAheadDepth = lookAheadDepth       
                                        args.lr = lr    
                                        args.num_samples = num_samples  # how many particles will be used during inference
                                        args.num_samples_L2 = num_samples_L2
                                        args.num_data_train=num_train_dat
                                        args.num_data_test=int(num_train_dat * 0.1)                                                           
                                        args.hidden_dim = hidDim                                                                  
                                        args.batch_size = bSize 
                                        args.num_epochs = 250
                                        args.experiment_name = experiment_name + str(num_train_dat * 3 / 1000) + "kDat_"  + str(args.hidden_dim) + \
                                            "dim_" + str(args.lr) + "lr_" + str(args.batch_size) + "bSize"
                                        yield (args, network)
                        
                             
def get_job_name(run_args):
    return train_car_action_pred.get_config_name(run_args)
                                       
                    
def main(args):                                                                                                                                   
    car_utils.slurm.submit_slurm_jobs(                                                                                                                        
        get_run_argss,                                                                                                                       
        train_car_action_pred.get_config_name,                                                                       
        get_job_name,                                                                                                 
        True,             # don't want to use om-repeat                                                      
        args.cancel,             
        args.rm,                       
    )                                                                  
                               
                                                                                                                                                                
if __name__ == "__main__":  
    torch.multiprocessing.set_start_method('spawn')                                                                                                                                    
    parser = car_utils.slurm.get_parser()                                                                                                                           
    args = parser.parse_args()                                                                                                                                  
    main(args)     