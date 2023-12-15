import utils.slurm                                                                                                                                              
import train_construction_desire_pred
import torch
                                                             
                        
def get_run_argss():                                                                                                      
    # sweep of params
    for num_samples in [5]:
        for num_samples_L2 in [2]:
            # experiment_name = f"L1dataGen{num_samples}L2{num_samples_L2}_"  
            # experiment_name = f"nllLoss"
            experiment_name = f"L0_KLLess_10_pct_"
            # for num_train_dat in range(1000, 3000, 100):
            # for num_train_dat in [170]:  # do this for L1 data:
            for num_train_dat in [10000]:
                for last in [0]:                                                                                                              
                    for beta in [0.01]:                                     
                        for num_colored_block_locations, num_possible_block_pairs in [[10, 45]]:                                                               
                            for hidDim in [128]:                                                             
                                for numChan in [128]:                                                                                                            
                                    for lr in [1e-4]:
                                        for bSize in [128]:                                                                                     
                                            args = train_construction_desire_pred.get_args_parser().parse_args([])
                                            args.num_colored_block_locations = num_colored_block_locations                                               
                                            args.num_possible_block_pairs = num_possible_block_pairs
                                            args.utility_mode = "ranking"                                                           
                                            args.last = last
                                            args.beta = beta            
                                            args.lr = lr    
                                            args.num_samples = num_samples  # how many particles will be used during L0 inference
                                            args.num_samples_L2 = num_samples_L2
                                            args.num_data_train=num_train_dat
                                            args.num_data_test=int(num_train_dat * 0.2)                                                           
                                            args.num_channels = numChan
                                            args.hidden_dim = hidDim                                                                  
                                            args.batch_size = bSize 
                                            args.num_epochs = 250
                                            if last == 1:        
                                                args.experiment_name = str(num_train_dat * 6 / 1000) + "kDat_last_" + experiment_name + str(args.hidden_dim) + \
                                                    "dim_" + str(args.num_channels) + "chan_" + str(args.lr) + "lr_" + str(args.batch_size) + "bSize"
                                            else:
                                                args.experiment_name = str(num_train_dat * 6 / 1000) + "kDat_" + experiment_name + str(args.hidden_dim) + \
                                                    "dim_" + str(args.num_channels) + "chan_" + str(args.lr) + "lr_" + str(args.batch_size) + "bSize"
                                            yield args
                        
                             
def get_job_name(run_args):
    return train_construction_desire_pred.get_config_name(run_args)
                                       
                    
def main(args):                                                                                                                                   
    utils.slurm.submit_slurm_jobs(                                                                                                                        
        get_run_argss,                                                                                                                       
        train_construction_desire_pred.get_config_name,                                                                       
        get_job_name,                                                                                                 
        True,             # don't want to use om-repeat                                                      
        args.cancel,             
        args.rm,                       
    )                                                                  
                               
                                                                                                                                                                
if __name__ == "__main__":  
    torch.multiprocessing.set_start_method('spawn')                                                                                                                                    
    parser = utils.slurm.get_parser()                                                                                                                           
    args = parser.parse_args()                                                                                                                                  
    main(args)     