import argparse                                                                                                                                                 
import getpass                                                                                                                     
import shutil                                                                                     
import socket                                                                                                                            
import subprocess                                                                   
from pathlib import Path        
import os             
import time                                                                       
                                                
                                                        
def cancel_all_my_non_bash_jobs():                                                                            
    print("Cancelling all non-bash jobs.")             
    jobs_status = (                                                                                                   
        subprocess.check_output(f"squeue -u {getpass.getuser()}", shell=True)
        .decode()                                                                            
        .split("\n")[1:-1]                                                                                                       
    )                                 
    non_bash_job_ids = []
    for job_status in jobs_status:
        if not ("bash" in job_status.split() or "zsh" in job_status.split()):
            non_bash_job_ids.append(job_status.split()[0])         
    if len(non_bash_job_ids) > 0:      
        cmd = "scancel {}".format(" ".join(non_bash_job_ids))
        print(cmd)                                                                                                                                
        print(subprocess.check_output(cmd, shell=True).decode())                                                                                          
    else:                                                                                                                                    
        print("No non-bash jobs to cancel.")                                                                                  
                                                                                                                      
                                                                                                             
def args_to_str(args):           
    result = ""                        
    for k, v in vars(args).items():                                    
        k_str = k.replace("_", "-")
        if v is None:                                                                                                                                           
            pass                                                                                                                                                
        elif isinstance(v, bool):                                                                                                                               
            if v:                                                                                                                                               
                result += " --{}".format(k_str)                                                                                                                 
        else:                                                                                                                                                   
            if isinstance(v, list):                                                                                                                             
                v_str = " ".join(map(str, v))                                                                                                                   
            else:                                                                                                                                               
                v_str = v                                                                                                                                       
            result += " --{} {}".format(k_str, v_str)                                                                                                           
    return result                                                                                                                                               
                                                                                                                                                                
                                                                                                                                                                
def submit_slurm_job(run_args, logs_dir, job_name, no_repeat=False, network="state_recog"):  
    possible_networks = ["state_recog", "nnL0_goal_recog", "nnL1_goal_recog", "nnL1_action_recog"]                                                              
    # SBATCH AND PYTHON CMD                                                                                                                                     
    args_str = args_to_str(run_args)                                                                                                                                                                                                                                                          
    sbatch_cmd = "sbatch"                                                                                                                             
    time_option = "48:0:0"
    if network == "state_recog":
        python_cmd = f'python -u train_belief_nn.py {args_str}' 
    elif network == "nnL0_goal_recog":
        python_cmd = f'python -u train_car_action_pred.py {args_str}' 
    else:
        python_cmd = f'python -u train_belief_nn.py {args_str}'                                                                                                 
                                                                                             
    if not no_repeat:                                                                                                                                                     
        sbatch_cmd = "om-repeat sbatch"
        time_option = "96:0:0"                                                                                                                 
        python_cmd = f"MKL_THREADING_LAYER=INTEL=1 {python_cmd}"

    # SBATCH OPTIONS
    Path(logs_dir).mkdir(parents=True, exist_ok=True)

    bash_file = [
        f"#!/bin/bash -l",
        f"#SBATCH --time={time_option}",
        f"#SBATCH --ntasks=1",
        f"#SBATCH -n 2",
        f"#SBATCH --gres=gpu:1",
        f"#SBATCH -p tenenbaum",
        f"#SBATCH --constraint=20GB",
        f"#SBATCH --mem=30G",
        f'#SBATCH -J "carSim"',
        f'#SBATCH -o "/om2/user/kunaljha/RecursiveReasoning/CARLO/save/rebuttal/%j.out"',
        f'#SBATCH -e "/om2/user/kunaljha/RecursiveReasoning/CARLO/save/rebuttal/%j.err"',
        " ",
        f'source ~/.bashrc',
        f'which python',
        " ",
        python_cmd  
    ]
    joined_bash = "\n".join(bash_file)

    temp_shell = open('temp_shell.sh', 'w')
    n = temp_shell.write(joined_bash)
    temp_shell.close()
    subprocess.call(f'{sbatch_cmd} temp_shell.sh', shell=True)

    # cleaning up
    os.remove('temp_shell.sh')


def submit_slurm_jobs(
    get_run_argss, get_config_name, get_job_name, no_repeat=True, cancel=False, rm=False
):
    if cancel:
        cancel_all_my_non_bash_jobs()

    if rm:
        dir_ = f"save/{next(iter(get_run_argss())).experiment_name}"
        if Path(dir_).exists():
            shutil.rmtree(dir_, ignore_errors=True)

    print(f"Launching {len(list(get_run_argss()))} runs from {socket.gethostname()}")
    for run_args, network in get_run_argss():
        config_name = get_config_name(run_args)
        job_name = get_job_name(run_args)
        submit_slurm_job(
            run_args,
            # f"save/scenario1/{run_args.experiment_name}/{config_name}/logs",
            f"save/scenario2/{run_args.experiment_name}/{config_name}/logs",
            job_name,
            no_repeat=no_repeat,
            network=network
        )


def get_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--rm", action="store_true", help="")
    parser.add_argument(
        "--no-repeat",
        action="store_true",
        help="run the jobs using standard sbatch."
        "if False, queues 2h jobs with dependencies "
        "until the script finishes",
    )
    parser.add_argument(
        "--cancel", action="store_true", help="cancels all non-bash jobs if run on the cluster"
    )
    return parser
