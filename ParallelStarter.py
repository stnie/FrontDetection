import torch.multiprocessing as mp
import argparse
import os


def ParseArguments(parseScriptArgs = None):       
    parser = argparse.ArgumentParser(description='ParallelParser')
    # Use SLURM TO SETUP COMMUNICATION
    parser.add_argument("--slurm", action='store_true', default=False, 
                        help="Set number of nodes, master node and "
                        "nproc per node from slurm environment variables")
    # Communication 
    parser.add_argument("--nnodes", type=int, default=1,
                        help="The number of nodes to use for distributed "
                             "training")
    parser.add_argument("--node_rank", type=int, default=0,
                        help="The rank of the node for multi-node distributed "
                             "training")
    parser.add_argument("--nproc_per_node", type=int, default=1,
                        help="The number of processes to use per node")
    parser.add_argument("--master_addr", default="127.0.0.1", type=str,
                        help="ip-Address of node with rank 0, default is fine for single node")

    # Set the port 
    parser.add_argument("--master_port", default=29500, type=int,
                        help="port of node with rank 0 for communication, default is fine for single node")

    # Additional Arguments to be parsed!
    parseScriptArgs(parser)
    
    return parser.parse_args()

def PrepareMultiprocessing(args, func):
    if(args.slurm):
        args.node_rank = int(os.environ["SLURM_NODEID"])
        args.master_addr = os.environ["SLURM_SRUN_COMM_HOST"]
        args.nnodes = int(os.environ["SLURM_STEP_NUM_NODES"])
        args.nproc_per_node = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))
        gpuList = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
        args.available_gpu = [int(gpu) for gpu in gpuList]

    world_size = args.nproc_per_node * args.nnodes
    local_ranks = args.nproc_per_node
    node_rank = args.node_rank
    os.environ["MASTER_ADDR"] = args.master_addr
    os.environ["MASTER_PORT"] = str(args.master_port)
    os.environ["WORLD_SIZE"] = str(world_size)
    #os.environ["NCCL_SOCKET_IFNAME"] = "ib"

    processes = []
    if 'OMP_NUM_THREADS' not in os.environ and args.nproc_per_node > 1:
        os.environ["OMP_NUM_THREADS"] = str(1)

    #Spawny
    if(world_size > 1):
        args.distributed = True
        globalLock = mp.Lock()
        prcs = mp.start_processes(func, args=[args, globalLock], nprocs=local_ranks, start_method='forkserver')
        
    else:
        args.distributed = False
        func(0,args,mp.Lock())
    

def BeginMultiprocessing(ArgumentParserFunc, TrainingFunc):
    #set MP forkserver method
    mp.set_start_method("forkserver") #Might be unnecessary
    # Parse input arguments 
    args = ParseArguments(ArgumentParserFunc)
    # Spawn Threads for training
    PrepareMultiprocessing(args, TrainingFunc)
