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
                        help="The number of processes to launch on each node, "
                             "for GPU training, this is recommended to be set "
                             "to the number of GPUs in your system so that "
                             "each process can be bound to a single GPU.")
    parser.add_argument("--master_addr", default="127.0.0.1", type=str,
                        help="Master node (rank 0)'s address, should be either "
                             "the IP address or the hostname of node 0, for "
                             "single node multi-proc training, the "
                             "--master_addr can simply be 127.0.0.1")

    # Set the port (if desired, else just keep the default)
    parser.add_argument("--master_port", default=29500, type=int,
                        help="Master node (rank 0)'s free port that needs to "
                             "be used for communication during distributed "
                             "training")

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
        print("*****************************************\n"
              "Setting OMP_NUM_THREADS environment variable for each process "
              "to be {} in default, to avoid your system being overloaded, "
              "please further tune the variable for optimal performance in "
              "your application as needed. \n"
              "*****************************************".format(os.environ["OMP_NUM_THREADS"]))

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
