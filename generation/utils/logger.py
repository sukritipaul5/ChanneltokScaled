import logging
import torch.distributed as dist

def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    # Check if distributed is initialized
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
    else:
        rank = 0  # Default to rank 0 for non-distributed mode
    
    if rank == 0:  # real logger
        if logging_dir:
            handlers = [logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        else:
            handlers = [logging.StreamHandler()]
        
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=handlers
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger