import torch

# multi-threaded CPU torch stalls on small ops on some machines; the circuit tests are tiny
torch.set_num_threads(1)
