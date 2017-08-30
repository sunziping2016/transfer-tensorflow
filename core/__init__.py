from .losses import *
from .sampler import *


mmd_losses = {
    'mmd': multiple_mmd_loss,
    'jmmd': jmmd_loss
}

samplers = {
    'fix': fix_sampler,
    'random': random_sampler
}
