from .losses import *
from .sampler import *


mmd_losses = {
    'mmd': mmd_loss,
    'jmmd': jmmd_loss
}

samplers = {
    'fix': fix_sampler,
    'random': random_sampler
}
