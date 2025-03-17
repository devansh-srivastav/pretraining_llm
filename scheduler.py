import math
import params

max_lr =params.max_lr
min_lr =params.min_lr
warmup_steps = params.warmup_steps
max_steps = params.max_steps

def lr_scheduler(step):
        if step < warmup_steps:
            return max_lr * (step+1) / warmup_steps
        if step >= max_steps:
            return min_lr
        
        decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
        assert 0<=decay_ratio<=1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (max_lr - min_lr)