from matplotlib import pyplot as plt
from torch import nn
from torch.optim import Adam
import math


class WarmupLinearLRSchedule:
    """
    Implements Warmup learning rate schedule until 'warmup_steps', going from 'init_lr' to 'peak_lr' for multiple optimizers.
    """
    def __init__(self, optimizer, init_lr, peak_lr, end_lr, warmup_epochs, epochs=100, current_step=0):
        self.init_lr = init_lr
        self.peak_lr = peak_lr
        self.optimizer = optimizer
        self.warmup_rate = (peak_lr - init_lr) / warmup_epochs
        self.decay_rate = (end_lr - peak_lr) / (epochs - warmup_epochs)
        self.update_steps = current_step
        self.lr = init_lr
        self.warmup_steps = warmup_epochs
        self.epochs = epochs
        if current_step > 0:
            self.lr = self.peak_lr + self.decay_rate * (current_step - 1 - warmup_epochs)

    def set_lr(self, lr):
        print(f"Setting lr: {lr}")
        for g in self.optimizer.param_groups:
            g['lr'] = lr

    def step(self):
        if self.update_steps <= self.warmup_steps:
            lr = self.init_lr + self.warmup_rate * self.update_steps
        # elif self.warmup_steps < self.update_steps <= self.epochs:
        else:
            lr = max(0., self.lr + self.decay_rate)
        self.set_lr(lr)
        self.lr = lr
        self.update_steps += 1
        return self.lr


class CosineDecayWithWarmupLRSchedule:
    """
    Implements cosine decay learning rate schedule with linear warmup.
    During warmup: linearly increase from init_lr to peak_lr over warmup_steps
    After warmup: cosine decay from peak_lr to min_lr over remaining steps
    """
    def __init__(self, optimizer, init_lr, peak_lr, min_lr, warmup_steps, total_steps, current_step=0):
        self.init_lr = init_lr
        self.peak_lr = peak_lr
        self.min_lr = min_lr
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.current_step = current_step
        
        # Calculate current learning rate if resuming
        if current_step > 0:
            self.lr = self._calculate_lr(current_step)
        else:
            self.lr = init_lr
    
    def _calculate_lr(self, step):
        """Calculate learning rate for given step"""
        if step <= self.warmup_steps:
            # Linear warmup
            return self.init_lr + (self.peak_lr - self.init_lr) * (step / self.warmup_steps)
        else:
            # Cosine decay
            decay_steps = self.total_steps - self.warmup_steps
            decay_step = step - self.warmup_steps
            cosine_decay = 0.5 * (1 + math.cos(math.pi * decay_step / decay_steps))
            return self.min_lr + (self.peak_lr - self.min_lr) * cosine_decay
    
    def set_lr(self, lr):
        """Set learning rate for all parameter groups"""
        for g in self.optimizer.param_groups:
            g['lr'] = lr
    
    def step(self):
        """Update learning rate and increment step counter"""
        self.lr = self._calculate_lr(self.current_step)
        self.set_lr(self.lr)
        self.current_step += 1
        return self.lr
    
    def get_lr(self):
        """Get current learning rate"""
        return self.lr


class CosineDecaySchedule:
    """
    Generic cosine decay schedule for any parameter (not just learning rate).
    Useful for decaying loss weights, regularization terms, etc.
    """
    def __init__(self, start_value, end_value, total_steps, warmup_steps=0, current_step=0):
        """
        Args:
            start_value: Initial value of the parameter
            end_value: Final value after decay
            total_steps: Total number of training steps
            warmup_steps: Number of steps to keep value at start_value before decay
            current_step: Current step (for resuming)
        """
        self.start_value = start_value
        self.end_value = end_value
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.current_step = current_step
        
        # Calculate current value if resuming
        if current_step > 0:
            self.value = self._calculate_value(current_step)
        else:
            self.value = start_value
    
    def _calculate_value(self, step):
        """Calculate parameter value for given step"""
        if step <= self.warmup_steps:
            # During warmup, keep at start value
            return self.start_value
        else:
            # Cosine decay after warmup
            decay_steps = self.total_steps - self.warmup_steps
            decay_step = min(step - self.warmup_steps, decay_steps)
            cosine_decay = 0.5 * (1 + math.cos(math.pi * decay_step / decay_steps))
            return self.end_value + (self.start_value - self.end_value) * cosine_decay
    
    def step(self):
        """Update parameter value and increment step counter"""
        self.value = self._calculate_value(self.current_step)
        self.current_step += 1
        return self.value
    
    def get_value(self):
        """Get current parameter value"""
        return self.value


class PiecewiseLinearSchedule:
    """Piecewise linear schedule with an initial hold, linear ramp, and optional final hold."""

    def __init__(
        self,
        start_value,
        peak_value,
        initial_hold_steps=0,
        ramp_steps=1,
        final_hold_steps=0,
        current_step=0,
    ):
        if ramp_steps <= 0:
            raise ValueError("ramp_steps must be > 0 for PiecewiseLinearSchedule")

        self.start_value = start_value
        self.peak_value = peak_value
        self.initial_hold_steps = max(0, int(initial_hold_steps))
        self.ramp_steps = int(ramp_steps)
        self.final_hold_steps = max(0, int(final_hold_steps))
        self.current_step = current_step

        self.total_schedule_steps = self.initial_hold_steps + self.ramp_steps + self.final_hold_steps
        if self.total_schedule_steps == 0:
            self.total_schedule_steps = self.ramp_steps

        # Initialize current value
        self.value = self._calculate_value(self.current_step)

    def _calculate_value(self, step):
        if step <= self.initial_hold_steps:
            return self.start_value

        ramp_end = self.initial_hold_steps + self.ramp_steps
        if step <= ramp_end:
            progress = (step - self.initial_hold_steps) / max(self.ramp_steps, 1)
            return self.start_value + progress * (self.peak_value - self.start_value)

        # Final stage: hold peak value
        return self.peak_value

    def step(self):
        self.value = self._calculate_value(self.current_step)
        self.current_step += 1
        return self.value

    def get_value(self):
        return self.value


if __name__ == '__main__':
    m = nn.Linear(10, 10)
    opt = Adam(m.parameters(), lr=1e-4)
    s = WarmupLinearLRSchedule(opt, 1e-6, 1e-4, 0., 2)
    lrs = []
    for i in range(101):
        s.step()
        lrs.append(s.lr)
        print(s.lr)

    m = nn.Linear(10, 10)
    opt = Adam(m.parameters(), lr=1e-4)
    s = WarmupLinearLRSchedule(opt, 1e-6, 1e-4, 0., 2, current_step=50)
    lrs_s = []
    for i in range(50, 101):
        s.step()
        lrs_s.append(s.lr)
        print(s.lr)

    plt.plot(lrs, label='Linear Schedule (full)')
    plt.plot(range(50, 101), lrs_s, label='Linear Schedule (resume from 50)')
    
    # Test cosine decay scheduler
    m_cos = nn.Linear(10, 10)
    opt_cos = Adam(m_cos.parameters(), lr=1e-4)
    s_cos = CosineDecayWithWarmupLRSchedule(
        optimizer=opt_cos,
        init_lr=1e-6,
        peak_lr=1e-3,
        min_lr=1e-5,
        warmup_steps=10,
        total_steps=100
    )
    lrs_cos = []
    for i in range(100):
        lr = s_cos.step()
        lrs_cos.append(lr)
    
    plt.plot(lrs_cos, label='Cosine Decay with Warmup')
    
    # Test generic cosine decay (for perceptual weight)
    weight_scheduler = CosineDecaySchedule(
        start_value=0.1,
        end_value=0.01,
        total_steps=100,
        warmup_steps=10
    )
    weights = []
    for i in range(100):
        w = weight_scheduler.step()
        weights.append(w)
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(lrs, label='Linear Schedule (full)')
    plt.plot(range(50, 101), lrs_s, label='Linear Schedule (resume from 50)')
    plt.plot(lrs_cos, label='Cosine Decay with Warmup')
    plt.xlabel('Step')
    plt.ylabel('Learning Rate')
    plt.legend()
    plt.title('Learning Rate Schedules Comparison')
    plt.yscale('log')
    
    plt.subplot(1, 2, 2)
    plt.plot(weights, label='Perceptual Weight Decay', color='green')
    plt.axvline(x=10, color='red', linestyle='--', alpha=0.5, label='Warmup End')
    plt.xlabel('Step')
    plt.ylabel('Weight Value')
    plt.legend()
    plt.title('Perceptual Loss Weight Cosine Decay')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
