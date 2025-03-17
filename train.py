from utils import get_device
from utils import set_seed
from model import GPT2
from config import GPT2Config
from dataloader import DataLoader
import torch
import torch.nn.functional as F
import time
from scheduler import lr_scheduler
import params
import os

if __name__ == "__main__":

    device = get_device()
    print("Using device:", device)
    set_seed(params.seed)

    model = GPT2(GPT2Config(vocab_size=50304)) # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for kernel efficiency
    model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params}")

    B = params.B
    T = params.T
    total_batch_size = params.total_batch_size
    assert total_batch_size % (B * T) == 0, "Total batch size must be divisible by B * T"
    grad_accum_steps = total_batch_size // (B * T)
    print(f"Using batch size {B} and sequence length {T}")
    print(f"Total desired batch size: {total_batch_size}")
    print(f"Gradient accumulation steps: {grad_accum_steps}")

    train_loader = DataLoader(B=B, T=T, split="train")
    val_loader = DataLoader(B=B, T=T, split="val")

    optimizer = model.configure_optimizers(weight_decay = params.weight_decay, learning_rate = params.max_lr, device = device)

    log_dir = "log"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"log.txt")
    with open(log_file, "w") as f: # open for writing to clear the file
        pass

    for step in range(params.max_steps):
        t0 = time.time()
        last_step = (step == params.max_steps - 1)

            
        # Training
        model.train()
        optimizer.zero_grad()
        loss_accum = 0.0
        for micro_step in range(grad_accum_steps):
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)
            logits, loss = model(x, y)
            loss = loss / grad_accum_steps
            loss_accum += loss.detach()
            loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        lr = lr_scheduler(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        optimizer.step()
        torch.mps.synchronize()
        t1 = time.time()
        dt = (t1 - t0) * 1000
        tokens_per_sec = (train_loader.B * train_loader.T * grad_accum_steps) / (dt / 1000)
        print(f"Step {step+1:4d} | Loss {loss_accum.item():.6f} | LR: {lr:.4e} | Norm: {norm:.4f} | Time: {dt:.2f}ms | Tokens/sec: {tokens_per_sec:.2f}")
        with open(log_file, "a") as f:
            f.write(f"Step {step+1:4d} | Loss {loss_accum.item():.6f} | LR: {lr:.4e} | Norm: {norm:.4f} | Time: {dt:.2f}ms | Tokens/sec: {tokens_per_sec:.2f}\n")

        # Validation every 100 steps
        if (step % 100 == 0 or last_step) and step > 0:
            model.eval()
            with torch.no_grad():
                val_loss_accum = 0.0
                val_loss_steps = 20
                for _ in range(val_loss_steps):
                    x, y = val_loader.next_batch()
                    x, y = x.to(device), y.to(device)
                    logits, loss = model(x, y)
                    loss = loss / val_loss_steps
                    val_loss_accum += loss.detach()
                print(f"Step: {step+1:4d} | Validation Loss {val_loss_accum.item():.6f}")
                with open(log_file, "a") as f:
                    f.write(f"Step: {step+1:4d} | Validation Loss: {val_loss_accum.item():.6f}\n")

        # Save checkpoint every 5000 steps or at the end of training
        if step > 0 and (step % 5000 == 0 or last_step):
            checkpoint_path = os.path.join(log_dir, f"model_{step:05d}.pt")
            checkpoint = {
                'model': model.state_dict(),
                'config': model.config,
                'step': step,
                'val_loss': val_loss_accum.item(),
                'optimizer': optimizer.state_dict()
            }
            torch.save(checkpoint, checkpoint_path)
