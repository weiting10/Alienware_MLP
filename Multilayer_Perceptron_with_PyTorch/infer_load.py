import argparse, time
import torch, torch.nn as nn, torch.nn.functional as F

class Multilayer_Perceptron(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_fc = nn.Linear(input_dim, 250)
        self.hidden_fc = nn.Linear(250, 100)
        self.output_fc = nn.Linear(100, output_dim)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = F.relu(self.input_fc(x))
        x = F.relu(self.hidden_fc(x))
        return self.output_fc(x)

# can customize batches with  --batches
# can customize load,idle time with --load-s --idle-s 


p = argparse.ArgumentParser()
p.add_argument("--batches", type=int, nargs="+", default=[1024, 8192, 65536])
p.add_argument("--load-s", type=float, default=5)   # seconds of load per phase
p.add_argument("--idle-s", type=float, default=5)   # seconds of idle between phases
args = p.parse_args()

device = torch.device("cuda")
model = Multilayer_Perceptron(784, 10).to(device)
model.load_state_dict(torch.load("tut1-model.pt", map_location=device))
model.eval()

def run_load(x, seconds):
    end = time.perf_counter() + seconds
    with torch.no_grad():
        while time.perf_counter() < end:
            for _ in range(50):          # queue many kernels back to back
                model(x)
            torch.cuda.synchronize()

# warm-up so CUDA init / library loading doesn't land in the measured window
with torch.no_grad():
    model(torch.randn(1024, 1, 28, 28, device=device)); torch.cuda.synchronize()

print("Starting phases"); t0 = time.time()
time.sleep(args.idle_s)
for b in args.batches:
    x = torch.randn(b, 1, 28, 28, device=device)
    print(f"[{time.time()-t0:6.1f}s] load: batch={b}")
    run_load(x, args.load_s)
    print(f"[{time.time()-t0:6.1f}s] idle")
    time.sleep(args.idle_s)
print("Done")
