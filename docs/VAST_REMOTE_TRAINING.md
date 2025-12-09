# Remote Training on VAST AI (SSH + Docker) - Step-by-step Guide

This document walks you through: 1) creating an SSH key on Windows and connecting to a VAST AI Ubuntu instance, and 2) deploying and training models (TFT/XGBoost/LSTM/Linear) from the MIDAS repo.

---

## 1) Generate an SSH key on Windows (PowerShell)

Open PowerShell and run:

```powershell
# Generate an ed25519 key (recommended)
ssh-keygen -t ed25519 -C "your_email@example.com" -f $env:USERPROFILE\.ssh\midas_vast_ed25519

# Or use RSA if you prefer
ssh-keygen -t rsa -b 4096 -C "your_email@example.com" -f $env:USERPROFILE\.ssh\midas_vast_rsa
```

Commands will prompt for a passphrase (recommended). This creates two files:
- `C:\Users\<you>\.ssh\midas_vast_ed25519` (private key)
- `C:\Users\<you>\.ssh\midas_vast_ed25519.pub` (public key)

Add your private key to the local ssh-agent (persist between restarts):

```powershell
# Start ssh-agent (requires admin) - only first time
Start-Service ssh-agent
# Or use (if using Windows OpenSSH) - run as admin

# Add key to agent
ssh-add $env:USERPROFILE\.ssh\midas_vast_ed25519
```

Note: You can use Git Bash or WSL if you prefer; the steps are identical (`ssh-keygen`, `ssh-add`).

---

## 2) Provision VAST AI Instance & Upload Public Key

You can either add your SSH public key during VAST instance creation in the VAST UI or paste the public key into the instance after creation:

- Get the contents of your public key:

```powershell
Get-Content $env:USERPROFILE\.ssh\midas_vast_ed25519.pub
```

Copy the full public key (starts with `ssh-ed25519` or `ssh-rsa`).

### Add via VAST web UI
- On instance creation, choose the "SSH Key" option and paste your public key.

### Add via running instance (if you already have a password)

SSH into the instance with password-based access or from the VAST console, and run:

```bash
# On the VAST Ubuntu host
mkdir -p ~/.ssh
echo "ssh-ed25519 AAAA... user@host" >> ~/.ssh/authorized_keys
chmod 700 ~/.ssh
chmod 600 ~/.ssh/authorized_keys
```

Replace the key string with the contents from your local `*.pub` file.

---

## 3) Connect to your VAST instance via SSH (PowerShell)

VAST typically publishes an IP and a port for SSH. Use the IP and port displayed in VAST UI. Example:

```powershell
$VAST_IP = "<VAST_IP>"
$VAST_PORT = 22 # or port assigned by VAST
ssh -p $VAST_PORT root@$VAST_IP -i $env:USERPROFILE\.ssh\midas_vast_ed25519
```

If you added your key to ssh-agent, you can omit `-i`.

Once connected, verify:

```bash
# On the VAST host (Ubuntu)
whoami
uname -a
lsb_release -a
```

---

## 4) Transfer Project Files to VAST (rsync or scp)

From your Windows machine you can use `scp` (PowerShell/Git Bash) or `rsync` (WSL or Git Bash). The project root path `c:\Users\olo\Programmes\MIDAS` is local.

### Using scp

```powershell
# Copy the entire repo to /workspace (recursively)
scp -P $VAST_PORT -r .\training\ root@$VAST_IP:/workspace/
# Or copy all files
scp -P $VAST_PORT -r .\* root@$VAST_IP:/workspace/midas/
```

### Using rsync (WSL / Git Bash recommended)

```bash
rsync -avz -e "ssh -p $VAST_PORT" . root@$VAST_IP:/workspace/midas/
```

---

## 5) Verify Docker, GPU and Docker Compose on VAST

SSH into the instance and verify:

```bash
# Verify driver (NVIDIA)
nvidia-smi

# Verify Docker
docker --version

# Verify Docker Compose (v2)
docker compose version

# Optional: test GPU in Docker
docker run --gpus all --rm nvidia/cuda:12.1.1-base-ubuntu20.04 nvidia-smi
```

If `nvidia-smi` doesn't show, make sure the VAST instance type includes GPU and NVIDIA libraries are properly installed (VAST should configure that by default for GPU images).

---

## 6) Build / Run MIDAS Training in Docker

From the repository root on the VAST Ubuntu host (assume you uploaded to `/workspace/midas`):

```bash
cd /workspace/midas
# Build training image
docker compose -f docker-compose.training.yml build training

# Run the training container interactively (nvidia GPU required)
docker compose -f docker-compose.training.yml run --rm training bash
```

Or run the training script directly inside the container (non-interactively):

```bash
# Example: run TFT training with one GPU
docker compose -f docker-compose.training.yml run --rm training python training/train.py --data-dir data/features --epochs 100 --gpus 1 --batch-size 64
```

Note: `--gpus 1` is required, also ensure `docker compose` picks up GPU capability in your `docker-compose.training.yml` (it should under `deploy` or `runtime: nvidia`).

---

## 7) Environment Check and Quick Verification

Inside the training container or on the host (with Python available), run the environment checks:

```bash
# In container
python training/run_env_check.py

# The script checks for CUDA availability, Python package versions, and XGBoost GPU support.
```

---

## 8) Feature Preparation & Normalization

The repo provides a features pipeline and normalization utilities. Run the features service to generate features and fit scalers, or use the provided script to normalize existing features.

Run features service (Docker):

```bash
docker compose up -d features
# Wait for feature files to appear
ls data/features/
# Scalers appear in data/scalers/ after the first processed file
ls data/scalers/
```

To normalize existing feature files in batch (without re-running the features service):

```bash
python scripts/normalize_existing_features.py --input-dir data/features --scaler-dir data/scalers --output-dir data/features_normalized
```

Set `--output-dir` to `data/features` to overwrite or to a new directory to keep original files.

---

## 9) Training Examples & Commands

### TensorFlow/Lightning TFT (recommended for best performance):

```bash
python training/train.py --data-dir data/features --epochs 100 --gpus 1 --batch-size 64 --hidden-dim 64
```

### GPU XGBoost (fast + GPU enabled):

```bash
python training/gpu_project/train_xgboost.py --data-dir data/features
python training/gpu_project/train_xgboost.py --data-dir data/features --hyperparameter-search --n-trials 50
```

### LSTM (quick experiments):

```bash
python training/gpu_project/train_lstm.py --data-dir data/features --sequence-length 60 --n-epochs 50
```

### Ridge / Lasso Baselines:

```bash
python training/gpu_project/train_linear.py --model-type ridge --data-dir data/features
```

---

## 10) Monitoring Training

Use TensorBoard to monitor training; start it on the GPU instance and map the port to a public port or create an SSH tunnel to view locally.

Start TensorBoard inside the container:

```bash
tensorboard --logdir logs/tensorboard --host 0.0.0.0 --port 6006
```

SSH tunnel (PowerShell):

```powershell
ssh -L 6006:localhost:6006 -p $VAST_PORT root@$VAST_IP -i $env:USERPROFILE\\.ssh\\midas_vast_ed25519
# Then open http://localhost:6006 in your local browser
```

---

## 11) Backtesting & Evaluation

Run the backtest script after training to see trade simulation metrics and P&L.

```bash
python training/backtest.py --model models/best_model.pt --data-dir data/features --normalizer models/normalizer.json
```

Save results and plots under `reports/`.

---

## 12) Persisting Models & Artifacts

Ensure `models/`, `logs/` and `data/` are persisted to a mounted volume or external storage (like S3) if your instance is ephemeral.

Copy artifacts locally using scp when done:

```powershell
scp -P $VAST_PORT -r root@$VAST_IP:/workspace/midas/models ./models
```

---

## 13) Cleaning Up Resources

When you finish, stop containers and shut down the VAST instance to avoid charges:

```bash
docker compose -f docker-compose.training.yml down --volumes --remove-orphans
# Optionally remove the Docker images
docker image prune -af
```

---

## 14) Common Troubleshooting

- "CUDA not found": Ensure your VAST instance has GPU and the Docker runtime is configured with `--gpus` and the container supports CUDA.
- "XGBoost GPU not available": install `xgboost` with GPU support or ensure the `xgboost` version is compatible.
- "OutOfMemoryError": Reduce batch size / sequence length / model size.
- "No feature files": Run `features` pipeline or move Parquet files into `data/features`.

---

## 15) Optional: Helpful PowerShell automation snippets

Sync the repo to the VAST host (PowerShell):

```powershell
#$VAST_PORT and $VAST_IP previously defined
$rsync = "rsync -avz -e \"ssh -p $VAST_PORT -i $env:USERPROFILE\\.ssh\\midas_vast_ed25519\" . root@$VAST_IP:/workspace/midas/"
# If rsync is not available on Windows, use scp or WSL
scp -P $VAST_PORT -r .\* root@$VAST_IP:/workspace/midas/
```

If you want, I can generate a PowerShell script that performs all of the steps above (copy, start features, run env checks, run training, download models), or a small `bash` script to be run inside the VAST instance.

---

## 16) Where to go next

- Want me to create a full automated `run_experiment.sh` that trains, logs, and stores artifacts automatically? I can add that to the repo.
- Want an example `docker compose` setup that exposes TensorBoard and persists `models` and `data` to `/workspace/persistent`? I can add it.

Good luck! If you want, I can also create a PowerShell script to automate the upload / training steps.
