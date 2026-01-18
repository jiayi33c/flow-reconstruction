#!/usr/bin/env bash
set -e

###############################################################################
# Run this INSIDE the Docker container.
# Assumes:
#   - PhysicsNEMO repo is mounted at /workspace
#   - You already have conf/config_dfsr_cond_train_debug.yaml
###############################################################################

echo ">>> [0] Go to flow_reconstruction_diffusion example dir"
cd /workspace/examples/cfd/flow_reconstruction_diffusion

###############################################################################
# 1. Install Python dependencies (idempotent)
###############################################################################
echo ">>> [1] Installing Python dependencies (physicsnemo, cftime, termcolor, hydra-core)"
python -m pip install physicsnemo cftime termcolor hydra-core

###############################################################################
# 2. Patch train.py logging so Hydra configs don't break json
###############################################################################
echo ">>> [2] Patching train.py logging + training_options writer (safe if already patched)"

python - << 'PYEOF'
from pathlib import Path
path = Path("train.py")
text = path.read_text()

# --- Patch 1: logger0.info(json.dumps(c, indent=2)) -> logger0.info(str(c))
old_log = '    logger0.info("Training options:")\n    logger0.info(json.dumps(c, indent=2))\n'
new_log = '    logger0.info("Training options:")\n    logger0.info(str(c))\n'
if old_log in text:
    text = text.replace(old_log, new_log)
    print("Patched logging json.dumps -> str(c)")
else:
    print("[WARN] logging json.dumps block not found (maybe already patched)")

# --- Patch 2: training_options.json write: json.dump -> plain str(c)
old_block = '''    if dist.rank == 0:
        os.makedirs(c.run_dir, exist_ok=True)
        with open(os.path.join(c.run_dir, "training_options.json"), "wt") as f:
            json.dump(c, f, indent=2)
        # utils.Logger(file_name=os.path.join(c.run_dir, 'log.txt'), file_mode='a', should_flush=True)
'''
new_block = '''    if dist.rank == 0:
        os.makedirs(c.run_dir, exist_ok=True)
        with open(os.path.join(c.run_dir, "training_options.json"), "wt") as f:
            # Hydra configs contain objects (e.g. ListConfig) that are not JSON-serializable.
            # For now we write a readable string representation instead of strict JSON.
            f.write(str(c))
        # utils.Logger(file_name=os.path.join(c.run_dir, 'log.txt'), file_mode='a', should_flush=True)
'''
if old_block in text:
    text = text.replace(old_block, new_block)
    print("Patched training_options.json writer to use str(c)")
else:
    print("[WARN] training_options.json block not found (maybe already patched)")

path.write_text(text)
print(">>> train.py patching complete.")
PYEOF

###############################################################################
# 3. Run training with the existing debug config
###############################################################################
echo ">>> [3] Starting training with config_dfsr_cond_train_debug.yaml"
python train.py --config-name config_dfsr_cond_train

echo ">>> Done. Check ./results_cond_02 for logs and snapshots."
