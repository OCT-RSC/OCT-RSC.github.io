# OCT shape control

OCT-guided tissue deformation policies with Full-volume, Tissue-only and
Point-cloud inputs, Explicit or Diffusion action prediction, and optional residuals.

## Install

Use Python 3.12 and install PyTorch for your CPU or CUDA device, then:

```bash
pip install -r requirements.txt
```

The paper training environment uses PyTorch 2.10.0, NumPy 1.26.4 and SciPy 1.16.2.
Tissue-only uses native PyTorch sparse operations.

## Train

Run from this directory:

```bash
python prepare_data.py --data-root /path/to/demonstrations \
  --data-profile configs/data/official_3dof.json \
  --config configs/phantom_point_residual_bc.json --output prepared/point

python train_policy.py --prepared prepared/point \
  --config configs/phantom_point_residual_bc.json --output outputs/point

python evaluate_policy.py --prepared prepared/point \
  --checkpoint outputs/point/checkpoint_best.pt --output outputs/point/validation.json
```

Choose a configuration in `configs/` and use `official_6dof.json` for directional
targets. Preprocessing is shared across heads and residual settings. Demonstrations
are split approximately 80/20 within each target condition using seed 42.
Cache generation uses CPU by default; `--device cuda` selects GPU resizing.
To reproduce the paper's Full-volume caches, use CPU for depth targets and
`--device cuda` for directional targets.
Training saves the best validation EMA weights and a last checkpoint for resuming.
Use `--fresh` to restart or `--epochs` to change the epoch limit.

## Inference

The 28 paper checkpoints are in `checkpoints/`. Their names use
`phantom` or `exvivo`, `full`/`tissue`/`point`, optional `residual`,
and `bc` (Explicit) or `diffusion`. The `_tilt` suffix denotes the
6-DoF directional task; names without it denote 3-DoF depth control.

```bash
python run_policy.py --trajectory /path/to/condition/traj_1 \
  --checkpoint checkpoints/phantom_point_residual_bc.pt --output predictions.json
```

For online observations:

```python
from oct_policy import load_policy

policy = load_policy("checkpoints/phantom_point_residual_bc.pt", device="cuda")
policy.reset("/path/to/traj_1/predeform/target_volume.npy", initial_tooltip_pose)
prediction = policy.step("/path/to/current/ascans.npy", current_tooltip_pose)
action = prediction["action_6dof_mm_deg"]
```

Poses are 4-by-4 transforms with translation in metres. Returned actions are local
translation increments in millimetres and Euler-xyz rotation increments in degrees.
The policy predicts eight actions and returns the first for execution.
These scripts do not send robot commands.
Saved deployment folders with `rollout_step_0`, `rollout_step_1`, etc. are also
accepted by `run_policy.py`. Use `--seed` to override the checkpoint inference seed.

## Demonstration format

```text
condition/traj_1/
  predeform/
    ascans.npy
    target_volume.npy
    target_selection.json
    targets/pre_surface_yx_vox.npy
    targets/.../gaussian_displacement_yx_mm.npy
  1/ascans.npy
  1/step_info.json
  2/ascans.npy
  2/step_info.json
  ...
```

OCT arrays are signed int8 in HWD order, `[256,256,768]`, optionally with outer
singleton dimensions. Target metadata identifies material, spacing and the selected
displacement. Each step records `tooltip_transform_4x4` and
`inputs_6dof_mm_deg`. Keep the predeformation metadata and selected target files
with each trajectory.
