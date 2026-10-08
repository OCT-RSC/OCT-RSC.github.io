# OCT-RSC

Training and inference code for OCT-guided soft-tissue shape control.

Supported inputs: Full-volume, Tissue-only, and Point-cloud.
Policies: Explicit and Diffusion, with or without residual conditioning.

## Installation

Use Python 3.12 and PyTorch 2.10.0. Install the remaining dependencies:

```bash
pip install -r requirements.txt
```

## Data

Each demonstration contains a predeformation scan, a fixed target, and a sequence of OCT observations and robot commands:

```text
condition/traj_1/
  predeform/
    ascans.npy
    target_volume.npy
    target_selection.json
    targets/pre_surface_yx_vox.npy
    targets/.../gaussian_displacement_yx_mm.npy
  1/
    ascans.npy
    step_info.json
  2/
    ascans.npy
    step_info.json
  ...
```

OCT volumes are signed `int8` arrays in row–column–depth order,
with shape `[256, 256, 768]`; outer singleton dimensions are also accepted.
Each `step_info.json` contains the tooltip transform
(`tooltip_transform_4x4`) and action command (`inputs_6dof_mm_deg`).
Keep the target metadata and displacement files with the trajectory.

## Training

Run the following commands from the code directory:

```bash
python prepare_data.py \
  --data-root /path/to/demonstrations \
  --data-profile configs/data/official_3dof.json \
  --config configs/phantom_point_residual_bc.json \
  --output prepared/point

python train_policy.py \
  --prepared prepared/point \
  --config configs/phantom_point_residual_bc.json \
  --output outputs/point

python evaluate_policy.py \
  --prepared prepared/point \
  --checkpoint outputs/point/checkpoint_best.pt \
  --output outputs/point/validation.json
```

Model configurations are in `configs/`. For directional tasks, use
`configs/data/official_6dof.json` and a model configuration ending in `_tilt`.

Data preparation creates an approximately 80/20 trajectory-level split
within each target condition, using seed 42. Prepared data can be reused
across Explicit/Diffusion and residual/non-residual configurations of
the same material, task, and representation.

For the paper's Full-volume caches, use CPU preparation for depth tasks
and add `--device cuda` for directional tasks.

Training saves `checkpoint_best.pt` using validation-selected EMA weights
and `checkpoint_last.pt` for resuming. Existing runs resume automatically;
use `--fresh` to start over or `--epochs` to change the epoch limit.

## Inference

The `checkpoints/` directory contains the 28 paper best EMA checkpoints.
Names identify the material, representation, residual setting, and policy.
`bc` denotes Explicit; `_tilt` denotes the 6-DoF directional task.
Checkpoints without `_tilt` are for 3-DoF depth control.

To predict actions from a saved trajectory:

```bash
python run_policy.py \
  --trajectory /path/to/condition/traj_1 \
  --checkpoint checkpoints/phantom_point_residual_bc.pt \
  --output predictions.json
```

Demonstration folders and deployment folders containing `rollout_step_N`
are supported. Add `--seed` to set the inference seed.

For incoming OCT observations:

```python
from oct_policy import load_policy

policy = load_policy(
    "checkpoints/phantom_point_residual_bc.pt",
    device="cuda",
)
policy.reset("predeform/target_volume.npy", initial_tooltip_pose)

prediction = policy.step("current/ascans.npy", current_tooltip_pose)
action = prediction["action_6dof_mm_deg"]
```

Tooltip poses are 4×4 transforms with translation in metres.
Returned actions contain local translation increments in millimetres
and extrinsic Euler-xyz rotation increments in degrees.

Each prediction contains eight actions; `action_6dof_mm_deg` is the first.
Robot communication is not included.
