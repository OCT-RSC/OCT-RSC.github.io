# Media and visualization

## Closed-loop videos

The three experiment tabs show one selected deployment for each target:

| Tab                    | Target conditions                               | Trajectories |
| ---------------------- | ----------------------------------------------- | -----------: |
| Phantom depth          | 2.0–5.0 mm in 0.5-mm increments                 |            7 |
| Phantom tilt direction | 0–180° in 22.5° increments                      |            9 |
| Ex vivo                | Five depths, 2.0–4.0 mm; five directions, 0–90° |           10 |

Each video includes every saved OCT state, steps 0–12, held for one second
per state. These are recorded states, not interpolated motion or real-time
execution. RGB frames accompany their corresponding post-action state; no
initial RGB frame is invented. Each cell shows the OCT target overlay, RGB
context, and a separate interaction-detail crop. The OCT render is not cropped.

The exports contain 26 trajectories, 338 OCT states, and 312 post-action RGB
frames. They illustrate target coverage, not all configurations or deployment
repeats. Coverage and video hashes are recorded in
`assets/media-verification.json`; `assets/demos.json` supplies the page links.

## Interactive representations

The Full-volume and Tissue-masked views ray-cast the saved normalized OCT
intensities over the complete 256 × 256 × 256 saved cube, including depth
indices 0–255. There is no additional display-depth crop. Float16 transport retains the
stored source intensities. Color and opacity follow the paper illustration's
reproduction package. The display-mask extraction is applied to the
complete cube. Narrow filaments are removed by retaining the largest
two-voxel-eroded body and restoring its boundary within the original mask;
this avoids a flat depth cutoff. The resulting 618,637-voxel mask is refined
for display; it is **not** the sparse policy input mask. OCT intensities are
unchanged and no synthetic background noise is added.
Bright non-tissue returns are suppressed by the background-opacity mapping,
while measured low-intensity air remains visible throughout the cube.

The Point-cloud view displays the saved 1,024 model samples. All three views
share one coordinate mapping, camera direction, and zoom. Coordinates are
normalized display units, not millimeters. Switching modes does not independently
fit or recenter the geometry. Browser WebGL rendering is not claimed to be
pixel-identical to the reference CPU rendering.

Array shapes, transfer-function settings, and asset hashes are recorded in
`assets/paper-render-style.json`.

## Other assets

- The manuscript and method figure come from the project paper.
- `assets/logo-sources.json` records the official IGMR Lab and Michigan
  Robotics wordmark sources and their checksums.
- The prepared training example contains two demonstrations, not the complete
  dataset. The smoke command checks input/output contracts with an untrained
  model; it does not evaluate a trained checkpoint.
- `downloads/verification.json` reports a separate full-data point-cloud
  training-pipeline check. Its action-prediction errors are not the deployment
  Shape RMSE values reported on the page.

No previous video drafts, local transfer logs, remote-machine credentials,
or raw OCT acquisition folders are included in this repository.
