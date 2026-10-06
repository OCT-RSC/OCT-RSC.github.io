# OCT-RSC project website

Project page for **Robotic Optical Coherence Tomography-Guided Geometric
Modeling of Soft Tissue: 3D Representations for Data-Driven Closed-Loop
Shape Control**.

**Website:** https://oct-rsc.github.io/

## Local preview

```sh
python3 -m http.server 8769 --bind 127.0.0.1
```

Open http://127.0.0.1:8769/. Use an HTTP server rather than opening
`index.html` directly, because the viewer loads binary arrays with `fetch`.
There is no build step, backend, API key, analytics, or external JavaScript
dependency.

## Layout

| File or directory     | Purpose                                                         |
| --------------------- | --------------------------------------------------------------- |
| `index.html`          | Paper information, figures, videos, and resource links          |
| `styles.css`          | Responsive page layout                                          |
| `app.js`              | Experiment tabs, results table, and copy controls               |
| `volume-viewer.js`    | Interactive WebGL 2 volume and point-cloud viewer               |
| `assets/`             | Published videos, viewer arrays, figures, and affiliation logos |
| `downloads/`          | Manuscript and research-code package                           |
| `tools/check_site.py` | Link, checksum, input-array, and media-coverage checks          |
| `MEDIA.md`            | Visualization scope and asset provenance                        |

The website and the research-policy package are separate. The research code
is distributed as a downloadable source archive; this repository
does not contain robot credentials, raw acquisition directories, or trained
checkpoints.

## Checks

```sh
python3 tools/check_site.py
node --check app.js
node --check volume-viewer.js
```

Keep the manuscript, the results in `app.js`, and the page text synchronized.
Preserve the recorded-step pairing and media verification records when
replacing videos. Visualization assets are not substitutes for the actual
policy input tensors; see `MEDIA.md`.

## GitHub Pages

In **Settings → Pages**, choose **Deploy from a branch**, branch **main**,
folder **/ (root)**, and save. The `.nojekyll` file disables Jekyll processing:
the site is served as ordinary static HTML, CSS, JavaScript, and assets.

No repository-wide software license has been selected. Paper, research code,
and institutional logos may have separate distribution terms.
