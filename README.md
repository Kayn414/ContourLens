# ContourLens
<img width="1919" height="1026" alt="image" src="https://github.com/user-attachments/assets/51ff63bc-ac5e-4784-9506-1950c85b81b1" />

## Project overview

**ContourLens** is a radiotherapy imaging workbench for evaluating organ-at-risk
(OAR) segmentation models:

- **Python (`source/`)** reads DICOM / NIfTI / NRRD (pydicom, SimpleITK,
  rt-utils), prepares nnU-Net datasets, trains and runs nnU-Net v2 locally or on
  [Modal](https://modal.com) cloud GPUs, and scores predictions (Dice, HD95).
- **C++ GUI (`gui/`)**, a Windows desktop viewer (Win32 + DirectX 11 + Dear
  ImGui docking + ImPlot): axial/coronal/sagittal MPR plus a GPU-raycast 3D
  view, ground-truth and prediction overlays, drag-and-drop loading, one-click
  inference with live progress, per-structure metrics, error maps, DVH, export
  to RTSTRUCT, and batch evaluation. All file-format and ML work is delegated
  to the Python tools, which the GUI runs as background jobs.

The bundled model is `Dataset501_HanSeg` (HaN-Seg head-and-neck OARs:
brainstem, optic chiasm, optic nerves, lenses, eyeballs, spinal cord), trained
with `nnUNetTrainer_500epochs`, folds 0-3, `3d_fullres`.

## Quick start

Requirements: Windows 10/11 (for the GUI), an NVIDIA GPU for local inference
(optional; Modal works without one), [uv](https://docs.astral.sh/uv/), CMake >=
3.21 and Visual Studio 2022 (C++ workload). `uv` must be on `PATH`, because the
GUI launches every Python tool as `uv run python -m source.<module>`.

```bash
uv sync                                          # Python env (Python >=3.12,<3.14) from uv.lock
cmake --preset windows-x64                       # configure the GUI (VS 2022, output in gui/build)
cmake --build gui/build --config Release         # build; close the GUI first (a running exe can't be relinked: LNK1104)
gui/build/Release/ContourLens.exe                # opens case "phantom"; or pass a case name, e.g. user/case_09_0000
```

`data/` is gitignored, so a fresh clone has no images, cases or model weights:
- Open any CT by dragging it onto the GUI (see "Loading data" below); that
  creates a case under `data/user/`.
- Test fixtures come from the [SlicerRt test data](http://www.slicerrt.org)
  (vendor TPS exports with CT/RTSTRUCT/RTDOSE/RTPLAN) and
  [HaN-Seg](https://doi.org/10.5281/zenodo.7442914). Place them under
  `data/SlicerRtData/` and `data/hanseg/raw/HaN-Seg.zip`.
- Model weights live under `data/nnUNet_results/` locally, or on the Modal
  volume `nnunet-results-volume`.

Dev commands:

```bash
uv run pytest                 # tests in tests/ (synthetic data only; no data/ needed)
uv run ruff check .           # lint
uv run ty check               # type-check (uses `ty`, not mypy)
```

## Repository map

| Path | What it is |
|---|---|
| `gui/src/main.cpp` | Window, render loop, all panels/menus/popups. State + helper lambdas at the top of `main()`: `LoadCase`, `LoadLabelSlot`, `RecomputeMetrics`/`RecomputeDVH`, `Start*Job`, `OnJobFinished`, `DrawJobStatus`, `DrawMetricsTable`, `DrawBatchResults` |
| `gui/src/inference.{h,cpp}` | `BackgroundJob`: runs a Python module as a child process (pipe for output, Job Object for Cancel), parses the progress protocol. Despite the name, it runs every job kind |
| `gui/src/structure_metrics.{h,cpp}` | Structure name matching + aliases, colors, per-structure Dice/volumes, error volume, jump-to-structure |
| `gui/src/volume.{h,cpp}` | Loaders for the raw+JSON volumes (`Volume` int16 HU, `LabelVolume` uint8, `DoseVolume` float32 Gy) |
| `gui/src/raycaster.{h,cpp}` | GPU volume raymarcher for the 3D panel (label tint for up to 16 structures) |
| `gui/src/dvh.{h,cpp}` | Cumulative DVH curves (+ Dmean/Dmax), synthetic dose fallback |
| `gui/src/inference_config.{h,cpp}` | GUI editor for `configs/<case>/inference_config.json` |
| `source/gui_load.py` | Dropped-file loader: `ct`, `masks`, `dose` subcommands |
| `source/export_gui_volume.py` | Raw+JSON writer, format docs, `GuiGrid` (resampling onto a case's CT grid), DICOM series resolution, orientation |
| `source/gui_progress.py` | The `@@progress`/`@@error`/`@@notice` protocol + tqdm parsing |
| `source/run_inference_for_gui.py` | "Run Inference": prediction for a case, exported as its overlay |
| `source/inference_backends.py` | `local` (nnUNetv2_predict) and `modal` backends; Modal auto-deploy |
| `source/inference_config.py` | Config dataclasses/schema; reads model label names from `dataset.json` |
| `source/gui_metrics.py` | HD95 for the Metrics table |
| `source/gui_export.py` | Metrics > Export: NIfTI + `dataset.json`, RTSTRUCT for DICOM cases |
| `source/batch_evaluate.py` | Batch evaluation over one fold's validation cases |
| `source/structure_names.py` | Python twin of the C++ name matching (`normalize`, `match_key`) |
| `source/modal_utils.py`, `source/modal_nnunet.py` | Modal app/image/volumes; preprocessing, training, `predict_nifti_bytes` |
| `source/data/hanseg.py`, `source/data/nnunet_dataset.py` | HaN-Seg extraction/conversion, nnU-Net `Dataset501_HanSeg` builder, `nnUNet_*` path resolution |
| `source/data/phantom.py`, `source/predict.py`, `source/evaluate.py` | Original phantom-only pipeline (GT masks from the eclipse phantom RTSTRUCT, predict, score) |
| `source/main.py` | The original exploratory DICOM script |
| `scripts/config.sh`, `scripts/train_nnunet.sh` | Local nnU-Net env vars and training loop |
| `configs/` | Per-case inference configs (`configs/user/` is gitignored) and `structure_aliases.json` |
| `tests/` | pytest suite (progress parsing, loaders, metrics, RTSTRUCT round trip, batch helpers) |

## Data layout and file formats

A **case** is a folder name relative to `data/` (`phantom`,
`hanseg_cases/case_03`, `user/<name>`). The GUI reads only raw+JSON files from
its `gui_export/`:

```
data/<case>/gui_export/
  ct.bin/.json          int16 HU          (required)
  masks.bin/.json       uint8 label ids   ground truth (optional)
  prediction.bin/.json  uint8 label ids   model prediction (optional)
  dose.bin/.json        float32 Gy        (optional; else the DVH uses a synthetic dose)
  metrics_hd95.json     {"pairs": {"<gt_id>:<pred_id>": mm | null}}; ignored if older than masks/prediction
data/<case>/exports/<YYYYmmdd_HHMMSS>/   Metrics > Export output
data/batch/<timestamp>_fold<K>/          Batch Evaluation output (results.json/.csv, predictions/)
```

- **Sidecar JSON:** `shape` is `(nz, ny, nx)`; `spacing`/`origin` are ITK
  `(x, y, z)` mm in **LPS**; `direction` is a 3x3 row-major matrix; label
  volumes carry `labels`, with `labels[id - 1]` = structure name.
  `source/export_gui_volume.py`'s docstring is the authoritative spec.
- **Orientation:** CT exports are reoriented to identity-direction LPS when the
  source is axis-aligned. The MPR slicing ignores `direction`, so oblique CTs
  display unrotated.
- **Resampling:** masks, predictions and dose are always resampled onto the
  case's `ct.json` grid in physical space (nearest-neighbour for labels, linear
  for dose), so they only need to share the CT's physical space.

**`configs/<case>/inference_config.json`** is created or edited by the GUI's
Inference Config section:

```json
{
  "backend": "local | modal",
  "source_ct": "NIfTI/NRRD file or DICOM series dir (repo-relative or absolute)",
  "model": {"dataset_id": "501", "configuration": "3d_fullres", "trainer": "nnUNetTrainer_500epochs",
            "plans": "nnUNetPlans", "folds": [0, 1, 2, 3], "results_dir": null},
  "modal": {"app_name": "nnunet-training", "function_name": "predict_nifti_bytes"}
}
```

**`configs/structure_aliases.json`**: `{"aliases": {"OpticNrv_L": "optic_nerve_l", ...}}`.
Two structure names match when their normalized forms (lowercase
alphanumerics) are equal after mapping through this table. It's global (all
cases) and user-editable from the GUI, so it may be empty.

## Using the GUI

### Layout
The **Controls** sidebar is on the left, with a 2x2 grid of **Axial |
Coronal / Sagittal | 3D**. View > Reset Layout restores it. The top of Controls
shows the current or last background job: progress bar, stage message,
**Cancel**, **Log**, notices (blue), and **Open folder** for exports and
batches. Only one job runs at a time.

### Mouse
- **Slice panels:**
  - Left-click/drag moves the shared crosshair; the other two views follow.
  - The wheel zooms toward the mouse (up to 8x); right-drag pans, middle-click resets.
  - Top-left readout: voxel `(i, j, k)`, HU, and the GT/prediction structure at the cursor with its Dice. Green means they agree, red means they disagree.
- **3D panel:**
  - Left-drag orbits, right-drag pans, the wheel zooms.
  - Double-click a slice plane to lock rotation to its axis; Space resets.
  - A/P/L/R/S/I buttons jump to standard views.

### Keyboard
Shortcuts are disabled while a text field has focus or a popup/menu is open.

| Key | Action |
|---|---|
| Q / E | Cycle the focused slice panel |
| Shift+S / Shift+W / Shift+L | Choose which parameter Shift+A/D adjusts: Slice / Width / Level |
| Shift+A / Shift+D | Decrease / increase that parameter by the sidebar's step |
| G | Toggle window/level between shared (all panels) and per-panel |
| Space (over 3D) | Reset the 3D camera |

### Menus
- **File:** Recent cases (every case under `data/` with a `ct.json`), Exit.
- **View:** Structure overlays, Prediction overlay, Error map, DVH, Batch Evaluation, Reset Layout.

### Loading data (drag and drop)
Drop files or folders onto the window. A **Load as** popup preselects a
sensible option.

| Dropped | Load as |
|---|---|
| DICOM folder, one `.dcm` slice, `.nii/.nii.gz`, `.nrrd/.nhdr`, `.mha/.mhd` | **CT**: creates and switches to case `user/<name>`. A DICOM folder that also contains a matching RTSTRUCT gets it loaded as ground truth |
| Multi-label NIfTI/NRRD | **Ground truth** or **Prediction**. Names come from a `dataset.json`/`labels.json` beside it or up to 2 folders above; otherwise `label_<id>` |
| `.seg.nrrd` files or a folder of them (HaN-Seg / 3D Slicer) | **Ground truth** or **Prediction** (names from Slicer metadata or the filename) |
| RTSTRUCT `.dcm` | **Ground truth** or **Prediction**. Needs the referenced CT series next to it, or the case's CT loaded from DICOM |
| RTDOSE `.dcm` | **Dose**: feeds the DVH |

Masks and dose require a CT to be loaded first. The same conversions are
available from the command line (`source.gui_load`, below).

### Controls sections
- **Window/level:** Slice / Width / Level sliders (highlighted when targeted by Shift+A/D) and presets (Soft, Lung, Bone, Brain).
- **Structures (ground truth):** per-structure visibility and overlay opacity.
- **Inference Config:** Source CT, model (dataset id, configuration, trainer, plans, results dir) and backend tab (**Local** or **Cloud (Modal)**). **Save Config** / **Reload Config** read and write `configs/<case>/inference_config.json`.
  - *Folds:* clicking fold *k* selects folds 0..k; Ctrl+click toggles a single fold.
- **Prediction:** **Run Inference** (saves the config first), then per-structure visibility and opacity for the prediction overlay.
- **Metrics** (auto-expands after inference): mean Dice and HD95, then a sortable per-structure table (Dice, HD95, GT/pred volume in cc, volume difference), worst Dice first.
  - Click a row to jump the crosshair to where that structure disagrees most.
  - **Error map** tints missed voxels orange and extra voxels cyan, for structures present on both sides.
  - **Compute HD95** runs automatically after inference or a mask load; the button re-runs it.
  - **Export** writes metrics CSV/JSON + `prediction.nii.gz`/`dataset.json`, plus `prediction_rtstruct.dcm` when the CT came from DICOM.
  - **Pair unmatched structures** saves an alias for differently named GT/prediction structures; **Structure aliases** lists them, with remove buttons.

### DVH window (View > DVH)
Cumulative DVHs from the case's real dose if one is loaded (otherwise a
clearly-labelled synthetic Gaussian dose). Ground-truth curves are drawn thick
and prediction curves thin (toggleable), with a Dmean/Dmax GT-vs-prediction
table for matched structures.

### Batch Evaluation window (View > Batch Evaluation)
Runs the current case's Inference Config model on **one fold's validation
cases** (from `nnUNet_preprocessed/Dataset<id>_*/splits_final.json`, CTs/labels
from `nnUNet_raw/.../imagesTr|labelsTr`), using only that fold's weights, so
every score is on unseen data.
- **Parallelism:** on Modal, up to 8 cloud GPUs at once (default 2); on Local,
  capped at the NVIDIA GPUs detected, with each case pinned to its own GPU.
- **Results:** per-structure mean ± std Dice/HD95 plus per-case results, saved
  under `data/batch/`.

## Inference backends

- **local:** runs `nnUNetv2_predict` on this machine. `nnUNet_raw/preprocessed/results`
  default to `data/nnUNet_*` unless set in the environment (see `scripts/config.sh`).
  Needs the trained model under `nnUNet_results` (or the config's `results_dir`).
- **modal:** stages the CT as NIfTI bytes and calls the deployed function
  `nnunet-training::predict_nifti_bytes`. The model must be on the Modal volume
  `nnunet-results-volume`; raw and preprocessed data live on `nnunet-raw-volume`
  and `nnunet-preprocessed`.
  - *Auto-deploy:* if the app isn't deployed, `ensure_modal_deployed` runs `modal deploy source/modal_nnunet.py` first and reports it as a GUI notice.
  - *Why deploy:* an app started with `modal run` stops when the run ends and isn't visible to `Function.from_name`.
  - *Progress:* cloud runs show only stages, since remote tqdm output isn't streamed.
- Training on Modal: `uv run modal run --detach source/modal_nnunet.py::main --step train_production`
  (add `--continue-training` to resume); see `main()` in that file for the other steps.

## Command-line tools

All run from the repo root. The GUI calls the first five itself.

```bash
uv run python -m source.gui_load ct <file|dicom_dir|slice.dcm> --case user/<name>
uv run python -m source.gui_load masks <file(s)|folder> --case <case> --slot gt|prediction
uv run python -m source.gui_load dose <RTDOSE.dcm> --case <case>
uv run python -m source.run_inference_for_gui <case>
uv run python -m source.gui_metrics hd95 --case <case> --pairs 1:1,2:2
uv run python -m source.gui_export --case <case> --out <dir>
uv run python -m source.batch_evaluate --case <case> --fold K --out <dir> [--parallel N] [--limit M]
uv run python -m source.export_gui_volume --help      # lower-level exporter (CT / --export-masks / --dose / --prediction)
uv run python -m source.data.hanseg                   # extract + convert data/hanseg/raw/HaN-Seg.zip
uv run python -m source.data.nnunet_dataset           # build nnUNet_raw/Dataset501_HanSeg
```

## Architecture notes for contributors

**GUI to Python job protocol.** The GUI starts `uv run python -m <module>` with
stdout+stderr on a pipe and reads it every frame (also while minimized, so the
child never blocks):
- `@@progress {"stage", "current", "total", "fraction", "message"}` drives the progress bar (`fraction: null` = indeterminate).
- `@@error {"message"}` is a readable failure reason; `@@notice {"message"}` is persistent info.
- Every other line goes to the job's Log and the GUI console.

A Python entry point should call `configure_stdio()`, emit progress, and on
failure `emit_error(...)` then exit non-zero. `run_with_progress` turns a child
process's tqdm bars into progress events.

**Adding a GUI job:**
1. A Python entry point, as above.
2. A `JobKind` in `gui/src/inference.h`.
3. A `Start...Job` lambda in `main.cpp` that sets `job.kind/title/case_name` and calls `StartJob`.
4. A case in `OnJobFinished` that reloads whatever the job wrote.

Never mutate `structure_metrics` or the label volumes while the Metrics table is
iterating them; set a flag (like `aliases_changed`) and apply it after the
Controls window.

**Structure matching exists twice and must stay in sync:**
`StructureMatchKey`/`NormalizeStructureName` in `gui/src/structure_metrics.cpp`
and `match_key`/`normalize` in `source/structure_names.py`. HD95 jobs receive
the GUI's own id pairs instead of re-matching. When nothing matches and one side
only has generic `label_<id>` names, the GUI pairs by id.

**Coordinates:** volumes are indexed `(k, j, i)` = `(z, y, x)`, and voxel
`(i, j, k)` sits at byte `((k*ny + j)*nx + i) * itemsize`. Coronal/sagittal
textures have row 0 = inferior and are displayed flipped (superior on top); see
`SliceView`/`SliceViewUV`/`PanelClicked`.

**Modifier keys:** read them with `GetKeyState(VK_SHIFT/VK_CONTROL)`, not
`io.KeyShift/KeyCtrl`, which go stale when the key was pressed before the
window got focus.

**DICOM RT concepts:**
- A CT series is the set of files sharing `SeriesInstanceUID`; sort slices by position, never by filename.
- Apply `RescaleSlope`/`RescaleIntercept` for HU (SimpleITK does this).
- RTSTRUCT/RTDOSE reference the CT via Study/Series/FrameOfReference UIDs.
- RTDOSE needs `DoseGridScaling`.
- `rt-utils` treats every pixel-data file in a directory as part of the series, so stage only the CT slices (`gui_load.stage_dicom_series`).

## Gotchas and known limits

- **Local RAM:** 4-fold local nnU-Net inference has exhausted 32 GB of RAM in its segmentation-export step, even on the small phantom. Prefer Modal, or fewer folds.
- **Honest scores:** a case scores honestly only on folds that didn't train on it. Check `splits_final.json` (e.g. HaN-Seg case_09 is in fold 0's validation set only); Batch Evaluation handles this for you.
- **Existing phantom prediction:** `data/phantom/gui_export/prediction.*` from the earlier pipeline is nearly empty (Dice 0 everywhere). That's the data, not a metrics bug.
- **3D tint limit:** the 3D view tints at most 16 labels; label volumes hold at most 255 structures.
- **RTSTRUCT export** needs a CT loaded from a DICOM series. NIfTI/NRRD cases export NIfTI only.
- **Drag and drop** from Explorer doesn't reach the GUI if it runs elevated (Windows UIPI).
- **Tests:** don't read `configs/structure_aliases.json` or other user-editable configs in tests; pass explicit alias tables or temp files.
- **Rebuilding:** close `ContourLens.exe` first, or linking fails with LNK1104.
