# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

`dicom_rt` is an early-stage Python project for reading and visualizing DICOM
radiotherapy (RT) data with `pydicom`/`numpy`/`matplotlib` (and `torch` for
future tensor work). The codebase is currently a single exploratory script;
there is no package structure, CLI, or test suite yet — expect to build
conventions as you go rather than follow established ones.

## Environment and commands

Dependencies and the venv are managed with `uv` (see `pyproject.toml` /
`uv.lock`). Project package name is `source`, importable as `import source`.

```bash
uv sync                       # install/update the environment from uv.lock
uv run python source/main.py  # run the main exploratory script
uv run pytest                 # run tests (none exist yet)
uv run ruff check .           # lint
uv run ty check               # type-check (uses `ty`, not mypy)
```

Requires Python >=3.12,<3.14 (note: the active venv here may be 3.11 —
confirm with `uv run python --version` if version-sensitive behavior matters).

## Data

- `data/*.dcm` — a single sample DICOM CT slice used by `source/main.py`.
- `data/SlicerRtData/` — the [SlicerRt](http://www.slicerrt.org) test dataset
  collection (`Links.txt` documents external sources, e.g. the POPI 4D-CT
  dataset from Creatis). Each subdirectory is a vendor/TPS export
  (Eclipse, Pinnacle, XiO, CERR, etc.) containing a full RT study: CT slices
  plus associated RTSTRUCT (contours), RTDOSE, RTPLAN, RTIMAGE, and RTRECORD
  files, distinguished by DICOM `Modality` in the filename. Use these as
  realistic, varied test fixtures when working with more than a single CT
  series — they exercise different vendor quirks in DICOM tag usage.

## Code architecture

`source/main.py` currently does everything inline (no functions/modules to
navigate beyond `inspect_dicom`): reads a DICOM CT file with `pydicom`,
prints key identifying tags (Modality, SOPClassUID, StudyInstanceUID,
SeriesInstanceUID), stacks per-slice pixel arrays into a 3D volume ordered by
`ImagePositionPatient[2]`, converts raw pixel values to Hounsfield Units via
`RescaleSlope`/`RescaleIntercept`, and saves a windowed grayscale slice
render to `ct_slice.png`.

Key DICOM RT domain concepts to keep in mind when extending this:
- A CT **series** is a set of DICOM files sharing `SeriesInstanceUID`, one
  per axial slice; slices must be sorted by `ImagePositionPatient[2]` (or
  `InstanceNumber`) before stacking into a volume — file/glob order is not
  reliable.
- Pixel data is raw scanner units; always apply `RescaleSlope` /
  `RescaleIntercept` to get Hounsfield Units (HU) before windowing/display.
- RTSTRUCT, RTDOSE, and RTPLAN files reference a CT series via
  `StudyInstanceUID`/`SeriesInstanceUID` (and `ReferencedSOPInstanceUID`
  chains) rather than containing image data themselves — associate them with
  a CT volume through those UIDs, not by filename/directory alone.
