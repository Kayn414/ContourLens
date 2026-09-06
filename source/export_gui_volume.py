"""Dump a NIfTI volume as raw little-endian binary + a JSON sidecar, so the
C++ GUI (gui/) can load it without linking a NIfTI library (see the GUI
planning doc's "Path A" loader decision: raw+JSON for milestone 1, migrate to
nifti_clib later if arbitrary .nii.gz files need to be opened directly).

Conventions (write these once, don't rediscover them in C++):
- The array is dumped exactly as SimpleITK returns it: index order (k, j, i)
  i.e. (z, y, x), C-contiguous. Byte offset of voxel (i, j, k) is
  ((k * ny + j) * nx + i) * itemsize.
- spacing / origin / direction are in ITK's (x, y, z) axis order (NOT the
  array's (z, y, x) order) and describe physical space in **LPS**
  millimeters (SimpleITK/DICOM convention -- see the GUI planning doc's
  coordinate-system section).
- world = origin + direction_matrix @ (spacing * (i, j, k)), where
  direction_matrix is the 3x3 from `direction`, row-major.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk

DEFAULT_CT = Path("data/phantom/nnunet_input/phantom_ent_0000.nii.gz")
DEFAULT_OUT_DIR = Path("data/phantom/gui_export")


def export_volume(nifti_path: Path, out_dir: Path, name: str) -> None:
    image = sitk.ReadImage(str(nifti_path))
    array = sitk.GetArrayFromImage(image)  # (z, y, x)

    if not np.issubdtype(array.dtype, np.integer):
        raise ValueError(f"{nifti_path}: expected an integer-valued volume, got {array.dtype}")
    if array.min() < np.iinfo(np.int16).min or array.max() > np.iinfo(np.int16).max:
        raise ValueError(f"{nifti_path}: value range [{array.min()}, {array.max()}] doesn't fit int16")
    array = array.astype("<i2")  # little-endian int16

    out_dir.mkdir(parents=True, exist_ok=True)
    bin_path = out_dir / f"{name}.bin"
    json_path = out_dir / f"{name}.json"

    array.tofile(bin_path)
    json_path.write_text(json.dumps({
        "shape": list(array.shape),  # (nz, ny, nx)
        "dtype": "int16",
        "spacing": list(image.GetSpacing()),  # (sx, sy, sz) mm
        "origin": list(image.GetOrigin()),    # (ox, oy, oz) mm, LPS
        "direction": list(image.GetDirection()),  # 3x3 row-major
    }, indent=2))

    print(f"{nifti_path} -> {bin_path} ({array.nbytes:,} bytes), {json_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nifti", type=Path, default=DEFAULT_CT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--name", default="ct")
    args = parser.parse_args()

    export_volume(args.nifti, args.out_dir, args.name)


if __name__ == "__main__":
    main()
