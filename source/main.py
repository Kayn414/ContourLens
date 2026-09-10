import pydicom
from pathlib import Path
from collections import defaultdict
import glob
import numpy as np
import matplotlib.pyplot as plt

file_name = "HITTest1Fld27.CT.Spezial_01HIT_.3.1.2012.03.23.10.48.59.159.25131122"
ds = pydicom.dcmread("data/HITTest1Fld27.CT.Spezial_01HIT_.3.1.2012.03.23.10.48.59.159.25131122.dcm")


def inspect_dicom(path):
    ds = pydicom.dcmread(path)

    print("Modality:", getattr(ds, "Modality", "N/A"))
    print("SOP Class:", getattr(ds, "SOPClassUID", "N/A"))
    print("Study UID:", getattr(ds, "StudyInstanceUID", "N/A"))
    print("Series UID:", getattr(ds, "SeriesInstanceUID", "N/A"))

    return ds


ds = inspect_dicom(f"data/{file_name}.dcm") # CT


files = glob.glob(f"data/{file_name}.dcm")

slices = [
    pydicom.dcmread(f)
    for f in files
]
print(len(slices))

slices.sort(
    key=lambda x: float(x.ImagePositionPatient[2])
)

ct_volume = np.stack(
    [s.pixel_array for s in slices]
)

print(ct_volume.shape)


ct = slices[0]

hu = (
    ct.pixel_array * ct.RescaleSlope
    + ct.RescaleIntercept
)

plt.imshow(
    hu,
    cmap="gray",
    vmin=-1000,
    vmax=1000
)

plt.colorbar(label="HU")
plt.imshow(hu, cmap="gray")
plt.colorbar(label="HU")
plt.savefig("ct_slice.png")