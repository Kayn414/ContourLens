#pragma once

// Dose-volume histogram support for the (toggleable) "DVH" window.
//
// There is no RTDOSE for the phantom case (only CT + OAR masks), so
// GenerateSyntheticDose() fabricates a Gaussian dose blob purely to exercise
// the DVH plot end-to-end. It is clearly labeled as synthetic wherever it's
// drawn -- swap it for a real loaded dose (same grid as the CT) once one
// exists, ComputeDVHCurves() doesn't care where `dose` came from.

#include "volume.h"
#include <string>
#include <vector>

struct DVHCurve
{
    std::string name;
    std::vector<float> dose_gy;     // bin edges, ascending
    std::vector<float> volume_pct;  // % of the structure's volume receiving >= dose_gy[bin]
};

// One float per voxel (same (k, j, i) order as Volume/LabelVolume), Gy.
std::vector<float> GenerateSyntheticDose(const Volume& volume, float peak_dose_gy = 70.0f);

// One cumulative DVH curve per label in `labels` (labels.labels[i] -> curves[i]).
// `dose` must be sized labels.nx*ny*nz and share its voxel grid.
std::vector<DVHCurve> ComputeDVHCurves(const std::vector<float>& dose, const LabelVolume& labels, int num_bins = 64);
