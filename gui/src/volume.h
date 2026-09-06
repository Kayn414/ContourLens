#pragma once

// Reads the raw+JSON sidecar written by source/export_gui_volume.py.
// See that file's docstring for the on-disk conventions (index order,
// spacing/origin/direction axis order, LPS).

#include <cstdint>
#include <string>
#include <vector>

struct Volume
{
    std::vector<int16_t> data; // C-order, index (k, j, i) == (z, y, x) 
    int nx = 0, ny = 0, nz = 0;
    double spacing[3] = { 1.0, 1.0, 1.0 };   // (sx, sy, sz) mm
    double origin[3] = { 0.0, 0.0, 0.0 };    // (ox, oy, oz) mm, LPS
    double direction[9] = { 1,0,0, 0,1,0, 0,0,1 }; // 3x3 row-major

    int16_t At(int i, int j, int k) const
    {
        return data[(size_t)(k * ny + j) * nx + i];
    }
};

// Loads "<base_path>.bin" + "<base_path>.json". Returns false (and logs to
// stderr) on failure; `out` is left untouched on failure.
bool LoadVolume(const std::string& base_path, Volume& out);

// A combined multi-label mask volume (source/export_gui_volume.py
// --export-masks): one byte per voxel, 0 = background, id = 1-based index
// into `labels`. Same (k, j, i) index order as Volume; assumed to share its
// CT's grid (caller should check nx/ny/nz match before indexing both by the
// same (i, j, k)).
struct LabelVolume
{
    std::vector<uint8_t> data;
    int nx = 0, ny = 0, nz = 0;
    std::vector<std::string> labels; // labels[id - 1] == structure name

    uint8_t At(int i, int j, int k) const
    {
        return data[(size_t)(k * ny + j) * nx + i];
    }
};

bool LoadLabelVolume(const std::string& base_path, LabelVolume& out);

// A dose volume in Gy (source/export_gui_volume.py --dose): float32, already
// resampled onto the reference CT's grid. Same (k, j, i) index order.
struct DoseVolume
{
    std::vector<float> data;
    int nx = 0, ny = 0, nz = 0;

    float At(int i, int j, int k) const
    {
        return data[(size_t)(k * ny + j) * nx + i];
    }
};

bool LoadDoseVolume(const std::string& base_path, DoseVolume& out);
