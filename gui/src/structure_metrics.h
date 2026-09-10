#pragma once

// Structure identity and comparison across label volumes (ground truth vs.
// prediction): name matching (with the global alias table), colors, the
// Metrics table's per-structure Dice/volumes, the error map, and
// jump-to-structure. source/structure_names.py mirrors NormalizeStructureName
// and StructureMatchKey exactly, so the Python evaluation tools pair
// structures the same way the GUI does.

#include "volume.h"
#include <array>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

// Lowercase alphanumerics only, so "Optic Nerve-Lt", "optic_nerve_lt" and
// "OpticNerveLt" all compare equal.
std::string NormalizeStructureName(const std::string& name);

// configs/structure_aliases.json under `repo_dir`: {"aliases": {"OpticNrv_L": "optic_nerve_l", ...}}.
// Missing file = no aliases. Call again after the file changes.
void LoadStructureAliases(const std::string& repo_dir);

// Adds/replaces (or removes) one alias in the file, keeping its other
// contents, then reloads. Returns false (logged) if the file is malformed or
// can't be written -- it is never overwritten in that case.
bool SaveStructureAlias(const std::string& repo_dir, const std::string& source, const std::string& target);
bool RemoveStructureAlias(const std::string& repo_dir, const std::string& source);

// The alias entries exactly as written in the file, for the GUI's alias list.
const std::vector<std::pair<std::string, std::string>>& StructureAliasEntries();

// What two names must share to be the same structure: the normalized name,
// mapped through the alias table when it has an entry.
std::string StructureMatchKey(const std::string& name);

// Canonical structures keep fixed colors; any other name gets a stable color
// hashed from its match key, so ground truth and prediction agree.
std::array<uint8_t, 3> ColorForStructureName(const std::string& name);

// Per-label-id (index = id - 1) colors for one label volume, so UploadSlice's
// hot loop does array lookups instead of string compares.
std::vector<std::array<uint8_t, 3>> BuildLabelColors(const std::vector<std::string>& names);

// One row of the Metrics table / crosshair readout: a structure in the
// ground truth and/or the prediction, matched across the two.
struct StructureMetric
{
    std::string name;
    int gt_id = 0, pred_id = 0; // label id on each side; 0 = that side has no such structure
    size_t gt_voxels = 0, pred_voxels = 0;
    double gt_cc = 0.0, pred_cc = 0.0;
    float dice = std::numeric_limits<float>::quiet_NaN();    // NaN unless both sides have the structure
    float hd95_mm = std::numeric_limits<float>::quiet_NaN(); // filled from source/gui_metrics.py's results, when available
};

struct StructureMetrics
{
    std::vector<StructureMetric> rows;
    std::vector<int> gt_id_to_row = std::vector<int>(256, -1); // label id -> index into rows (-1 = none)
    std::vector<int> pred_id_to_row = std::vector<int>(256, -1);
    bool matched_by_id = false; // no names matched, but one side only had generic label_<id> names
    float mean_dice = std::numeric_limits<float>::quiet_NaN();
    int scored = 0;
};

// Per-structure Dice and volumes for whichever of `gt`/`pred` are loaded
// (nullptr = not loaded) -- a full-volume pass, so run when either is
// (re)loaded, never per frame. Structures pair up by StructureMatchKey. If
// nothing matches and one side only has generic label_<id> names (a label
// file without names), they pair up by id instead; otherwise unmatched
// structures are simply unscored rather than compared against the wrong thing.
StructureMetrics ComputeStructureMetrics(const LabelVolume* gt, const LabelVolume* pred, const double spacing[3]);

// Error-map voxel values (see ComputeErrorVolume).
enum : uint8_t { kErrorNone = 0, kErrorMissed = 1, kErrorExtra = 2 };

// Where the prediction disagrees with the ground truth, for scored structures
// only (present on both sides -- a structure the model doesn't predict at all
// would otherwise flood the map): kErrorMissed = a GT voxel the prediction
// doesn't give that structure, kErrorExtra = the reverse. Empty (nx == 0)
// if the two volumes' shapes differ.
LabelVolume ComputeErrorVolume(const LabelVolume& gt, const LabelVolume& pred, const StructureMetrics& metrics);

// A voxel to put the cursor on to inspect `row`: the axial slice with the
// most of that structure's disagreement voxels (or, if it has none, its
// voxels on either side), at that slice's centroid of them. Returns false if
// the structure has no voxels at all. Either volume may be nullptr.
bool FindStructureFocus(const LabelVolume* gt, const LabelVolume* pred, const StructureMetric& row, int* out_i, int* out_j, int* out_k);
