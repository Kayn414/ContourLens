#include "structure_metrics.h"

#include "imgui.h" // ImGui::ColorConvertHSVtoRGB

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <fstream>
#include <map>

#include <nlohmann/json.hpp>

// Canonical structure name -> color, so the SAME structure always gets the
// SAME color whether it comes from the ground-truth masks or the model
// prediction (source/data/hanseg.py's LABEL_IDS names). Matching by name (not
// by list position/index) is what makes overlaying both meaningful for
// comparison. Any other name gets a stable hashed color (ColorForStructureName).
static const char* kCanonicalStructureNames[] = {
    "brainstem", "optic_chiasm", "optic_nerve_l", "optic_nerve_r", "lens_l",
    "lens_r", "eyeball_l", "eyeball_r", "spinal_cord", "brain",
};
static const uint8_t kCanonicalStructureColors[][3] = {
    { 230, 60, 60 },   { 60, 160, 230 },  { 250, 200, 60 },
    { 120, 220, 120 }, { 200, 100, 220 }, { 80, 220, 200 },
    { 240, 140, 60 },  { 160, 160, 250 }, { 250, 250, 120 }, { 200, 200, 200 },
};
static const int kNumCanonicalStructures = sizeof(kCanonicalStructureNames) / sizeof(kCanonicalStructureNames[0]);

static std::vector<std::pair<std::string, std::string>> g_alias_entries; // as written in the file
static std::map<std::string, std::string> g_aliases;                     // normalized source -> normalized target

std::string NormalizeStructureName(const std::string& name)
{
    std::string normalized;
    for (unsigned char c : name)
        if (std::isalnum(c))
            normalized += (char)std::tolower(c);
    return normalized;
}

static std::string AliasesPath(const std::string& repo_dir)
{
    return repo_dir + "/configs/structure_aliases.json";
}

// The alias file's JSON: an empty object if the file doesn't exist; false
// (logged) if it exists but is malformed, so callers never overwrite it.
static bool ReadAliasesJson(const std::string& repo_dir, nlohmann::json& out)
{
    out = nlohmann::json::object();
    std::ifstream f(AliasesPath(repo_dir));
    if (!f)
        return true;
    try
    {
        f >> out;
    }
    catch (const nlohmann::json::exception& e)
    {
        fprintf(stderr, "configs/structure_aliases.json is malformed (%s); aliases disabled until it's fixed\n", e.what());
        return false;
    }
    if (!out.is_object())
    {
        fprintf(stderr, "configs/structure_aliases.json must be a JSON object; aliases disabled\n");
        return false;
    }
    return true;
}

void LoadStructureAliases(const std::string& repo_dir)
{
    g_alias_entries.clear();
    g_aliases.clear();

    nlohmann::json j;
    if (!ReadAliasesJson(repo_dir, j) || !j.contains("aliases") || !j["aliases"].is_object())
        return;
    for (const auto& item : j["aliases"].items())
    {
        if (!item.value().is_string())
            continue;
        std::string target = item.value().get<std::string>();
        g_alias_entries.emplace_back(item.key(), target);
        std::string from = NormalizeStructureName(item.key());
        std::string to = NormalizeStructureName(target);
        if (!from.empty() && !to.empty())
            g_aliases[from] = to;
    }
}

static bool WriteAliasesJson(const std::string& repo_dir, const nlohmann::json& j)
{
    std::ofstream out(AliasesPath(repo_dir));
    if (!out)
    {
        fprintf(stderr, "Can't write %s\n", AliasesPath(repo_dir).c_str());
        return false;
    }
    out << j.dump(2) << "\n";
    return (bool)out;
}

bool SaveStructureAlias(const std::string& repo_dir, const std::string& source, const std::string& target)
{
    nlohmann::json j;
    if (!ReadAliasesJson(repo_dir, j))
        return false;
    if (!j.contains("aliases") || !j["aliases"].is_object())
        j["aliases"] = nlohmann::json::object();
    j["aliases"][source] = target;
    bool ok = WriteAliasesJson(repo_dir, j);
    LoadStructureAliases(repo_dir);
    return ok;
}

bool RemoveStructureAlias(const std::string& repo_dir, const std::string& source)
{
    nlohmann::json j;
    if (!ReadAliasesJson(repo_dir, j))
        return false;
    if (j.contains("aliases") && j["aliases"].is_object())
        j["aliases"].erase(source);
    bool ok = WriteAliasesJson(repo_dir, j);
    LoadStructureAliases(repo_dir);
    return ok;
}

const std::vector<std::pair<std::string, std::string>>& StructureAliasEntries()
{
    return g_alias_entries;
}

std::string StructureMatchKey(const std::string& name)
{
    std::string key = NormalizeStructureName(name);
    auto it = g_aliases.find(key);
    return it != g_aliases.end() ? it->second : key;
}

// -1 if `name` isn't (an alias of) one of the canonical structures.
static int CanonIndexForStructureName(const std::string& name)
{
    std::string key = StructureMatchKey(name);
    for (int i = 0; i < kNumCanonicalStructures; ++i)
        if (key == NormalizeStructureName(kCanonicalStructureNames[i]))
            return i;
    return -1;
}

std::array<uint8_t, 3> ColorForStructureName(const std::string& name)
{
    int i = CanonIndexForStructureName(name);
    if (i >= 0)
        return { kCanonicalStructureColors[i][0], kCanonicalStructureColors[i][1], kCanonicalStructureColors[i][2] };

    uint32_t hash = 2166136261u; // FNV-1a of the match key -> hue: stable across runs, and shared by aliased names
    for (char c : StructureMatchKey(name))
    {
        hash ^= (uint8_t)c;
        hash *= 16777619u;
    }
    float r, g, b;
    ImGui::ColorConvertHSVtoRGB((float)(hash % 360) / 360.0f, 0.60f, 0.95f, r, g, b);
    return { (uint8_t)(r * 255.0f), (uint8_t)(g * 255.0f), (uint8_t)(b * 255.0f) };
}

std::vector<std::array<uint8_t, 3>> BuildLabelColors(const std::vector<std::string>& names)
{
    std::vector<std::array<uint8_t, 3>> colors(names.size());
    for (size_t i = 0; i < names.size(); ++i)
        colors[i] = ColorForStructureName(names[i]);
    return colors;
}

// "label_3" (or empty): the placeholder source/gui_load.py writes for a label file with no names.
static bool IsGenericLabelName(const std::string& name)
{
    if (name.empty())
        return true;
    if (name.size() <= 6 || name.compare(0, 6, "label_") != 0)
        return false;
    return std::all_of(name.begin() + 6, name.end(), [](unsigned char c) { return std::isdigit(c) != 0; });
}

StructureMetrics ComputeStructureMetrics(const LabelVolume* gt, const LabelVolume* pred, const double spacing[3])
{
    StructureMetrics m;
    std::vector<std::string> keys; // per row, for name matching
    auto add_row = [&](const std::string& name)
    {
        m.rows.push_back({});
        m.rows.back().name = name;
        keys.push_back(StructureMatchKey(name));
        return (int)m.rows.size() - 1;
    };

    if (gt)
        for (size_t i = 0; i < gt->labels.size() && i < 255; ++i)
        {
            int row = add_row(gt->labels[i]);
            m.rows[row].gt_id = (int)i + 1;
            m.gt_id_to_row[i + 1] = row;
        }

    if (pred)
    {
        size_t n_pred = std::min<size_t>(pred->labels.size(), 255);
        int name_matches = 0;
        if (gt)
            for (size_t i = 0; i < n_pred; ++i)
                if (std::find(keys.begin(), keys.end(), StructureMatchKey(pred->labels[i])) != keys.end())
                    ++name_matches;
        auto all_generic = [](const LabelVolume* v) { return std::all_of(v->labels.begin(), v->labels.end(), IsGenericLabelName); };
        m.matched_by_id = gt && name_matches == 0 && !gt->labels.empty() && n_pred > 0 && (all_generic(gt) || all_generic(pred));

        for (size_t i = 0; i < n_pred; ++i)
        {
            int id = (int)i + 1;
            int row = -1;
            if (m.matched_by_id)
            {
                row = m.gt_id_to_row[id];
                if (row >= 0 && IsGenericLabelName(m.rows[row].name))
                    m.rows[row].name = pred->labels[i]; // keep whichever side has a real name
            }
            else if (gt)
            {
                std::string key = StructureMatchKey(pred->labels[i]);
                for (size_t r = 0; r < m.rows.size(); ++r)
                    if (m.rows[r].gt_id != 0 && m.rows[r].pred_id == 0 && keys[r] == key)
                    {
                        row = (int)r;
                        break;
                    }
            }
            if (row < 0)
                row = add_row(pred->labels[i]);
            m.rows[row].pred_id = id;
            m.pred_id_to_row[id] = row;
        }
    }

    std::array<size_t, 256> gt_count{}, pred_count{};
    std::vector<size_t> intersection(m.rows.size(), 0);
    if (gt && pred && gt->data.size() == pred->data.size())
    {
        for (size_t v = 0; v < gt->data.size(); ++v)
        {
            uint8_t g = gt->data[v], p = pred->data[v];
            ++gt_count[g];
            ++pred_count[p];
            int row = m.gt_id_to_row[g];
            if (g != 0 && p != 0 && row >= 0 && row == m.pred_id_to_row[p])
                ++intersection[row];
        }
    }
    else
    {
        if (gt)
            for (uint8_t g : gt->data) ++gt_count[g];
        if (pred)
            for (uint8_t p : pred->data) ++pred_count[p];
    }

    double voxel_cc = spacing[0] * spacing[1] * spacing[2] / 1000.0; // mm^3 -> cc
    double dice_sum = 0.0;
    for (size_t r = 0; r < m.rows.size(); ++r)
    {
        StructureMetric& row = m.rows[r];
        row.gt_voxels = row.gt_id ? gt_count[row.gt_id] : 0;
        row.pred_voxels = row.pred_id ? pred_count[row.pred_id] : 0;
        row.gt_cc = row.gt_voxels * voxel_cc;
        row.pred_cc = row.pred_voxels * voxel_cc;
        size_t denom = row.gt_voxels + row.pred_voxels;
        if (row.gt_id && row.pred_id && denom > 0)
        {
            row.dice = 2.0f * (float)intersection[r] / (float)denom;
            dice_sum += row.dice;
            ++m.scored;
        }
    }
    if (m.scored > 0)
        m.mean_dice = (float)(dice_sum / m.scored);
    return m;
}

LabelVolume ComputeErrorVolume(const LabelVolume& gt, const LabelVolume& pred, const StructureMetrics& metrics)
{
    LabelVolume errors;
    if (gt.data.empty() || gt.data.size() != pred.data.size())
        return errors;
    errors.nx = gt.nx;
    errors.ny = gt.ny;
    errors.nz = gt.nz;
    errors.labels = { "missed", "extra" };
    errors.data.assign(gt.data.size(), kErrorNone);

    std::vector<bool> scored(metrics.rows.size());
    for (size_t r = 0; r < metrics.rows.size(); ++r)
        scored[r] = metrics.rows[r].gt_id != 0 && metrics.rows[r].pred_id != 0;

    for (size_t v = 0; v < gt.data.size(); ++v)
    {
        int gt_row = gt.data[v] ? metrics.gt_id_to_row[gt.data[v]] : -1;
        int pred_row = pred.data[v] ? metrics.pred_id_to_row[pred.data[v]] : -1;
        if (gt_row == pred_row)
            continue;
        if (gt_row >= 0 && scored[gt_row])
            errors.data[v] = kErrorMissed;
        else if (pred_row >= 0 && scored[pred_row])
            errors.data[v] = kErrorExtra;
    }
    return errors;
}

bool FindStructureFocus(const LabelVolume* gt, const LabelVolume* pred, const StructureMetric& row, int* out_i, int* out_j, int* out_k)
{
    if (gt && pred && gt->data.size() != pred->data.size())
        pred = nullptr; // only index volumes that share a grid
    const LabelVolume* ref = gt ? gt : pred;
    if (!ref || ref->data.empty())
        return false;

    const int nx = ref->nx, ny = ref->ny, nz = ref->nz;
    const size_t slice_size = (size_t)nx * ny;
    // mode 0: voxels where the two sides disagree about this structure; mode 1: any of its voxels.
    auto hit = [&](size_t idx, int mode)
    {
        bool in_gt = gt && row.gt_id && gt->data[idx] == row.gt_id;
        bool in_pred = pred && row.pred_id && pred->data[idx] == row.pred_id;
        return mode == 0 ? in_gt != in_pred : (in_gt || in_pred);
    };

    for (int mode = 0; mode < 2; ++mode)
    {
        if (mode == 0 && !(gt && pred && row.gt_id && row.pred_id))
            continue;

        std::vector<size_t> per_slice(nz, 0);
        for (int k = 0; k < nz; ++k)
            for (size_t p = 0; p < slice_size; ++p)
                if (hit(k * slice_size + p, mode))
                    ++per_slice[k];

        auto best = std::max_element(per_slice.begin(), per_slice.end());
        if (best == per_slice.end() || *best == 0)
            continue;

        int k = (int)(best - per_slice.begin());
        double sum_i = 0.0, sum_j = 0.0;
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i)
                if (hit(k * slice_size + (size_t)j * nx + i, mode))
                {
                    sum_i += i;
                    sum_j += j;
                }
        *out_i = (int)(sum_i / *best + 0.5);
        *out_j = (int)(sum_j / *best + 0.5);
        *out_k = k;
        return true;
    }
    return false;
}
