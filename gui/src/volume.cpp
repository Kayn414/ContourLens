#include "volume.h"

#include <algorithm>
#include <fstream>
#include <iostream>

#include <nlohmann/json.hpp>

// Each public loader wraps its *Impl in a catch for nlohmann's exceptions, so
// a malformed or half-written sidecar logs and returns false instead of
// terminating the GUI (these files now also come from user-dropped data).
static bool LoadVolumeImpl(const std::string& base_path, Volume& out)
{
    std::ifstream json_file(base_path + ".json");
    if (!json_file)
    {
        std::cerr << "LoadVolume: cannot open " << base_path << ".json\n";
        return false;
    }

    nlohmann::json j;
    json_file >> j;

    auto shape = j.at("shape").get<std::vector<int>>(); // (nz, ny, nx)
    if (shape.size() != 3)
    {
        std::cerr << "LoadVolume: expected \"shape\" of length 3\n";
        return false;
    }

    std::string dtype = j.at("dtype").get<std::string>();
    if (dtype != "int16")
    {
        std::cerr << "LoadVolume: unsupported dtype \"" << dtype << "\" (only int16 is implemented)\n";
        return false;
    }

    auto spacing = j.at("spacing").get<std::vector<double>>();
    auto origin = j.at("origin").get<std::vector<double>>();
    auto direction = j.at("direction").get<std::vector<double>>();
    if (spacing.size() != 3 || origin.size() != 3 || direction.size() != 9)
    {
        std::cerr << "LoadVolume: malformed spacing/origin/direction\n";
        return false;
    }

    std::ifstream bin_file(base_path + ".bin", std::ios::binary);
    if (!bin_file)
    {
        std::cerr << "LoadVolume: cannot open " << base_path << ".bin\n";
        return false;
    }

    Volume volume;
    volume.nz = shape[0];
    volume.ny = shape[1];
    volume.nx = shape[2];
    for (int i = 0; i < 3; ++i) { volume.spacing[i] = spacing[i]; volume.origin[i] = origin[i]; }
    for (int i = 0; i < 9; ++i) volume.direction[i] = direction[i];

    size_t voxel_count = (size_t)volume.nx * volume.ny * volume.nz;
    volume.data.resize(voxel_count);
    bin_file.read(reinterpret_cast<char*>(volume.data.data()), voxel_count * sizeof(int16_t));
    if (!bin_file)
    {
        std::cerr << "LoadVolume: " << base_path << ".bin is smaller than the " << voxel_count << " voxels shape implies\n";
        return false;
    }

    out = std::move(volume);
    return true;
}

static bool LoadLabelVolumeImpl(const std::string& base_path, LabelVolume& out)
{
    std::ifstream json_file(base_path + ".json");
    if (!json_file)
    {
        std::cerr << "LoadLabelVolume: cannot open " << base_path << ".json\n";
        return false;
    }

    nlohmann::json j;
    json_file >> j;

    auto shape = j.at("shape").get<std::vector<int>>(); // (nz, ny, nx)
    if (shape.size() != 3)
    {
        std::cerr << "LoadLabelVolume: expected \"shape\" of length 3\n";
        return false;
    }

    std::string dtype = j.at("dtype").get<std::string>();
    if (dtype != "uint8")
    {
        std::cerr << "LoadLabelVolume: unsupported dtype \"" << dtype << "\" (only uint8 is implemented)\n";
        return false;
    }

    std::ifstream bin_file(base_path + ".bin", std::ios::binary);
    if (!bin_file)
    {
        std::cerr << "LoadLabelVolume: cannot open " << base_path << ".bin\n";
        return false;
    }

    LabelVolume labels;
    labels.nz = shape[0];
    labels.ny = shape[1];
    labels.nx = shape[2];
    labels.labels = j.at("labels").get<std::vector<std::string>>();

    size_t voxel_count = (size_t)labels.nx * labels.ny * labels.nz;
    labels.data.resize(voxel_count);
    bin_file.read(reinterpret_cast<char*>(labels.data.data()), voxel_count * sizeof(uint8_t));
    if (!bin_file)
    {
        std::cerr << "LoadLabelVolume: " << base_path << ".bin is smaller than the " << voxel_count << " voxels shape implies\n";
        return false;
    }

    // Every id present must have a name: callers index labels[id - 1] unchecked.
    uint8_t max_id = labels.data.empty() ? 0 : *std::max_element(labels.data.begin(), labels.data.end());
    while (labels.labels.size() < max_id)
        labels.labels.push_back("label_" + std::to_string(labels.labels.size() + 1));

    out = std::move(labels);
    return true;
}

static bool LoadDoseVolumeImpl(const std::string& base_path, DoseVolume& out)
{
    std::ifstream json_file(base_path + ".json");
    if (!json_file)
    {
        std::cerr << "LoadDoseVolume: cannot open " << base_path << ".json\n";
        return false;
    }

    nlohmann::json j;
    json_file >> j;

    auto shape = j.at("shape").get<std::vector<int>>(); // (nz, ny, nx)
    if (shape.size() != 3)
    {
        std::cerr << "LoadDoseVolume: expected \"shape\" of length 3\n";
        return false;
    }

    std::string dtype = j.at("dtype").get<std::string>();
    if (dtype != "float32")
    {
        std::cerr << "LoadDoseVolume: unsupported dtype \"" << dtype << "\" (only float32 is implemented)\n";
        return false;
    }

    std::ifstream bin_file(base_path + ".bin", std::ios::binary);
    if (!bin_file)
    {
        std::cerr << "LoadDoseVolume: cannot open " << base_path << ".bin\n";
        return false;
    }

    DoseVolume dose;
    dose.nz = shape[0];
    dose.ny = shape[1];
    dose.nx = shape[2];

    size_t voxel_count = (size_t)dose.nx * dose.ny * dose.nz;
    dose.data.resize(voxel_count);
    bin_file.read(reinterpret_cast<char*>(dose.data.data()), voxel_count * sizeof(float));
    if (!bin_file)
    {
        std::cerr << "LoadDoseVolume: " << base_path << ".bin is smaller than the " << voxel_count << " voxels shape implies\n";
        return false;
    }

    out = std::move(dose);
    return true;
}

bool LoadVolume(const std::string& base_path, Volume& out)
{
    try { return LoadVolumeImpl(base_path, out); }
    catch (const nlohmann::json::exception& e)
    {
        std::cerr << "LoadVolume: malformed " << base_path << ".json: " << e.what() << "\n";
        return false;
    }
}

bool LoadLabelVolume(const std::string& base_path, LabelVolume& out)
{
    try { return LoadLabelVolumeImpl(base_path, out); }
    catch (const nlohmann::json::exception& e)
    {
        std::cerr << "LoadLabelVolume: malformed " << base_path << ".json: " << e.what() << "\n";
        return false;
    }
}

bool LoadDoseVolume(const std::string& base_path, DoseVolume& out)
{
    try { return LoadDoseVolumeImpl(base_path, out); }
    catch (const nlohmann::json::exception& e)
    {
        std::cerr << "LoadDoseVolume: malformed " << base_path << ".json: " << e.what() << "\n";
        return false;
    }
}
