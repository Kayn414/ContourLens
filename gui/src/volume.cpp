#include "volume.h"

#include <fstream>
#include <iostream>

#include <nlohmann/json.hpp>

bool LoadVolume(const std::string& base_path, Volume& out)
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
