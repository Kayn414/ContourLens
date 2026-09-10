#include "dvh.h"

#include <algorithm>
#include <cmath>

std::vector<float> GenerateSyntheticDose(const Volume& volume, float peak_dose_gy)
{
    double center_x = volume.origin[0] + volume.nx * volume.spacing[0] * 0.5;
    double center_y = volume.origin[1] + volume.ny * volume.spacing[1] * 0.5;
    double center_z = volume.origin[2] + volume.nz * volume.spacing[2] * 0.5;

    // Sigma sized off the volume's own extent so the falloff looks
    // reasonable regardless of the case's physical dimensions.
    double extent_x = volume.nx * volume.spacing[0];
    double extent_y = volume.ny * volume.spacing[1];
    double extent_z = volume.nz * volume.spacing[2];
    double sigma = 0.25 * std::min({ extent_x, extent_y, extent_z });
    double inv_two_sigma2 = 1.0 / (2.0 * sigma * sigma);

    std::vector<float> dose((size_t)volume.nx * volume.ny * volume.nz);
    for (int k = 0; k < volume.nz; ++k)
    {
        double world_z = volume.origin[2] + (k + 0.5) * volume.spacing[2];
        double dz = world_z - center_z;
        for (int j = 0; j < volume.ny; ++j)
        {
            double world_y = volume.origin[1] + (j + 0.5) * volume.spacing[1];
            double dy = world_y - center_y;
            size_t row_base = (size_t)(k * volume.ny + j) * volume.nx;
            for (int i = 0; i < volume.nx; ++i)
            {
                double world_x = volume.origin[0] + (i + 0.5) * volume.spacing[0];
                double dx = world_x - center_x;
                double r2 = dx * dx + dy * dy + dz * dz;
                dose[row_base + i] = (float)(peak_dose_gy * std::exp(-r2 * inv_two_sigma2));
            }
        }
    }
    return dose;
}

std::vector<DVHCurve> ComputeDVHCurves(const std::vector<float>& dose, const LabelVolume& labels, int num_bins)
{
    size_t voxel_count = (size_t)labels.nx * labels.ny * labels.nz;
    size_t num_labels = labels.labels.size();

    float max_dose = 0.0f;
    for (float d : dose) max_dose = std::max(max_dose, d);
    if (max_dose <= 0.0f) max_dose = 1.0f;

    std::vector<size_t> structure_voxels(num_labels, 0);
    std::vector<double> dose_sum(num_labels, 0.0);
    std::vector<float> dose_max(num_labels, 0.0f);
    std::vector<std::vector<size_t>> histogram(num_labels, std::vector<size_t>(num_bins, 0));

    for (size_t idx = 0; idx < voxel_count; ++idx)
    {
        uint8_t label = labels.data[idx];
        if (label == 0 || (size_t)(label - 1) >= num_labels)
            continue;
        size_t li = label - 1;
        structure_voxels[li]++;
        dose_sum[li] += dose[idx];
        dose_max[li] = std::max(dose_max[li], dose[idx]);
        int bin = (int)std::clamp((dose[idx] / max_dose) * num_bins, 0.0f, (float)(num_bins - 1));
        histogram[li][bin]++;
    }

    std::vector<DVHCurve> curves(num_labels);
    for (size_t li = 0; li < num_labels; ++li)
    {
        curves[li].name = labels.labels[li];
        curves[li].mean_gy = structure_voxels[li] > 0 ? (float)(dose_sum[li] / structure_voxels[li]) : 0.0f;
        curves[li].max_gy = dose_max[li];
        curves[li].dose_gy.resize(num_bins);
        curves[li].volume_pct.resize(num_bins);

        // Cumulative DVH: %volume receiving AT LEAST dose_gy[bin], so sum from the top bin down.
        size_t running = 0;
        std::vector<size_t> cumulative(num_bins, 0);
        for (int b = num_bins - 1; b >= 0; --b)
        {
            running += histogram[li][b];
            cumulative[b] = running;
        }

        for (int b = 0; b < num_bins; ++b)
        {
            curves[li].dose_gy[b] = (float)b / num_bins * max_dose;
            curves[li].volume_pct[b] = structure_voxels[li] > 0 ? 100.0f * cumulative[b] / structure_voxels[li] : 0.0f;
        }
    }
    return curves;
}

