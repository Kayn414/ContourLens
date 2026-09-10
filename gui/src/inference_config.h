#pragma once

// Editable mirror of source/inference_config.py's InferenceConfig schema, so
// the GUI can create/edit configs/<case_name>/inference_config.json directly
// instead of requiring users to hand-write JSON to point the app at their
// own model/dataset and choose a local vs. cloud backend.

#include <string>

struct ModelConfigUI
{
    std::string dataset_id = "501";
    std::string configuration = "3d_fullres";
    std::string trainer = "nnUNetTrainer_500epochs";
    std::string plans = "nnUNetPlans";
    bool fold_enabled[5] = { true, true, true, true, false }; // nnU-Net's standard 5-fold CV (0-4)
    std::string results_dir; // empty => JSON null => use the nnUNet_results env var
};

struct ModalConfigUI
{
    std::string app_name = "nnunet-training";
    std::string function_name = "predict_nifti_bytes";
};

struct InferenceConfigUI
{
    std::string backend = "local"; // "local" | "modal"
    std::string source_ct;         // NIfTI/NRRD file or DICOM series dir, relative to the repo root
    ModelConfigUI model;
    ModalConfigUI modal;
};

// Reads configs/<case_name>/inference_config.json under `repo_dir` into `out`.
// Returns false (leaving `out` at its defaults) if the file doesn't exist yet --
// the GUI lets the user fill in fields and Save to create one for a new case.
bool LoadInferenceConfig(const std::string& repo_dir, const std::string& case_name, InferenceConfigUI& out);

// Writes `cfg` to configs/<case_name>/inference_config.json under `repo_dir`
// (creating configs/<case_name>/ if needed), in source/inference_config.py's schema.
void SaveInferenceConfig(const std::string& repo_dir, const std::string& case_name, const InferenceConfigUI& cfg);
