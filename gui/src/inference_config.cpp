#include "inference_config.h"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <vector>

#include <nlohmann/json.hpp>

using json = nlohmann::json;

bool LoadInferenceConfig(const std::string& repo_dir, const std::string& case_name, InferenceConfigUI& out)
{
    std::ifstream f(repo_dir + "/configs/" + case_name + "/inference_config.json");
    if (!f)
        return false;

    json j;
    try
    {
        f >> j;
    }
    catch (const json::exception& e)
    {
        // A hand-edited typo shouldn't crash the GUI at startup; keep the current settings.
        fprintf(stderr, "LoadInferenceConfig: malformed configs/%s/inference_config.json: %s\n", case_name.c_str(), e.what());
        return false;
    }
    if (!j.is_object())
        return false;

    out.backend = j.value("backend", std::string("local"));
    out.source_ct = j.value("source_ct", std::string());

    json m = j.value("model", json::object());
    out.model.dataset_id = m.value("dataset_id", std::string("501"));
    out.model.configuration = m.value("configuration", std::string("3d_fullres"));
    out.model.trainer = m.value("trainer", std::string("nnUNetTrainer_500epochs"));
    out.model.plans = m.value("plans", std::string("nnUNetPlans"));
    for (bool& enabled : out.model.fold_enabled)
        enabled = false;
    for (int fold : m.value("folds", std::vector<int>{ 0, 1, 2, 3 }))
        if (fold >= 0 && fold < 5)
            out.model.fold_enabled[fold] = true;
    out.model.results_dir = (m.contains("results_dir") && !m["results_dir"].is_null())
        ? m["results_dir"].get<std::string>() : std::string();

    json mo = j.value("modal", json::object());
    out.modal.app_name = mo.value("app_name", std::string("nnunet-training"));
    out.modal.function_name = mo.value("function_name", std::string("predict_nifti_bytes"));
    return true;
}

void SaveInferenceConfig(const std::string& repo_dir, const std::string& case_name, const InferenceConfigUI& cfg)
{
    std::filesystem::create_directories(repo_dir + "/configs/" + case_name);

    std::vector<int> folds;
    for (int i = 0; i < 5; ++i)
        if (cfg.model.fold_enabled[i])
            folds.push_back(i);

    json j;
    j["backend"] = cfg.backend;
    j["source_ct"] = cfg.source_ct;
    j["model"] = {
        { "dataset_id", cfg.model.dataset_id },
        { "configuration", cfg.model.configuration },
        { "trainer", cfg.model.trainer },
        { "plans", cfg.model.plans },
        { "folds", folds },
        { "results_dir", cfg.model.results_dir.empty() ? json(nullptr) : json(cfg.model.results_dir) },
    };
    j["modal"] = {
        { "app_name", cfg.modal.app_name },
        { "function_name", cfg.modal.function_name },
    };

    std::ofstream out(repo_dir + "/configs/" + case_name + "/inference_config.json");
    out << j.dump(2);
}
