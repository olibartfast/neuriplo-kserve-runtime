#include "ModelRepository.hpp"
#include "ModelRegistry.hpp"
#include "PipelineExecutor.hpp"
#include "RuntimeConfig.hpp"
#include "Test.hpp"

#include <filesystem>
#include <fstream>
#include <string>

namespace fs = std::filesystem;

namespace {

// Each test gets its own tree so scans cannot see another test's models.
class TempRepository {
  public:
    explicit TempRepository(const std::string &tag) {
        root_ = fs::temp_directory_path() / ("neuriplo-repo-" + tag);
        fs::remove_all(root_);
        fs::create_directories(root_);
    }

    ~TempRepository() {
        fs::remove_all(root_);
    }

    TempRepository(const TempRepository &) = delete;
    TempRepository &operator=(const TempRepository &) = delete;

    void addModelFile(const std::string &model, const std::string &version,
                      const std::string &file_name) const {
        const auto dir = root_ / model / version;
        fs::create_directories(dir);
        std::ofstream out(dir / file_name);
        out << "not a real model";
    }

    void addVersionDir(const std::string &model, const std::string &version) const {
        fs::create_directories(root_ / model / version);
    }

    std::string root() const {
        return root_.string();
    }

  private:
    fs::path root_;
};

RuntimeConfig defaults() {
    RuntimeConfig config;
    config.backend = "stub";
    config.use_gpu = true;
    config.instances = 3;
    return config;
}

} // namespace

TEST_CASE(model_repository_maps_extensions_to_backends) {
    REQUIRE_EQ(backendForModelFile("model.onnx"), std::string("onnx_runtime"));
    REQUIRE_EQ(backendForModelFile("model.plan"), std::string("tensorrt"));
    REQUIRE_EQ(backendForModelFile("model.engine"), std::string("tensorrt"));
    REQUIRE_EQ(backendForModelFile("model.pte"), std::string("executorch"));
    REQUIRE_EQ(backendForModelFile("labels.txt"), std::string());
}

TEST_CASE(model_repository_extension_match_is_case_insensitive) {
    REQUIRE_EQ(backendForModelFile("MODEL.ONNX"), std::string("onnx_runtime"));
    REQUIRE_EQ(backendForModelFile("engine.PLAN"), std::string("tensorrt"));
}

TEST_CASE(model_repository_discovers_model_in_versioned_tree) {
    const TempRepository repo("discovers");
    repo.addModelFile("yolo26n-depth", "1", "model.onnx");

    std::vector<std::string> warnings;
    const auto configs = scanModelRepository(repo.root(), defaults(), warnings);
    REQUIRE_EQ(configs.size(), 1);
    REQUIRE_EQ(configs[0].model_name, std::string("yolo26n-depth"));
    REQUIRE_EQ(configs[0].model_version, std::string("1"));
    REQUIRE_EQ(configs[0].backend, std::string("onnx_runtime"));
    REQUIRE(configs[0].model_path.find("yolo26n-depth/1/model.onnx") != std::string::npos);
}

TEST_CASE(model_repository_inherits_defaults_the_tree_does_not_express) {
    const TempRepository repo("inherits");
    repo.addModelFile("m", "1", "model.onnx");

    std::vector<std::string> warnings;
    const auto configs = scanModelRepository(repo.root(), defaults(), warnings);
    REQUIRE_EQ(configs.size(), 1);
    REQUIRE(configs[0].use_gpu);
    REQUIRE_EQ(configs[0].instances, 3);
}

TEST_CASE(model_repository_serves_highest_numeric_version) {
    const TempRepository repo("versions");
    repo.addModelFile("m", "1", "model.onnx");
    repo.addModelFile("m", "9", "model.onnx");
    repo.addModelFile("m", "10", "model.onnx");

    std::vector<std::string> warnings;
    const auto configs = scanModelRepository(repo.root(), defaults(), warnings);
    REQUIRE_EQ(configs.size(), 1);
    // Lexical ordering would pick "9"; the highest version is 10.
    REQUIRE_EQ(configs[0].model_version, std::string("10"));
}

TEST_CASE(model_repository_maps_dali_pipeline_to_the_dali_backend) {
    REQUIRE_EQ(backendForModelFile("pipeline.dali"), std::string("dali"));
}

// A real repository keeps one export per backend in the same version directory.
// OpenVINO's .bin is a weight blob, not a model entry point, so only the .xml
// counts.
TEST_CASE(model_repository_picks_one_backend_from_a_multi_export_directory) {
    const TempRepository repo("multi-export");
    repo.addModelFile("ecdet", "1", "model.onnx");
    repo.addModelFile("ecdet", "1", "model.engine");
    repo.addModelFile("ecdet", "1", "model.xml");
    repo.addModelFile("ecdet", "1", "model.bin");
    repo.addModelFile("ecdet", "1", "model.pte");

    std::vector<std::string> warnings;
    const auto configs = scanModelRepository(repo.root(), defaults(), warnings);
    REQUIRE_EQ(configs.size(), 1);
    REQUIRE_EQ(configs[0].backend, std::string("tensorrt"));
    REQUIRE(configs[0].model_path.find("model.engine") != std::string::npos);
}

TEST_CASE(model_repository_ignores_openvino_weight_blob_without_its_xml) {
    const TempRepository repo("bin-only");
    repo.addModelFile("weights_only", "1", "model.bin");

    std::vector<std::string> warnings;
    const auto configs = scanModelRepository(repo.root(), defaults(), warnings);
    REQUIRE(configs.empty());
    REQUIRE(!warnings.empty());
}

TEST_CASE(model_repository_prefers_engine_over_the_onnx_it_was_built_from) {
    const TempRepository repo("prefers-engine");
    repo.addModelFile("m", "1", "model.onnx");
    repo.addModelFile("m", "1", "model.plan");

    std::vector<std::string> warnings;
    const auto configs = scanModelRepository(repo.root(), defaults(), warnings);
    REQUIRE_EQ(configs.size(), 1);
    REQUIRE_EQ(configs[0].backend, std::string("tensorrt"));
    REQUIRE(configs[0].model_path.find("model.plan") != std::string::npos);
}

TEST_CASE(model_repository_discovers_ensemble_graph_as_ensemble_backend) {
    const TempRepository repo("ensemble");
    repo.addModelFile("yolo_ensemble", "1", "graph.json");

    std::vector<std::string> warnings;
    const auto configs = scanModelRepository(repo.root(), defaults(), warnings);
    REQUIRE_EQ(configs.size(), 1);
    REQUIRE_EQ(configs[0].backend, std::string(pipelineBackendId()));
    // The executor reads the graph from model_path when pipeline_graph is empty.
    REQUIRE(configs[0].pipeline_graph.empty());
}

TEST_CASE(model_repository_discovers_multiple_models) {
    const TempRepository repo("multiple");
    repo.addModelFile("a", "1", "model.onnx");
    repo.addModelFile("b", "1", "model.plan");

    std::vector<std::string> warnings;
    const auto configs = scanModelRepository(repo.root(), defaults(), warnings);
    REQUIRE_EQ(configs.size(), 2);
    REQUIRE(warnings.empty());
}

TEST_CASE(model_repository_skips_model_without_numeric_version) {
    const TempRepository repo("no-version");
    repo.addModelFile("loose", "notaversion", "model.onnx");

    std::vector<std::string> warnings;
    const auto configs = scanModelRepository(repo.root(), defaults(), warnings);
    REQUIRE(configs.empty());
    REQUIRE(!warnings.empty());
}

TEST_CASE(model_repository_skips_version_without_recognized_model_file) {
    const TempRepository repo("empty-version");
    repo.addVersionDir("m", "1");

    std::vector<std::string> warnings;
    const auto configs = scanModelRepository(repo.root(), defaults(), warnings);
    REQUIRE(configs.empty());
    REQUIRE(!warnings.empty());
}

// One malformed model must not stop the rest of the repository from serving.
TEST_CASE(model_repository_reports_bad_model_but_keeps_the_good_one) {
    const TempRepository repo("mixed");
    repo.addModelFile("good", "1", "model.onnx");
    repo.addVersionDir("bad", "1");

    std::vector<std::string> warnings;
    const auto configs = scanModelRepository(repo.root(), defaults(), warnings);
    REQUIRE_EQ(configs.size(), 1);
    REQUIRE_EQ(configs[0].model_name, std::string("good"));
    REQUIRE_EQ(warnings.size(), 1);
}

// Regression: ModelLifecycle::load fell back to "1" whenever no explicit
// version override was passed, so a model configured for any other version was
// registered and reported as version 1. A repository tree makes that visible,
// since its version directory is the only place the version comes from.
TEST_CASE(model_registry_serves_the_explicitly_requested_version) {
    RuntimeConfig config;
    config.model_name = "demo";
    config.backend = "stub";
    config.model_version = "7";
    config.model_version_explicit = true;

    ModelRegistry registry(config);
    REQUIRE(registry.ready("demo"));
    REQUIRE_EQ(registry.defaultVersion("demo").value(), std::string("7"));
    REQUIRE(registry.findVersion("demo", "7").has_value());
}

// Without an explicit request the backend's own reported version still wins.
TEST_CASE(model_registry_defaults_to_the_backend_reported_version) {
    RuntimeConfig config;
    config.model_name = "demo";
    config.backend = "stub";

    ModelRegistry registry(config);
    REQUIRE(registry.ready("demo"));
    REQUIRE_EQ(registry.defaultVersion("demo").value(), std::string("1"));
}

TEST_CASE(model_repository_registry_serves_each_tree_version) {
    const TempRepository repo("registry-versions");
    repo.addModelFile("alpha", "3", "graph.json");

    std::vector<std::string> warnings;
    auto configs = scanModelRepository(repo.root(), defaults(), warnings);
    REQUIRE_EQ(configs.size(), 1);
    // Serve the discovered ensemble through a stub-friendly backend so the test
    // exercises versioning rather than a real backend build.
    configs[0].backend = "stub";

    ModelRegistry registry(configs);
    REQUIRE_EQ(registry.listModels().size(), 1);
    REQUIRE_EQ(registry.defaultVersion("alpha").value(), std::string("3"));
}

TEST_CASE(model_repository_warns_when_root_is_not_a_directory) {
    std::vector<std::string> warnings;
    const auto configs = scanModelRepository("/nonexistent/model/repository", defaults(), warnings);
    REQUIRE(configs.empty());
    REQUIRE_EQ(warnings.size(), 1);
}
