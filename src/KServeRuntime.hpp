#pragma once

#include "HttpTypes.hpp"
#include "ModelRegistry.hpp"

#include <string>

class KServeRuntime {
public:
    explicit KServeRuntime(ModelRegistry registry);

    HttpResponse handle(const HttpRequest& request) const;

private:
    HttpResponse serverMetadata() const;
    HttpResponse live() const;
    HttpResponse ready() const;
    HttpResponse modelMetadata(const std::string& model_name) const;
    HttpResponse modelReady(const std::string& model_name) const;
    HttpResponse infer(const std::string& model_name, const HttpRequest& request) const;

    ModelRegistry registry_;
};

