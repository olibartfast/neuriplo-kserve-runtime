#pragma once

#ifdef NEURIPLO_RUNTIME_WITH_GRPC

#include "Executor.hpp"
#include "ModelMetadata.hpp"
#include "kserve_grpc.pb.h"

#include <string>

namespace grpc_v2 {

inference::ModelMetadataResponse buildModelMetadataResponse(const ModelMetadata &metadata);

ExecutionRequest convertInferRequest(const inference::ModelInferRequest &proto_request);

inference::ModelInferResponse buildInferResponse(const ExecutionResponse &exec_response,
                                                 const std::string &model_name,
                                                 const std::string &model_version,
                                                 const std::optional<std::string> &request_id);

} // namespace grpc_v2

#endif // NEURIPLO_RUNTIME_WITH_GRPC
