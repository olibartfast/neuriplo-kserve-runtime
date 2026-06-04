#include "Executor.hpp"

ExecutionResponse Executor::inferStreaming(const ExecutionRequest &request,
                                           StreamingTokenCallback /*callback*/) {
    return infer(request);
}