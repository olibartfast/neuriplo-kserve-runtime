#pragma once

#ifdef NEURIPLO_RUNTIME_WITH_GRPC

#include "MetricsRegistry.hpp"
#include "ModelRegistry.hpp"

#include <atomic>
#include <memory>
#include <string>
#include <thread>

namespace grpc_v2 {

class GrpcServer {
  public:
    GrpcServer(std::string host, int port, ModelRegistry &registry, MetricsRegistry &metrics);
    ~GrpcServer();

    GrpcServer(const GrpcServer &) = delete;
    GrpcServer &operator=(const GrpcServer &) = delete;

    void run();
    void stop();

  private:
    std::string host_;
    int port_;
    ModelRegistry &registry_;
    MetricsRegistry &metrics_;
    std::unique_ptr<class GrpcServiceImpl> service_;
    std::unique_ptr<class GrpcServerImpl> impl_;
};

} // namespace grpc_v2

#endif // NEURIPLO_RUNTIME_WITH_GRPC
