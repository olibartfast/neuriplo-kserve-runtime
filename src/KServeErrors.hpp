#pragma once

#include <string>

namespace KServeErrors {

inline constexpr const char *InvalidArgument = "INVALID_ARGUMENT";
inline constexpr const char *ModelNotFound = "MODEL_NOT_FOUND";
inline constexpr const char *ModelNotReady = "MODEL_NOT_READY";
inline constexpr const char *QueueFull = "QUEUE_FULL";
inline constexpr const char *DeadlineExceeded = "DEADLINE_EXCEEDED";
inline constexpr const char *BackendError = "BACKEND_ERROR";
inline constexpr const char *Unavailable = "UNAVAILABLE";
inline constexpr const char *Internal = "INTERNAL";
inline constexpr const char *NotFound = "NOT_FOUND";
inline constexpr const char *MethodNotAllowed = "METHOD_NOT_ALLOWED";
inline constexpr const char *PayloadTooLarge = "PAYLOAD_TOO_LARGE";

inline int httpStatusForCode(const std::string &code) {
    if (code == InvalidArgument)
        return 400;
    if (code == ModelNotFound)
        return 404;
    if (code == ModelNotReady)
        return 409;
    if (code == QueueFull)
        return 429;
    if (code == DeadlineExceeded)
        return 504;
    if (code == BackendError || code == Internal)
        return 500;
    if (code == Unavailable)
        return 503;
    if (code == NotFound)
        return 404;
    if (code == MethodNotAllowed)
        return 405;
    if (code == PayloadTooLarge)
        return 413;
    return 500;
}

} // namespace KServeErrors
