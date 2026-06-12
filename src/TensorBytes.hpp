#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

// Canonical tensor payloads are contiguous native-endian byte buffers typed by
// the KServe datatype string. FP16 is stored widened to float32 internally so
// every element is directly addressable without a half-precision library; the
// neuriplo adapter has always handed FP16 tensors to backends as float bytes.

inline size_t tensorElementSize(const std::string &datatype) {
    if (datatype == "BOOL" || datatype == "INT8" || datatype == "UINT8") {
        return 1;
    }
    if (datatype == "INT16" || datatype == "UINT16") {
        return 2;
    }
    if (datatype == "INT32" || datatype == "UINT32" || datatype == "FP32" || datatype == "FP16") {
        return 4;
    }
    if (datatype == "INT64" || datatype == "UINT64" || datatype == "FP64") {
        return 8;
    }
    return 0; // BYTES and unknown datatypes carry no fixed-size elements
}

// Calls fn with a value-initialized element of the C++ type backing datatype.
// Returns false (without calling fn) for BYTES and unknown datatypes.
template <typename Fn> bool withTensorElementType(const std::string &datatype, Fn &&fn) {
    if (datatype == "BOOL" || datatype == "INT8") {
        fn(int8_t{});
        return true;
    }
    if (datatype == "INT16") {
        fn(int16_t{});
        return true;
    }
    if (datatype == "INT32") {
        fn(int32_t{});
        return true;
    }
    if (datatype == "INT64") {
        fn(int64_t{});
        return true;
    }
    if (datatype == "UINT8") {
        fn(uint8_t{});
        return true;
    }
    if (datatype == "UINT16") {
        fn(uint16_t{});
        return true;
    }
    if (datatype == "UINT32") {
        fn(uint32_t{});
        return true;
    }
    if (datatype == "UINT64") {
        fn(uint64_t{});
        return true;
    }
    if (datatype == "FP16" || datatype == "FP32") {
        fn(float{});
        return true;
    }
    if (datatype == "FP64") {
        fn(double{});
        return true;
    }
    return false;
}

inline size_t tensorElementCount(const std::string &datatype, const std::vector<std::byte> &bytes) {
    const auto element_size = tensorElementSize(datatype);
    return element_size == 0 ? 0 : bytes.size() / element_size;
}

template <typename T> void appendTensorScalar(std::vector<std::byte> &bytes, T value) {
    const size_t offset = bytes.size();
    bytes.resize(offset + sizeof(T));
    std::memcpy(bytes.data() + offset, &value, sizeof(T));
}

template <typename T> T tensorScalarAt(const std::vector<std::byte> &bytes, size_t element_index) {
    T value;
    std::memcpy(&value, bytes.data() + (element_index * sizeof(T)), sizeof(T));
    return value;
}

inline std::vector<std::byte> tensorBytesFromDoubles(const std::string &datatype,
                                                     const std::vector<double> &values) {
    std::vector<std::byte> bytes;
    withTensorElementType(datatype, [&](auto element) {
        using T = decltype(element);
        bytes.resize(values.size() * sizeof(T));
        for (size_t i = 0; i < values.size(); ++i) {
            const auto typed = static_cast<T>(values[i]);
            std::memcpy(bytes.data() + (i * sizeof(T)), &typed, sizeof(T));
        }
    });
    return bytes;
}

inline std::vector<double> tensorValuesAsDoubles(const std::string &datatype,
                                                 const std::vector<std::byte> &bytes) {
    std::vector<double> values;
    withTensorElementType(datatype, [&](auto element) {
        using T = decltype(element);
        const size_t count = bytes.size() / sizeof(T);
        values.resize(count);
        for (size_t i = 0; i < count; ++i) {
            T typed;
            std::memcpy(&typed, bytes.data() + (i * sizeof(T)), sizeof(T));
            values[i] = static_cast<double>(typed);
        }
    });
    return values;
}
