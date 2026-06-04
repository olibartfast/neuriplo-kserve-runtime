#pragma once

#include <functional>
#include <map>
#include <string>

struct HttpRequest {
    std::string method;
    std::string path;
    std::string body;
    std::map<std::string, std::string> headers;
};

struct StreamWriter {
    virtual ~StreamWriter() = default;
    virtual bool write(const std::string &data) = 0;
};

struct HttpResponse {
    int status = 200;
    std::string content_type = "application/json";
    std::string body;
    std::map<std::string, std::string> headers;
    bool streaming = false;
    std::function<void(StreamWriter &)> stream_callback;
};