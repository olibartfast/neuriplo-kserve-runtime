#pragma once

#include <map>
#include <string>

struct HttpRequest {
    std::string method;
    std::string path;
    std::string body;
    std::map<std::string, std::string> headers;
};

struct HttpResponse {
    int status = 200;
    std::string content_type = "application/json";
    std::string body;
    std::map<std::string, std::string> headers;
};
