#include "HttpServer.hpp"

#include <arpa/inet.h>
#include <cerrno>
#include <cstring>
#include <iostream>
#include <netinet/in.h>
#include <sstream>
#include <stdexcept>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>

namespace {

std::string reasonPhrase(int status) {
    switch (status) {
    case 200:
        return "OK";
    case 400:
        return "Bad Request";
    case 404:
        return "Not Found";
    case 405:
        return "Method Not Allowed";
    case 409:
        return "Conflict";
    case 500:
        return "Internal Server Error";
    case 503:
        return "Service Unavailable";
    default:
        return "OK";
    }
}

std::string trim(const std::string &value) {
    const auto first = value.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) {
        return "";
    }
    const auto last = value.find_last_not_of(" \t\r\n");
    return value.substr(first, last - first + 1);
}

HttpRequest parseRequest(const std::string &raw) {
    HttpRequest request;
    const auto header_end = raw.find("\r\n\r\n");
    const auto header_block = raw.substr(0, header_end);
    if (header_end != std::string::npos) {
        request.body = raw.substr(header_end + 4);
    }

    std::istringstream stream(header_block);
    std::string line;
    if (!std::getline(stream, line)) {
        return request;
    }
    if (!line.empty() && line.back() == '\r') {
        line.pop_back();
    }

    std::istringstream request_line(line);
    request_line >> request.method >> request.path;

    while (std::getline(stream, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        const auto colon = line.find(':');
        if (colon == std::string::npos) {
            continue;
        }
        request.headers[trim(line.substr(0, colon))] = trim(line.substr(colon + 1));
    }

    return request;
}

std::string serializeResponse(const HttpResponse &response) {
    std::ostringstream out;
    out << "HTTP/1.1 " << response.status << ' ' << reasonPhrase(response.status) << "\r\n";
    out << "Content-Type: " << response.content_type << "\r\n";
    out << "Content-Length: " << response.body.size() << "\r\n";
    out << "Connection: close\r\n";
    for (const auto &[name, value] : response.headers) {
        out << name << ": " << value << "\r\n";
    }
    out << "\r\n";
    out << response.body;
    return out.str();
}

} // namespace

HttpServer::HttpServer(std::string host, int port, Handler handler)
    : host_(std::move(host)), port_(port), handler_(std::move(handler)) {}

HttpServer::~HttpServer() {
    stop();
}

void HttpServer::run() {
    server_fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd_ < 0) {
        throw std::runtime_error("socket failed: " + std::string(std::strerror(errno)));
    }

    int reuse = 1;
    if (::setsockopt(server_fd_, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse)) < 0) {
        throw std::runtime_error("setsockopt failed: " + std::string(std::strerror(errno)));
    }

    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_port = htons(static_cast<uint16_t>(port_));
    if (::inet_pton(AF_INET, host_.c_str(), &address.sin_addr) != 1) {
        throw std::runtime_error("invalid listen host: " + host_);
    }

    if (::bind(server_fd_, reinterpret_cast<sockaddr *>(&address), sizeof(address)) < 0) {
        throw std::runtime_error("bind failed: " + std::string(std::strerror(errno)));
    }

    if (::listen(server_fd_, SOMAXCONN) < 0) {
        throw std::runtime_error("listen failed: " + std::string(std::strerror(errno)));
    }

    running_ = true;
    std::cout << "neuriplo-kserve-runtime listening on " << host_ << ':' << port_ << '\n';

    while (running_) {
        sockaddr_in client_address{};
        socklen_t client_len = sizeof(client_address);
        const int client_fd =
            ::accept(server_fd_, reinterpret_cast<sockaddr *>(&client_address), &client_len);
        if (client_fd < 0) {
            if (running_) {
                std::cerr << "accept failed: " << std::strerror(errno) << '\n';
            }
            continue;
        }
        std::thread(&HttpServer::handleClient, this, client_fd).detach();
    }
}

void HttpServer::stop() {
    running_ = false;
    if (server_fd_ >= 0) {
        ::shutdown(server_fd_, SHUT_RDWR);
        ::close(server_fd_);
        server_fd_ = -1;
    }
}

void HttpServer::handleClient(int client_fd) const {
    std::string raw;
    char buffer[4096];

    while (true) {
        const auto received = ::recv(client_fd, buffer, sizeof(buffer), 0);
        if (received <= 0) {
            break;
        }
        raw.append(buffer, static_cast<size_t>(received));
        const auto header_end = raw.find("\r\n\r\n");
        if (header_end == std::string::npos) {
            continue;
        }

        const auto header_block = raw.substr(0, header_end);
        auto content_length = size_t{0};
        std::istringstream stream(header_block);
        std::string line;
        while (std::getline(stream, line)) {
            if (!line.empty() && line.back() == '\r') {
                line.pop_back();
            }
            const auto colon = line.find(':');
            if (colon == std::string::npos) {
                continue;
            }
            if (trim(line.substr(0, colon)) == "Content-Length") {
                content_length = static_cast<size_t>(std::stoul(trim(line.substr(colon + 1))));
            }
        }
        if (raw.size() >= header_end + 4 + content_length) {
            break;
        }
    }

    HttpResponse response;
    try {
        response = handler_(parseRequest(raw));
    } catch (const std::exception &error) {
        response.status = 500;
        response.body = std::string(R"({"error":")") + error.what() + R"("})";
    }

    const auto serialized = serializeResponse(response);
    ::send(client_fd, serialized.data(), serialized.size(), 0);
    ::shutdown(client_fd, SHUT_RDWR);
    ::close(client_fd);
}
