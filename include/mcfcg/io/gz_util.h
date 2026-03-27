#pragma once

#include <stdexcept>
#include <string>

#include <zlib.h>

namespace mcfcg {

inline bool ends_with_gz(const std::string & path) {
    return path.size() > 3 && path.compare(path.size() - 3, 3, ".gz") == 0;
}

inline std::string decompress_gz(const std::string & path) {
    gzFile gz = gzopen(path.c_str(), "rb");
    if (!gz) {
        throw std::runtime_error("Cannot open gzip file: " + path);
    }

    std::string result;
    char buf[1 << 16];
    int n;
    while ((n = gzread(gz, buf, sizeof(buf))) > 0) {
        result.append(buf, static_cast<size_t>(n));
    }
    if (n < 0) {
        int errnum = 0;
        const char * msg = gzerror(gz, &errnum);
        gzclose(gz);
        throw std::runtime_error("gzip read error in " + path + ": " +
                                 (msg ? msg : "unknown"));
    }
    gzclose(gz);
    return result;
}

}  // namespace mcfcg
