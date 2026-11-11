#include "mobilellm/model_loader.h"
#include <fstream>
#include <cstring>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>

namespace mobilellm {

// GGUF Metadata Value Types
enum class GGUFType : uint32_t {
    UINT8   = 0,
    INT8    = 1,
    UINT16  = 2,
    INT16   = 3,
    UINT32  = 4,
    INT32   = 5,
    FLOAT32 = 6,
    BOOL    = 7,
    STRING  = 8,
    ARRAY   = 9,
    UINT64  = 10,
    INT64   = 11,
    FLOAT64 = 12,
};

static uint64_t get_file_size(const std::string& path) {
    struct stat st;
    if (stat(path.c_str(), &st) != 0) {
        return 0;
    }
    return static_cast<uint64_t>(st.st_size);
}

Status ModelLoader::validate_file(const std::string& filepath) noexcept {
    if (filepath.empty()) {
        return Status::error(StatusCode::INVALID_ARGUMENT, "Model file path cannot be empty");
    }

    uint64_t file_size = get_file_size(filepath);
    if (file_size == 0) {
        return Status::error(StatusCode::FILE_NOT_FOUND, "Model file does not exist or is empty: " + filepath);
    }

    if (file_size < sizeof(uint32_t) * 2 + sizeof(uint64_t) * 2) {
        return Status::error(StatusCode::UNSUPPORTED_FORMAT, "File is too small to contain a valid GGUF header");
    }

    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        return Status::error(StatusCode::FILE_NOT_FOUND, "Failed to open model file: " + filepath);
    }

    uint32_t magic = 0;
    file.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    if (magic != GGUF_MAGIC) {
        return Status::error(StatusCode::UNSUPPORTED_FORMAT, 
            "Invalid GGUF magic bytes: expected 0x46554747, got 0x" + std::to_string(magic));
    }

    uint32_t version = 0;
    file.read(reinterpret_cast<char*>(&version), sizeof(version));
    if (version != GGUF_VERSION_2 && version != GGUF_VERSION_3) {
        return Status::error(StatusCode::UNSUPPORTED_FORMAT, 
            "Unsupported GGUF version: " + std::to_string(version) + " (supported: v2, v3)");
    }

    return Status::ok();
}

static bool read_string(std::ifstream& file, std::string& out_str) {
    uint64_t len = 0;
    if (!file.read(reinterpret_cast<char*>(&len), sizeof(len))) return false;
    if (len > 1024 * 1024) return false; // Sanity limit for metadata strings

    out_str.resize(len);
    if (!file.read(&out_str[0], len)) return false;
    return true;
}

static bool skip_value(std::ifstream& file, uint32_t type_val) {
    auto type = static_cast<GGUFType>(type_val);
    switch (type) {
        case GGUFType::UINT8:
        case GGUFType::INT8:
        case GGUFType::BOOL:
            file.seekg(1, std::ios::cur);
            return file.good();
        case GGUFType::UINT16:
        case GGUFType::INT16:
            file.seekg(2, std::ios::cur);
            return file.good();
        case GGUFType::UINT32:
        case GGUFType::INT32:
        case GGUFType::FLOAT32:
            file.seekg(4, std::ios::cur);
            return file.good();
        case GGUFType::UINT64:
        case GGUFType::INT64:
        case GGUFType::FLOAT64:
            file.seekg(8, std::ios::cur);
            return file.good();
        case GGUFType::STRING: {
            std::string dummy;
            return read_string(file, dummy);
        }
        case GGUFType::ARRAY: {
            uint32_t elem_type = 0;
            uint64_t count = 0;
            if (!file.read(reinterpret_cast<char*>(&elem_type), sizeof(elem_type))) return false;
            if (!file.read(reinterpret_cast<char*>(&count), sizeof(count))) return false;
            for (uint64_t i = 0; i < count; ++i) {
                if (!skip_value(file, elem_type)) return false;
            }
            return file.good();
        }
        default:
            return false;
    }
}

Result<ModelMetadata> ModelLoader::inspect_file(const std::string& filepath) noexcept {
    Status validation = validate_file(filepath);
    if (!validation.is_ok()) {
        return validation;
    }

    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        return Status::error(StatusCode::FILE_NOT_FOUND, "Cannot open file: " + filepath);
    }

    GGUFHeader header;
    file.read(reinterpret_cast<char*>(&header.magic), sizeof(header.magic));
    file.read(reinterpret_cast<char*>(&header.version), sizeof(header.version));
    file.read(reinterpret_cast<char*>(&header.tensor_count), sizeof(header.tensor_count));
    file.read(reinterpret_cast<char*>(&header.metadata_kv_count), sizeof(header.metadata_kv_count));

    ModelMetadata meta;
    meta.filepath = filepath;
    meta.gguf_version = header.version;
    meta.tensor_count = header.tensor_count;
    meta.metadata_kv_count = header.metadata_kv_count;
    meta.file_size_bytes = get_file_size(filepath);
    meta.architecture = "unknown";
    meta.quantization = "unknown";

    // Read Key-Value metadata
    for (uint64_t i = 0; i < header.metadata_kv_count; ++i) {
        if (!file.good()) break;

        std::string key;
        if (!read_string(file, key)) break;

        uint32_t val_type_raw = 0;
        if (!file.read(reinterpret_cast<char*>(&val_type_raw), sizeof(val_type_raw))) break;
        auto val_type = static_cast<GGUFType>(val_type_raw);

        if (key == "general.architecture" && val_type == GGUFType::STRING) {
            read_string(file, meta.architecture);
        } else if (key == "general.name" && val_type == GGUFType::STRING) {
            read_string(file, meta.description);
        } else if (key.find(".context_length") != std::string::npos) {
            if (val_type == GGUFType::UINT32) {
                file.read(reinterpret_cast<char*>(&meta.context_length), sizeof(uint32_t));
            } else if (val_type == GGUFType::UINT64) {
                uint64_t val = 0;
                file.read(reinterpret_cast<char*>(&val), sizeof(val));
                meta.context_length = static_cast<uint32_t>(val);
            } else {
                skip_value(file, val_type_raw);
            }
        } else if (key.find(".embedding_length") != std::string::npos) {
            if (val_type == GGUFType::UINT32) {
                file.read(reinterpret_cast<char*>(&meta.embedding_length), sizeof(uint32_t));
            } else if (val_type == GGUFType::UINT64) {
                uint64_t val = 0;
                file.read(reinterpret_cast<char*>(&val), sizeof(val));
                meta.embedding_length = static_cast<uint32_t>(val);
            } else {
                skip_value(file, val_type_raw);
            }
        } else if (key.find(".block_count") != std::string::npos) {
            if (val_type == GGUFType::UINT32) {
                file.read(reinterpret_cast<char*>(&meta.block_count), sizeof(uint32_t));
            } else if (val_type == GGUFType::UINT64) {
                uint64_t val = 0;
                file.read(reinterpret_cast<char*>(&val), sizeof(val));
                meta.block_count = static_cast<uint32_t>(val);
            } else {
                skip_value(file, val_type_raw);
            }
        } else if (key.find(".attention.head_count") != std::string::npos) {
            if (val_type == GGUFType::UINT32) {
                file.read(reinterpret_cast<char*>(&meta.head_count), sizeof(uint32_t));
            } else if (val_type == GGUFType::UINT64) {
                uint64_t val = 0;
                file.read(reinterpret_cast<char*>(&val), sizeof(val));
                meta.head_count = static_cast<uint32_t>(val);
            } else {
                skip_value(file, val_type_raw);
            }
        } else if (key == "general.file_type") {
            uint32_t ftype = 0;
            if (val_type == GGUFType::UINT32) {
                file.read(reinterpret_cast<char*>(&ftype), sizeof(ftype));
                meta.quantization = "type_" + std::to_string(ftype);
            } else {
                skip_value(file, val_type_raw);
            }
        } else {
            if (!skip_value(file, val_type_raw)) break;
        }
    }

    return meta;
}

Result<void*> ModelLoader::mmap_file(const std::string& filepath, size_t& out_size) noexcept {
    int fd = open(filepath.c_str(), O_RDONLY);
    if (fd < 0) {
        return Status::error(StatusCode::FILE_NOT_FOUND, "Cannot open file for mmap: " + filepath);
    }

    struct stat sb;
    if (fstat(fd, &sb) < 0) {
        close(fd);
        return Status::error(StatusCode::INTERNAL_ERROR, "Cannot fstat file: " + filepath);
    }

    out_size = static_cast<size_t>(sb.st_size);
    if (out_size == 0) {
        close(fd);
        return Status::error(StatusCode::INVALID_ARGUMENT, "Cannot mmap zero-length file: " + filepath);
    }

    void* addr = mmap(nullptr, out_size, PROT_READ, MAP_SHARED, fd, 0);
    close(fd); // File descriptor can be closed once mapped

    if (addr == MAP_FAILED) {
        return Status::error(StatusCode::OUT_OF_MEMORY, "mmap failed for: " + filepath);
    }

    return addr;
}

Status ModelLoader::munmap_file(void* addr, size_t size) noexcept {
    if (!addr || size == 0) {
        return Status::ok();
    }
    if (munmap(addr, size) != 0) {
        return Status::error(StatusCode::INTERNAL_ERROR, "Failed to munmap memory buffer");
    }
    return Status::ok();
}

} // namespace mobilellm
