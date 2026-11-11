#include <gtest/gtest.h>
#include "mobilellm/model_loader.h"
#include <fstream>
#include <cstdio>

using namespace mobilellm;

class ModelLoaderTest : public ::testing::Test {
protected:
    std::string valid_gguf_path = "/tmp/test_valid_model.gguf";
    std::string invalid_magic_path = "/tmp/test_invalid_magic.gguf";
    std::string invalid_version_path = "/tmp/test_invalid_ver.gguf";
    std::string truncated_path = "/tmp/test_truncated.gguf";

    void SetUp() override {
        // 1. Create a valid mock GGUF v3 file
        {
            std::ofstream f(valid_gguf_path, std::ios::binary);
            uint32_t magic = GGUF_MAGIC;
            uint32_t version = GGUF_VERSION_3;
            uint64_t tensor_count = 128;
            uint64_t kv_count = 0;
            f.write(reinterpret_cast<char*>(&magic), sizeof(magic));
            f.write(reinterpret_cast<char*>(&version), sizeof(version));
            f.write(reinterpret_cast<char*>(&tensor_count), sizeof(tensor_count));
            f.write(reinterpret_cast<char*>(&kv_count), sizeof(kv_count));
        }

        // 2. Create invalid magic file
        {
            std::ofstream f(invalid_magic_path, std::ios::binary);
            uint32_t magic = 0xDEADBEEF;
            uint32_t version = 3;
            uint64_t tensor_count = 10;
            uint64_t kv_count = 0;
            f.write(reinterpret_cast<char*>(&magic), sizeof(magic));
            f.write(reinterpret_cast<char*>(&version), sizeof(version));
            f.write(reinterpret_cast<char*>(&tensor_count), sizeof(tensor_count));
            f.write(reinterpret_cast<char*>(&kv_count), sizeof(kv_count));
        }

        // 3. Create invalid version file (v1)
        {
            std::ofstream f(invalid_version_path, std::ios::binary);
            uint32_t magic = GGUF_MAGIC;
            uint32_t version = 1; // v1 is not supported
            uint64_t tensor_count = 10;
            uint64_t kv_count = 0;
            f.write(reinterpret_cast<char*>(&magic), sizeof(magic));
            f.write(reinterpret_cast<char*>(&version), sizeof(version));
            f.write(reinterpret_cast<char*>(&tensor_count), sizeof(tensor_count));
            f.write(reinterpret_cast<char*>(&kv_count), sizeof(kv_count));
        }

        // 4. Create truncated file (only 4 bytes)
        {
            std::ofstream f(truncated_path, std::ios::binary);
            uint32_t magic = GGUF_MAGIC;
            f.write(reinterpret_cast<char*>(&magic), sizeof(magic));
        }
    }

    void TearDown() override {
        std::remove(valid_gguf_path.c_str());
        std::remove(invalid_magic_path.c_str());
        std::remove(invalid_version_path.c_str());
        std::remove(truncated_path.c_str());
    }
};

TEST_F(ModelLoaderTest, RejectsMissingOrEmptyPath) {
    auto res = ModelLoader::validate_file("");
    EXPECT_EQ(res.code, StatusCode::INVALID_ARGUMENT);

    auto res2 = ModelLoader::validate_file("/tmp/nonexistent_model_file.gguf");
    EXPECT_EQ(res2.code, StatusCode::FILE_NOT_FOUND);
}

TEST_F(ModelLoaderTest, RejectsTruncatedFile) {
    auto res = ModelLoader::validate_file(truncated_path);
    EXPECT_EQ(res.code, StatusCode::UNSUPPORTED_FORMAT);
}

TEST_F(ModelLoaderTest, RejectsInvalidMagicBytes) {
    auto res = ModelLoader::validate_file(invalid_magic_path);
    EXPECT_EQ(res.code, StatusCode::UNSUPPORTED_FORMAT);
}

TEST_F(ModelLoaderTest, RejectsUnsupportedVersion) {
    auto res = ModelLoader::validate_file(invalid_version_path);
    EXPECT_EQ(res.code, StatusCode::UNSUPPORTED_FORMAT);
}

TEST_F(ModelLoaderTest, ValidatesCorrectGGUFHeader) {
    auto res = ModelLoader::validate_file(valid_gguf_path);
    EXPECT_TRUE(res.is_ok());

    auto meta_res = ModelLoader::inspect_file(valid_gguf_path);
    ASSERT_TRUE(meta_res.is_ok());
    EXPECT_EQ(meta_res.value().gguf_version, 3);
    EXPECT_EQ(meta_res.value().tensor_count, 128);
}

TEST_F(ModelLoaderTest, MmapAndMunmap) {
    size_t size = 0;
    auto map_res = ModelLoader::mmap_file(valid_gguf_path, size);
    ASSERT_TRUE(map_res.is_ok());
    EXPECT_GT(size, 0);

    auto unmap_res = ModelLoader::munmap_file(map_res.value(), size);
    EXPECT_TRUE(unmap_res.is_ok());
}
