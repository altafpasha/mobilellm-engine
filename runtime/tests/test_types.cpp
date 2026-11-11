#include <gtest/gtest.h>
#include "mobilellm/types.h"

using namespace mobilellm;

TEST(TypesTest, StatusOk) {
    Status s = Status::ok();
    EXPECT_TRUE(s.is_ok());
    EXPECT_TRUE(static_cast<bool>(s));
    EXPECT_EQ(s.code, StatusCode::OK);
    EXPECT_TRUE(s.message.empty());
}

TEST(TypesTest, StatusError) {
    Status s = Status::error(StatusCode::INVALID_ARGUMENT, "Invalid parameter");
    EXPECT_FALSE(s.is_ok());
    EXPECT_FALSE(static_cast<bool>(s));
    EXPECT_EQ(s.code, StatusCode::INVALID_ARGUMENT);
    EXPECT_EQ(s.message, "Invalid parameter");
}

TEST(TypesTest, ResultValue) {
    Result<int> res(42);
    EXPECT_TRUE(res.is_ok());
    EXPECT_TRUE(static_cast<bool>(res));
    EXPECT_EQ(res.value(), 42);
    EXPECT_TRUE(res.status().is_ok());
}

TEST(TypesTest, ResultStatusError) {
    Result<std::string> res(Status::error(StatusCode::FILE_NOT_FOUND, "Missing file"));
    EXPECT_FALSE(res.is_ok());
    EXPECT_FALSE(static_cast<bool>(res));
    EXPECT_EQ(res.status().code, StatusCode::FILE_NOT_FOUND);
    EXPECT_EQ(res.status().message, "Missing file");
}
