#include <gtest/gtest.h>

#include "mcfcg/graph/static_map.h"

TEST(StaticMap, DefaultConstruction) {
    mcfcg::static_map<uint32_t, int> m;
    EXPECT_EQ(m.size(), 0u);
}

TEST(StaticMap, SizedConstruction) {
    mcfcg::static_map<uint32_t, int> m(10);
    EXPECT_EQ(m.size(), 10u);
}

TEST(StaticMap, InitValueConstruction) {
    mcfcg::static_map<uint32_t, int> m(5, 42);
    EXPECT_EQ(m.size(), 5u);
    for (uint32_t i = 0; i < 5; ++i) {
        EXPECT_EQ(m[i], 42);
    }
}

TEST(StaticMap, Indexing) {
    mcfcg::static_map<uint32_t, int> m(3, 0);
    m[0] = 10;
    m[1] = 20;
    m[2] = 30;
    EXPECT_EQ(m[0], 10);
    EXPECT_EQ(m[1], 20);
    EXPECT_EQ(m[2], 30);
}

TEST(StaticMap, Fill) {
    mcfcg::static_map<uint32_t, int> m(4, 0);
    m.fill(99);
    for (uint32_t i = 0; i < 4; ++i) {
        EXPECT_EQ(m[i], 99);
    }
}

TEST(StaticMap, CopyConstruction) {
    mcfcg::static_map<uint32_t, int> m(3, 0);
    m[0] = 1;
    m[1] = 2;
    m[2] = 3;
    mcfcg::static_map<uint32_t, int> m2(m);
    EXPECT_EQ(m2[0], 1);
    EXPECT_EQ(m2[1], 2);
    EXPECT_EQ(m2[2], 3);
    // Verify independent
    m2[0] = 99;
    EXPECT_EQ(m[0], 1);
}

TEST(StaticMap, MoveConstruction) {
    mcfcg::static_map<uint32_t, int> m(3, 7);
    mcfcg::static_map<uint32_t, int> m2(std::move(m));
    EXPECT_EQ(m2.size(), 3u);
    EXPECT_EQ(m2[0], 7);
}

TEST(StaticMap, Iteration) {
    mcfcg::static_map<uint32_t, int> m(3, 0);
    m[0] = 10;
    m[1] = 20;
    m[2] = 30;
    int sum = 0;
    for (int v : m)
        sum += v;
    EXPECT_EQ(sum, 60);
}
