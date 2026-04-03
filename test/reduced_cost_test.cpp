#include <gtest/gtest.h>

#include <cstdio>
#include <fstream>
#include <string>

#include "cg_test_util.h"
#include "mcfcg/instance.h"

namespace {

void write_instance(const std::string& path, uint32_t vertices, uint32_t arcs,
                    uint32_t commodities, const std::string& arc_lines,
                    const std::string& commodity_lines) {
    std::ofstream f(path);
    f << vertices << '\n' << arcs << '\n' << commodities << '\n';
    f << arc_lines << commodity_lines;
}

}  // namespace

// --- Capacity binding (single commodity) ---

class RCCapBinding : public ::testing::Test {
   protected:
    std::string path = "rc_cap_binding.txt";
    void SetUp() override {
        write_instance(path, 4, 4, 1, "1 2 1 3\n1 3 5 10\n2 3 1 10\n3 4 1 10\n", "1 4 5\n");
    }
    void TearDown() override { std::remove(path.c_str()); }
};

TEST_F(RCCapBinding, PathRC) {
    auto inst = mcfcg::read_commalab(path);
    mcfcg::test::solve_and_validate_path_rc(inst, 21.0, 1e-4, true);
}

TEST_F(RCCapBinding, TreeRC) {
    auto inst = mcfcg::read_commalab(path);
    mcfcg::test::solve_and_validate_tree_rc(inst, 21.0, 1e-4, true);
}

// --- Multi-source + capacity ---

class RCMultiSourceCap : public ::testing::Test {
   protected:
    std::string path = "rc_multi_cap.txt";
    void SetUp() override {
        write_instance(path, 4, 4, 2, "1 3 1 10\n2 3 2 10\n3 4 1 5\n1 4 4 10\n",
                       "1 4 4\n2 4 3\n");
    }
    void TearDown() override { std::remove(path.c_str()); }
};

TEST_F(RCMultiSourceCap, PathRC) {
    auto inst = mcfcg::read_commalab(path);
    mcfcg::test::solve_and_validate_path_rc(inst, 21.0, 1e-4, true);
}

TEST_F(RCMultiSourceCap, TreeRC) {
    auto inst = mcfcg::read_commalab(path);
    mcfcg::test::solve_and_validate_tree_rc(inst, 21.0, 1e-4, true);
}
