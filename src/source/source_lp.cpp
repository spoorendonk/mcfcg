#include "mcfcg/source/source_lp.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "mcfcg/io/gz_util.h"
#include "mcfcg/util/limits.h"

namespace mcfcg {

namespace {

// Append the CSC nonzeros of one column f^s_e (edge from->to in source block
// `base`) to `lp`: +1 in the tail conservation row, -1 in the head row, and +1
// in the capacity row when cap_row >= 0. Entries are merged by row so a
// self-loop (from == to) cancels instead of emitting a duplicate row index.
void append_source_column(SourceLP& lp, uint32_t base, uint32_t from, uint32_t to, int64_t cap_row,
                          uint32_t n_cons_rows) {
    std::pair<uint32_t, double> e[3];
    int n = 0;
    e[n++] = {base + from, 1.0};
    e[n++] = {base + to, -1.0};
    if (cap_row >= 0) {
        e[n++] = {n_cons_rows + static_cast<uint32_t>(cap_row), 1.0};
    }
    std::sort(e, e + n);
    for (int r = 0; r < n;) {
        const uint32_t row = e[r].first;
        double coef = e[r].second;
        for (++r; r < n && e[r].first == row; ++r) {
            coef += e[r].second;
        }
        if (coef != 0.0) {
            lp.row_index.push_back(row);
            lp.value.push_back(coef);
        }
    }
    lp.col_start.push_back(static_cast<uint32_t>(lp.row_index.size()));
}

}  // namespace

SourceLPSize source_lp_size(const Instance& inst) {
    const auto& g = inst.graph;
    SourceLPSize s;
    for (uint32_t a : g.arcs()) {
        if (inst.capacity[a] < INF) {
            ++s.capacitated_arcs;
        }
        if (g.arc_source(a) == g.arc_target(a)) {
            ++s.self_loop_arcs;
        }
    }
    const uint64_t n_sources = inst.sources.size();
    const uint64_t n_arcs = g.num_arcs();
    s.cols = n_sources * n_arcs;
    s.rows = n_sources * g.num_vertices() + s.capacitated_arcs;
    // Mirrors append_source_column entry for entry: +1 at the tail and -1 at the
    // head unless both land on the same row (self-loop -> cancels to nothing),
    // plus one capacity entry per capacitated arc. Every source repeats it.
    s.nnz = n_sources * (2 * (n_arcs - s.self_loop_arcs) + s.capacitated_arcs);
    return s;
}

SourceLP build_source_lp(const Instance& inst) {
    const auto& g = inst.graph;
    const auto n_sources = static_cast<uint32_t>(inst.sources.size());
    const uint32_t n_vertices = g.num_vertices();
    const uint32_t n_arcs = g.num_arcs();

    // Map each capacitated arc to its capacity-row offset; INF arcs get none.
    std::vector<int64_t> cap_row(n_arcs, -1);
    uint32_t n_cap = 0;
    for (uint32_t a : g.arcs()) {
        if (inst.capacity[a] < INF) {
            cap_row[a] = n_cap++;
        }
    }

    // The compact source LP has |S|*|E| columns, |S|*|V|+cap rows, and up to
    // 3*|S|*|E| nonzeros. Compute in 64-bit and reject before any of these
    // overflow the uint32_t indices below — on unique-source families
    // (intermodal, |S| ~ |K|) |S|*|E| can exceed 2^32, which would silently
    // wrap and emit a corrupt MPS. Such instances are impractical to dump
    // anyway, so failing loudly is correct.
    const uint64_t cols = static_cast<uint64_t>(n_sources) * n_arcs;
    const uint64_t rows = (static_cast<uint64_t>(n_sources) * n_vertices) + n_cap;
    if (cols > UINT32_MAX || rows > UINT32_MAX || cols * 3 > UINT32_MAX) {
        throw std::runtime_error(
            "compact source LP too large for the MPS exporter: |S|*|E| or "
            "|S|*|V| exceeds 2^32 (the source formulation is only practical "
            "when |S| is small relative to |K|)");
    }

    SourceLP lp;
    lp.num_cols = static_cast<uint32_t>(cols);
    lp.num_rows = static_cast<uint32_t>(rows);

    // Row bounds: conservation rows (equalities) then capacity rows (<=).
    lp.row_lower.assign(lp.num_rows, 0.0);
    lp.row_upper.assign(lp.num_rows, 0.0);
    for (uint32_t s = 0; s < n_sources; ++s) {
        const Source& src = inst.sources[s];
        const uint32_t base = s * n_vertices;
        // b[source] = D_s = sum of demands; b[sink] -= demand.
        double demand_sum = 0.0;
        for (uint32_t k : src.commodity_indices) {
            const Commodity& c = inst.commodities[k];
            lp.row_lower[base + c.sink] -= c.demand;
            lp.row_upper[base + c.sink] -= c.demand;
            demand_sum += c.demand;
        }
        lp.row_lower[base + src.vertex] += demand_sum;
        lp.row_upper[base + src.vertex] += demand_sum;
    }
    for (uint32_t a : g.arcs()) {
        if (cap_row[a] >= 0) {
            const uint32_t row = (n_sources * n_vertices) + static_cast<uint32_t>(cap_row[a]);
            lp.row_lower[row] = -INF;
            lp.row_upper[row] = inst.capacity[a];
        }
    }

    // Columns f^s_e: cost c_e, nonneg, nonzeros in the two conservation rows
    // (+1 outgoing at the tail, -1 incoming at the head) and the capacity row.
    lp.col_cost.reserve(lp.num_cols);
    lp.col_lower.assign(lp.num_cols, 0.0);
    lp.col_upper.assign(lp.num_cols, INF);
    lp.col_start.reserve(lp.num_cols + 1);
    lp.col_start.push_back(0);

    const uint32_t n_cons_rows = n_sources * n_vertices;
    for (uint32_t s = 0; s < n_sources; ++s) {
        const uint32_t base = s * n_vertices;
        for (uint32_t a : g.arcs()) {
            lp.col_cost.push_back(inst.cost[a]);
            append_source_column(lp, base, g.arc_source(a), g.arc_target(a), cap_row[a],
                                 n_cons_rows);
        }
    }

    return lp;
}

namespace {

// Buffered output sink that streams to a gzip file (path ends in .gz) or a plain
// file otherwise. Holds at most ~1 MiB of MPS *text* in memory regardless of
// instance size (the LP model itself is materialized by build_source_lp). Write
// failures are sticky: any failed flush or the final close sets a failure flag
// that close() reports, so a mid-file or disk-full error is never swallowed.
class MpsSink {
public:
    explicit MpsSink(const std::string& path) : _gz(ends_with_gz(path)) {
        if (_gz) {
            _gzf = gzopen(path.c_str(), "wb");
        } else {
            _fp = std::fopen(path.c_str(), "wb");
        }
        _failed = !ok();
        _buf.reserve(FLUSH + 256);
    }
    ~MpsSink() { close(); }
    MpsSink(const MpsSink&) = delete;
    MpsSink& operator=(const MpsSink&) = delete;

    [[nodiscard]] bool ok() const { return _gz ? _gzf != nullptr : _fp != nullptr; }

    void word(const char* s) { _buf.append(s); }
    void word(const std::string& s) { _buf.append(s); }
    void word(uint32_t v) { _buf.append(std::to_string(v)); }
    void num(double v) {
        char b[32];
        std::snprintf(b, sizeof(b), "%.17g", v);
        _buf.append(b);
    }
    void nl() {
        _buf.push_back('\n');
        if (_buf.size() >= FLUSH) {
            flush();
        }
    }
    void flush() {
        if (_failed || _buf.empty()) {
            return;
        }
        bool wrote;
        if (_gz) {
            wrote = gzwrite(_gzf, _buf.data(), static_cast<unsigned>(_buf.size())) > 0;
        } else {
            wrote = std::fwrite(_buf.data(), 1, _buf.size(), _fp) == _buf.size();
        }
        _buf.clear();
        _failed = _failed || !wrote;
    }
    // Flush remaining text, close the handle, and report whether everything
    // (including the final deflate/close) succeeded. Idempotent.
    bool close() {
        flush();
        if (_gzf != nullptr) {
            _failed = _failed || gzclose(_gzf) != Z_OK;
            _gzf = nullptr;
        }
        if (_fp != nullptr) {
            _failed = _failed || std::fclose(_fp) != 0;
            _fp = nullptr;
        }
        return !_failed;
    }

private:
    static constexpr size_t FLUSH = 1U << 20;
    bool _gz;
    bool _failed = false;
    gzFile _gzf = nullptr;
    std::FILE* _fp = nullptr;
    std::string _buf;
};

// Model name from the output path: basename with extensions stripped.
std::string model_name(const std::string& path) {
    size_t slash = path.find_last_of("/\\");
    std::string name = (slash == std::string::npos) ? path : path.substr(slash + 1);
    size_t dot = name.find('.');
    return dot == std::string::npos ? name : name.substr(0, dot);
}

}  // namespace

bool write_source_mps(const Instance& inst, const std::string& path) {
    const SourceLP lp = build_source_lp(inst);

    // A row is an equality (E, conservation) when its bounds coincide, otherwise
    // a <= row (L, capacity); free-format names (R<i>/C<j>) keep it simple.
    auto is_equality = [&](uint32_t i) { return lp.row_lower[i] == lp.row_upper[i]; };

    MpsSink out(path);
    if (!out.ok()) {
        return false;
    }

    out.word("NAME          ");
    out.word(model_name(path));
    out.nl();

    out.word("ROWS");
    out.nl();
    out.word(" N  COST");
    out.nl();
    for (uint32_t i = 0; i < lp.num_rows; ++i) {
        out.word(is_equality(i) ? " E  R" : " L  R");
        out.word(i);
        out.nl();
    }

    out.word("COLUMNS");
    out.nl();
    for (uint32_t j = 0; j < lp.num_cols; ++j) {
        if (lp.col_cost[j] != 0.0) {
            out.word("    C");
            out.word(j);
            out.word("  COST  ");
            out.num(lp.col_cost[j]);
            out.nl();
        }
        for (uint32_t p = lp.col_start[j]; p < lp.col_start[j + 1]; ++p) {
            out.word("    C");
            out.word(j);
            out.word("  R");
            out.word(lp.row_index[p]);
            out.word("  ");
            out.num(lp.value[p]);
            out.nl();
        }
    }

    out.word("RHS");
    out.nl();
    for (uint32_t i = 0; i < lp.num_rows; ++i) {
        const double rhs = is_equality(i) ? lp.row_lower[i] : lp.row_upper[i];
        if (rhs != 0.0) {
            out.word("    RHS  R");
            out.word(i);
            out.word("  ");
            out.num(rhs);
            out.nl();
        }
    }

    // The source LP's columns are all 0 <= f <= +inf (MPS default), so no BOUNDS
    // section is needed; guard generically in case that ever changes.
    bool need_bounds = false;
    for (uint32_t j = 0; j < lp.num_cols && !need_bounds; ++j) {
        need_bounds = lp.col_lower[j] != 0.0 || lp.col_upper[j] < INF;
    }
    if (need_bounds) {
        out.word("BOUNDS");
        out.nl();
        for (uint32_t j = 0; j < lp.num_cols; ++j) {
            if (lp.col_lower[j] != 0.0) {
                out.word(" LO BND  C");
                out.word(j);
                out.word("  ");
                out.num(lp.col_lower[j]);
                out.nl();
            }
            if (lp.col_upper[j] < INF) {
                out.word(" UP BND  C");
                out.word(j);
                out.word("  ");
                out.num(lp.col_upper[j]);
                out.nl();
            }
        }
    }

    out.word("ENDATA");
    out.nl();
    return out.close();
}

}  // namespace mcfcg
