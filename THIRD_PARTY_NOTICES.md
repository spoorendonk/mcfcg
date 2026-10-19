# Third-party notices

mcfcg is MIT-licensed (`LICENSE`). Parts of its graph layer derive from other
work under other licences, listed here.

## melon

- Project: melon, a C++ graph library — https://github.com/fhamonic/melon
- Copyright: François Hamonic
- Licence: Boost Software License 1.0, text in `LICENSES/BSL-1.0.txt`

Derived from melon (copied, then adapted), so these files remain under BSL-1.0:

| File | melon source |
|---|---|
| `include/mcfcg/graph/static_map.h` | `include/melon/container/static_map.hpp` |
| `include/mcfcg/graph/static_digraph.h` | `include/melon/container/static_digraph.hpp` |
| `include/mcfcg/graph/dijkstra.h` (the `dijkstra` class; its A* and bounded searches are mcfcg's own) | `include/melon/algorithm/dijkstra.hpp` |
| `include/mcfcg/graph/dijkstra_workspace.h` (vertex status and per-vertex arrays) | `include/melon/algorithm/dijkstra.hpp` |

The following files follow melon's names and interfaces but were implemented
independently: `include/mcfcg/graph/d_ary_heap.h` (`updatable_d_ary_heap`),
`include/mcfcg/graph/static_digraph_builder.h` (`static_digraph_builder`) and
`include/mcfcg/graph/semiring.h` (`shortest_path_semiring`).

The comparison was made against melon at commit 5c0f8d2, the last one before
mcfcg's graph layer was added (2026-03-27). The evidence is recorded on
https://github.com/spoorendonk/mcfcg/issues/51.
