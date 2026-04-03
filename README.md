# mcfcg

Column generation solver for minimum-cost multicommodity flow (MCF). Supports path-based and tree-based Dantzig-Wolfe decompositions.

## Build

Requires C++23, CMake 3.20+, and zlib.

```bash
cmake -B build
cmake --build build -j$(nproc)
```

## Test

```bash
ctest --test-dir build --output-on-failure -j$(nproc)
```

## Usage

```bash
# CommaLab format
./build/mcfcg_cli data/commalab/grid/grid1

# TNTP transportation format (auto-detects trips file and demand coefficient)
./build/mcfcg_cli data/transportation/Winnipeg_net.tntp.gz

# Tree formulation
./build/mcfcg_cli data/commalab/grid/grid1 --formulation tree
```

## Instance data

Four instance families from public sources:

| Family | Format | Source |
|--------|--------|--------|
| Grid | CommaLab | [UniPi MCF benchmark](https://commalab.di.unipi.it/datasets/mmcf/) |
| Planar | CommaLab | [UniPi MCF benchmark](https://commalab.di.unipi.it/datasets/mmcf/) |
| Transportation | TNTP (gz) | [TransportationNetworks](https://github.com/bstabler/TransportationNetworks) |
| Intermodal | CommaLab (gz) | [Lienkamp & Schiffer 2024](https://doi.org/10.1016/j.ejor.2023.09.019) |

Download scripts are in `scripts/`.

## License

MIT
