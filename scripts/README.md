# Scripts

## Instance Preparation

### `download_commalab.sh`

Downloads CommaLab/UniPi benchmark instances into `data/commalab/`.
These are the Grid and Planar instance families used in the paper.

### `prepare_intermodal.sh`

End-to-end pipeline for intermodal instances (SUBWAY, BUS, SBT families):

1. Clones the [tumBAIS intermodal repo](https://github.com/tumBAIS/intermodalTransportationNetworksCG)
2. Fetches LFS data files (network XML, schedule XML)
3. Generates raw instances via `generate_instances.py`
4. Cleans instances with `mcfcg_clean` (removes unreachable commodities)

**Prerequisites:** Python 3 with networkx, pandas, numpy, geopandas, shapely, lxml.
Build `mcfcg_clean` first: `cmake --build build -j$(nproc)`

### `generate_instances.py`

Generates intermodal MMCF instances from the tumBAIS repo data.
Called by `prepare_intermodal.sh`; can also be run standalone.

```
python3 generate_instances.py --repo data/intermodal-repo --output data/intermodal/raw \
    --seeds 0 --modes subway bus sbt
```

## Instance Families

| Family | Format | Instances | Source |
|--------|--------|-----------|--------|
| Grid | CommaLab | 15 (grid1-grid15) | [CommaLab](https://commalab.di.unipi.it/) |
| Planar | CommaLab | 10 (planar30-planar2500) | [CommaLab](https://commalab.di.unipi.it/) |
| SUBWAY | CommaLab | 4 (4 passenger counts, seed 0) | tumBAIS intermodal repo |
| BUS | CommaLab | 5 (5 passenger counts, seed 0) | tumBAIS intermodal repo |
| SBT | CommaLab | 5 (5 passenger counts, seed 0) | tumBAIS intermodal repo |
| Transportation | TNTP | 9 cities | [TransportationNetworks](https://github.com/bstabler/TransportationNetworks) |

### Transportation Cities

TNTP instances are committed as gzipped files in `data/transportation/`.
Each city has a demand coefficient used to scale raw OD demands:

| City | Coefficient |
|------|------------|
| Austin | 6.0 |
| Barcelona | 5050.0 |
| BerlinCenter | 0.5 |
| Birmingham | 0.9 |
| ChicagoRegional | 4.1 |
| ChicagoSketch | 2.4 |
| Philadelphia | 7.0 |
| Sydney | 1.9 |
| Winnipeg | 2000.0 |

Pass TNTP files to the CLI (plain or gzipped):
```
./build/mcfcg_cli data/transportation/Winnipeg_net.tntp.gz
./build/mcfcg_cli path/to/CityName_net.tntp
```
The CLI auto-detects the format, derives the trips path, and looks up the coefficient.
