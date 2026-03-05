# Politifact Pattern Report (Graph 0)

- Source dir: `/home/seojm/GPM/patterns/politifact/128_8_1_0.1`
- Shapes: ptn=(128, 314, 9), nid=(128, 314, 9), eid=(128, 314, 8)
- Graph idx: 0
- Patterns: 128, Walk length: 8 (nodes per walk: 9)
- Constant walks: 117/128 (91.41%)
- Unique nodes per pattern: min=1, median=1, max=5

## Files

- `pattern_table.csv`: per-pattern table
- `summary.json`: aggregate stats
- `heatmap_local_walk.txt`: ASCII heatmap by local node id
- `heatmap_transition.txt`: ASCII heatmap by move/stay transition

## Top 10 Informative Patterns (by unique_nodes)

- pattern 106: unique_nodes=5, repeat_ratio=0.4444
  - local: 420 -> 481 -> 482 -> 483 -> 484 -> 484 -> 484 -> 484 -> 484
  - global: 420 -> 481 -> 482 -> 483 -> 484 -> 484 -> 484 -> 484 -> 484
  - edges: 894 -> 973 -> 975 -> 977 -> 978 -> 978 -> 978 -> 978
- pattern 52: unique_nodes=4, repeat_ratio=0.5556
  - local: 481 -> 481 -> 481 -> 482 -> 483 -> 484 -> 484 -> 484 -> 484
  - global: 481 -> 481 -> 481 -> 482 -> 483 -> 484 -> 484 -> 484 -> 484
  - edges: 972 -> 972 -> 973 -> 975 -> 977 -> 978 -> 978 -> 978
- pattern 73: unique_nodes=3, repeat_ratio=0.6667
  - local: 481 -> 481 -> 482 -> 482 -> 482 -> 482 -> 483 -> 483 -> 483
  - global: 481 -> 481 -> 482 -> 482 -> 482 -> 482 -> 483 -> 483 -> 483
  - edges: 972 -> 973 -> 974 -> 974 -> 974 -> 975 -> 976 -> 976
- pattern 9: unique_nodes=2, repeat_ratio=0.7778
  - local: 483 -> 484 -> 484 -> 484 -> 484 -> 484 -> 484 -> 484 -> 484
  - global: 483 -> 484 -> 484 -> 484 -> 484 -> 484 -> 484 -> 484 -> 484
  - edges: 977 -> 978 -> 978 -> 978 -> 978 -> 978 -> 978 -> 978
- pattern 23: unique_nodes=2, repeat_ratio=0.7778
  - local: 185 -> 467 -> 467 -> 467 -> 467 -> 467 -> 467 -> 467 -> 467
  - global: 185 -> 467 -> 467 -> 467 -> 467 -> 467 -> 467 -> 467 -> 467
  - edges: 649 -> 954 -> 954 -> 954 -> 954 -> 954 -> 954 -> 954
- pattern 26: unique_nodes=2, repeat_ratio=0.7778
  - local: 8 -> 8 -> 451 -> 451 -> 451 -> 451 -> 451 -> 451 -> 451
  - global: 8 -> 8 -> 451 -> 451 -> 451 -> 451 -> 451 -> 451 -> 451
  - edges: 458 -> 459 -> 935 -> 935 -> 935 -> 935 -> 935 -> 935
- pattern 36: unique_nodes=2, repeat_ratio=0.7778
  - local: 67 -> 460 -> 460 -> 460 -> 460 -> 460 -> 460 -> 460 -> 460
  - global: 67 -> 460 -> 460 -> 460 -> 460 -> 460 -> 460 -> 460 -> 460
  - edges: 527 -> 944 -> 944 -> 944 -> 944 -> 944 -> 944 -> 944
- pattern 39: unique_nodes=2, repeat_ratio=0.7778
  - local: 477 -> 478 -> 478 -> 478 -> 478 -> 478 -> 478 -> 478 -> 478
  - global: 477 -> 478 -> 478 -> 478 -> 478 -> 478 -> 478 -> 478 -> 478
  - edges: 968 -> 969 -> 969 -> 969 -> 969 -> 969 -> 969 -> 969
- pattern 47: unique_nodes=2, repeat_ratio=0.7778
  - local: 471 -> 471 -> 472 -> 472 -> 472 -> 472 -> 472 -> 472 -> 472
  - global: 471 -> 471 -> 472 -> 472 -> 472 -> 472 -> 472 -> 472 -> 472
  - edges: 960 -> 961 -> 962 -> 962 -> 962 -> 962 -> 962 -> 962
- pattern 90: unique_nodes=2, repeat_ratio=0.7778
  - local: 152 -> 466 -> 466 -> 466 -> 466 -> 466 -> 466 -> 466 -> 466
  - global: 152 -> 466 -> 466 -> 466 -> 466 -> 466 -> 466 -> 466 -> 466
  - edges: 615 -> 953 -> 953 -> 953 -> 953 -> 953 -> 953 -> 953

## Most Frequent Local Nodes (top 15)

- local_node 411: 27
- local_node 26: 18
- local_node 155: 18
- local_node 86: 18
- local_node 167: 18
- local_node 220: 18
- local_node 276: 18
- local_node 272: 18
- local_node 18: 18
- local_node 465: 18
- local_node 85: 18
- local_node 230: 18
- local_node 413: 18
- local_node 484: 17
- local_node 467: 17
