# File I/O

## Reading files

All readers are lazy — data is not loaded until you trigger evaluation.

```python
from pyfloe import read_csv, read_tsv, read_jsonl, read_json, read_fixed_width

ff = read_csv("data.csv")       # lazy, types + datetime inferred
ff = read_tsv("data.tsv")
ff = read_jsonl("events.jsonl")  # streaming
ff = read_json("cities.json")    # parsed at read time
ff = read_fixed_width("report.txt", widths=[10, 20, 8, 12], has_header=True)
```

### Parquet (optional — requires pyarrow)

```python
from pyfloe import read_parquet

ff = read_parquet("data.parquet")
ff = read_parquet("data.parquet", columns=["id", "score"])
```

## Writing files

Writers stream from the query plan with constant memory.

```python
ff.to_csv("out.csv")           # streams from plan, constant memory
ff.to_tsv("out.tsv")
ff.to_jsonl("out.jsonl")
ff.to_json("out.json", indent=2)
ff.to_parquet("out.parquet")   # requires pyarrow
```
