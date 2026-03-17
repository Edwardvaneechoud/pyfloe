# Expressions

Expressions are composable AST nodes. They enable type inference, optimization, and IDE autocomplete.

## Column references and literals

```python
from pyfloe import col, lit

col("amount")           # column reference
lit(42)                 # literal value
col("amount") * 1.1     # arithmetic (1.1 auto-wrapped as lit)
```

## Comparisons and logic

```python
col("amount") > 100
col("region") == "EU"
(col("amount") > 100) & (col("region") == "EU")   # AND
(col("x") < 0) | (col("x") > 100)                 # OR
~(col("active"))                                    # NOT
col("region").is_in(["EU", "APAC"])
col("value").is_null()
col("value").is_not_null()
```

## Arithmetic

```python
col("price") * col("quantity")
col("amount") + lit(100)
col("total") / col("count")
col("score") % 10
-col("delta")
100 + col("amount")                       # reverse ops work
```

## Type casting

```python
col("amount").cast(str)
col("id").cast(float)
```

## Conditional logic (CASE WHEN)

```python
from pyfloe import when

when(col("amount") > 200, "large") \
    .when(col("amount") > 100, "medium") \
    .otherwise("small")
```

## String methods

```python
col("name").str.upper()              # "ALICE"
col("name").str.lower()              # "alice"
col("name").str.strip()              # trim whitespace
col("name").str.title()              # "Alice"
col("name").str.len()                # 5
col("name").str.contains("li")       # True
col("name").str.startswith("Al")     # True
col("name").str.endswith("ce")       # True
col("name").str.replace("A", "a")    # "alice"
col("name").str.slice(0, 3)          # "Ali"
```

## Aggregations

```python
col("amount").sum()
col("amount").mean()
col("amount").min()
col("amount").max()
col("amount").count()
col("amount").n_unique()
col("amount").first()
col("amount").last()
```

Used with `group_by`:

```python
orders.group_by("region").agg(
    col("amount").sum().alias("total_revenue"),
    col("order_id").count().alias("order_count"),
    col("amount").mean().alias("avg_order"),
)
```

## Window functions

```python
from pyfloe import row_number, rank, dense_rank

# Ranking
row_number().over(partition_by="region", order_by="amount")
rank().over(partition_by="dept", order_by="salary")
dense_rank().over(order_by="score")

# Running aggregates
col("amount").cumsum().over(partition_by="region", order_by="date")
col("score").cummax().over(order_by="round")

# Lag / Lead
col("value").lag(1, default=0).over(partition_by="user", order_by="ts")
col("value").lead(1).over(order_by="ts")

# Window aggregation (partition total on every row)
col("amount").sum().over(partition_by="region")
```
