# Floe

The `Floe` is the central object in pyfloe. It represents a lazy dataframe — operations build a query plan without executing it. Data flows only when you trigger evaluation with `.collect()`, `.to_pylist()`, `.to_csv()`, or similar methods.

---

## Floe

::: pyfloe.Floe
    options:
      show_root_heading: true
      show_signature: true
      separate_signature: true
      show_source: false
      heading_level: 3
      show_symbol_type_heading: true
      show_root_members_full_path: false
      summary: true
      show_symbol_type_toc: true
      members_order: source
      docstring_section_style: list

---

## TypedFloe

The `TypedFloe` wraps a Floe with a known row type (a `TypedDict`), enabling static type checkers and IDEs to infer the shape of results from `.to_pylist()`.

::: pyfloe.TypedFloe
    options:
      show_root_heading: true
      show_signature: true
      separate_signature: true
      show_source: false
      heading_level: 3
      show_symbol_type_heading: true
      show_root_members_full_path: false
      summary: true
      show_symbol_type_toc: true
      docstring_section_style: list

---

## GroupByBuilder

The `GroupByBuilder` is created by calling `Floe.group_by()`. Call `.agg()` on it to specify aggregation expressions and produce the grouped result.

::: pyfloe.GroupByBuilder
    options:
      show_root_heading: true
      show_signature: true
      separate_signature: true
      show_source: false
      heading_level: 3
      show_symbol_type_heading: true
      show_root_members_full_path: false
      summary: true
      show_symbol_type_toc: true
      docstring_section_style: list
