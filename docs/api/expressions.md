# Expressions

Expressions are composable AST nodes that represent computations on column values. They are the building blocks for filters, computed columns, aggregations, and window functions.

---

## Constructor Functions

These are the primary entry points for creating expressions.

### col

::: pyfloe.col
    options:
      show_root_heading: true
      show_signature: true
      separate_signature: true
      docstring_section_style: list
      show_source: false
      heading_level: 4
      show_symbol_type_heading: true
      show_root_members_full_path: false
      summary: true

### lit

::: pyfloe.lit
    options:
      show_root_heading: true
      show_signature: true
      separate_signature: true
      docstring_section_style: list
      show_source: false
      heading_level: 4
      show_symbol_type_heading: true
      show_root_members_full_path: false
      summary: true

### when

::: pyfloe.when
    options:
      show_root_heading: true
      show_signature: true
      separate_signature: true
      docstring_section_style: list
      show_source: false
      heading_level: 4
      show_symbol_type_heading: true
      show_root_members_full_path: false
      summary: true

---

## Window Functions

### row_number

::: pyfloe.row_number
    options:
      show_root_heading: true
      show_signature: true
      separate_signature: true
      docstring_section_style: list
      show_source: false
      heading_level: 4
      show_symbol_type_heading: true
      show_root_members_full_path: false
      summary: true

### rank

::: pyfloe.rank
    options:
      show_root_heading: true
      show_signature: true
      separate_signature: true
      docstring_section_style: list
      show_source: false
      heading_level: 4
      show_symbol_type_heading: true
      show_root_members_full_path: false
      summary: true

### dense_rank

::: pyfloe.dense_rank
    options:
      show_root_heading: true
      show_signature: true
      separate_signature: true
      docstring_section_style: list
      show_source: false
      heading_level: 4
      show_symbol_type_heading: true
      show_root_members_full_path: false
      summary: true

---

## Expression Classes

### Expr

The base class for all expression types. Supports arithmetic (`+`, `-`, `*`, `/`), comparisons (`>`, `<`, `==`, etc.), logical operators (`&`, `|`, `~`), and method chaining.

::: pyfloe.Expr
    options:
      show_root_heading: true
      show_signature: true
      separate_signature: true
      docstring_section_style: list
      show_source: false
      heading_level: 4
      show_symbol_type_heading: true
      show_root_members_full_path: false
      summary: true
      show_symbol_type_toc: true
      members_order: source

### Col

::: pyfloe.Col
    options:
      show_root_heading: true
      show_signature: true
      separate_signature: true
      docstring_section_style: list
      show_source: false
      heading_level: 4
      show_symbol_type_heading: true
      show_root_members_full_path: false
      summary: true
      show_symbol_type_toc: true

### Lit

::: pyfloe.Lit
    options:
      show_root_heading: true
      show_signature: true
      separate_signature: true
      docstring_section_style: list
      show_source: false
      heading_level: 4
      show_symbol_type_heading: true
      show_root_members_full_path: false
      summary: true
      show_symbol_type_toc: true
