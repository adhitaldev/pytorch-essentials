#!/usr/bin/env python3
"""
generate_data.py

Generates a table of random data given:
  - columns:      a list of column names
  - value_ranges: a list of (min, max) tuples, one per column
  - num_rows:     the number of rows to generate

Usage examples
--------------
As a script (CLI):
    python generate_data.py \
        --columns "age,height,score" \
        --value_ranges "(18,65),(140,200),(0,100)" \
        --num_rows 10 \
        --output data.csv

As an importable function:
    from generate_data import generate_data
    df = generate_data(
        columns=["age", "height", "score"],
        value_ranges=[(18, 65), (140, 200), (0, 100)],
        num_rows=10,
    )
"""

import argparse
import ast
import random
import sys
import csv


def parse_columns(columns_arg):
    """
    Parse the `columns` argument into a list of column name strings.

    Accepts either:
      - an actual list (e.g. when called programmatically), or
      - a comma-separated string (e.g. "age,height,score") when called via CLI.
    """
    if isinstance(columns_arg, (list, tuple)):
        return [str(c).strip() for c in columns_arg]

    if isinstance(columns_arg, str):
        return [c.strip() for c in columns_arg.split(",") if c.strip()]

    raise TypeError(f"Unsupported type for columns: {type(columns_arg)}")


def parse_value_ranges(value_ranges_arg):
    """
    Parse the `value_ranges` argument into a list of (min, max) tuples.

    Accepts either:
      - an actual list of tuples/lists (programmatic use), or
      - a string like "(18,65),(140,200),(0,100)" (CLI use).
    """
    if isinstance(value_ranges_arg, (list, tuple)):
        parsed = []
        for item in value_ranges_arg:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                parsed.append((item[0], item[1]))
            else:
                raise ValueError(f"Invalid value range entry: {item!r}")
        return parsed

    if isinstance(value_ranges_arg, str):
        # Wrap the string so ast.literal_eval can parse it as a tuple of tuples.
        # "(18,65),(140,200)" -> "((18,65),(140,200))"
        wrapped = f"({value_ranges_arg})"
        try:
            result = ast.literal_eval(wrapped)
        except (SyntaxError, ValueError) as e:
            raise ValueError(f"Could not parse value_ranges string: {value_ranges_arg!r}") from e

        # Handle the single-range edge case: "(0,100)" -> ((0,100)) -> (0,100)
        if len(result) == 2 and all(isinstance(x, (int, float)) for x in result):
            result = (result,)

        parsed = []
        for item in result:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                parsed.append((item[0], item[1]))
            else:
                raise ValueError(f"Invalid value range entry: {item!r}")
        return parsed

    raise TypeError(f"Unsupported type for value_ranges: {type(value_ranges_arg)}")


def parse_num_rows(num_rows_arg):
    """Parse num_rows into an int."""
    try:
        num_rows = int(num_rows_arg)
    except (TypeError, ValueError) as e:
        raise ValueError(f"num_rows must be an integer, got: {num_rows_arg!r}") from e

    if num_rows < 0:
        raise ValueError("num_rows must be non-negative")

    return num_rows


def generate_data(columns, value_ranges, num_rows):
    """
    Generate `num_rows` rows of random data.

    Parameters
    ----------
    columns : list or str
        Column names (list) or comma-separated string of column names.
    value_ranges : list or str
        List of (min, max) tuples, or a string like "(0,10),(5,20)".
    num_rows : int or str
        Number of rows to generate.

    Returns
    -------
    list[dict]
        A list of row dicts, e.g. [{"age": 34, "height": 172}, ...]
    """
    col_names = parse_columns(columns)
    ranges = parse_value_ranges(value_ranges)
    n_rows = parse_num_rows(num_rows)

    if len(col_names) != len(ranges):
        raise ValueError(
            f"Length of columns ({len(col_names)}) must equal "
            f"length of value_ranges ({len(ranges)})"
        )

    rows = []
    for _ in range(n_rows):
        row = {}
        for name, (low, high) in zip(col_names, ranges):
            if isinstance(low, float) or isinstance(high, float):
                row[name] = round(random.uniform(low, high), 2)
            else:
                row[name] = random.randint(low, high)
        rows.append(row)

    return col_names, rows


def write_csv(col_names, rows, output_path):
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=col_names)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Generate random tabular data from columns, value_ranges, and num_rows."
    )
    parser.add_argument(
        "--columns",
        required=True,
        help='Comma-separated column names, e.g. "age,height,score"',
    )
    parser.add_argument(
        "--value_ranges",
        required=True,
        help='Comma-separated (min,max) tuples, e.g. "(18,65),(140,200),(0,100)"',
    )
    parser.add_argument(
        "--num_rows",
        required=True,
        help="Number of rows to generate",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to write a CSV file. If omitted, prints to stdout.",
    )

    args = parser.parse_args()

    try:
        col_names, rows = generate_data(args.columns, args.value_ranges, args.num_rows)
    except (ValueError, TypeError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    if args.output:
        write_csv(col_names, rows, args.output)
        print(f"Wrote {len(rows)} rows to {args.output}")
    else:
        writer = csv.DictWriter(sys.stdout, fieldnames=col_names)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()