dict(
    deps=["{var_one}.csv", "{var_two}.csv"],
    outs=["{var_one}_processed.csv", "{var_two}_processed.csv"],
    matrix={
        "var_one": ["value1", {"value2": {"deps": ["value3.csv", "value4.csv"], "outs": ["value3_processed.csv", "value4_processed.csv"]}}],
    }
)