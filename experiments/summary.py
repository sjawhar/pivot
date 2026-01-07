"""Summary: Best declarative options for matrix stages."""

print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    MATRIX STAGES: DECLARATIVE OPTIONS                         ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  GOAL: Single global function, different deps/outs per variant               ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  OPTION 1: Loop-based (already works, needs `name` param)                    ║
║  ─────────────────────────────────────────────────────────────────────────   ║
║                                                                              ║
║    for v1, deps in [...]:                                                    ║
║        for v2 in [...]:                                                      ║
║            @stage(name=f"process_{v1}_{v2}", deps=deps, ...)                 ║
║            def process(v1=v1, v2=v2):  # Default args capture values!        ║
║                ...                                                           ║
║                                                                              ║
║    ✅ Works today (just add `name` param)                                    ║
║    ✅ Full Python flexibility                                                 ║
║    ⚠️  Requires default args pattern to avoid closure bug                    ║
║    ⚠️  Function redefined each iteration (shared fingerprint)                ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  OPTION 2: Dict comprehension in `variants` param                            ║
║  ─────────────────────────────────────────────────────────────────────────   ║
║                                                                              ║
║    variant_deps = {"v1": base_deps, "v2": [*base_deps, "extra.csv"]}         ║
║                                                                              ║
║    @stage(variants={                                                         ║
║        f"{v1}_{v2}": {                                                       ║
║            'deps': variant_deps[v1],                                         ║
║            'outs': [f"{v1}_out.csv", f"{v2}_out.csv"],                       ║
║            'params': Params(var_three=v1 + v2),                              ║
║        }                                                                     ║
║        for v1 in ["value1", "value2"]                                        ║
║        for v2 in ["value3", "value4"]                                        ║
║    })                                                                        ║
║    def process(variant: str):                                                ║
║        ...                                                                   ║
║                                                                              ║
║    ✅ Single global function                                                  ║
║    ✅ Declarative (all config in decorator)                                   ║
║    ✅ Different deps/outs per variant                                         ║
║    ✅ Pre-configured params instances                                         ║
║    ✅ No closure bugs (comprehension evaluates immediately)                   ║
║    ⚠️  Function only gets variant name (not individual dimensions)           ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  OPTION 3: List with `bind` for multi-value functions                        ║
║  ─────────────────────────────────────────────────────────────────────────   ║
║                                                                              ║
║    @stage(variants=[                                                         ║
║        {                                                                     ║
║            'name': f"{v1}_{v2}",                                             ║
║            'deps': variant_deps[v1],                                         ║
║            'outs': [f"{v1}_out.csv"],                                        ║
║            'params': Params(...),                                            ║
║            'bind': {'var_one': v1, 'var_two': v2},  # Multiple values!       ║
║        }                                                                     ║
║        for v1 in ["value1", "value2"]                                        ║
║        for v2 in ["value3", "value4"]                                        ║
║    ])                                                                        ║
║    def process(var_one: str, var_two: str):  # Gets both dimensions!         ║
║        ...                                                                   ║
║                                                                              ║
║    ✅ Function gets individual dimension values                               ║
║    ✅ Most explicit and flexible                                              ║
║    ✅ Type hints work correctly                                               ║
║    ⚠️  Slightly more verbose                                                 ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  RECOMMENDATION                                                              ║
║  ─────────────────────────────────────────────────────────────────────────   ║
║                                                                              ║
║  Implement OPTION 2 (variants dict) as the primary API:                      ║
║                                                                              ║
║    @stage(variants={...})                                                    ║
║    def process(variant: str): ...                                            ║
║                                                                              ║
║  With optional `bind` support for multi-value functions:                     ║
║                                                                              ║
║    @stage(variants=[{'name': ..., 'bind': {...}, ...}])                      ║
║    def process(v1: str, v2: str): ...                                        ║
║                                                                              ║
║  Keep loop-based (Option 1) as documented alternative for edge cases.        ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")
