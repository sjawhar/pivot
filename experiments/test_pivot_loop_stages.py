"""Test loop-based stage registration with actual Pivot."""

import sys
import os
import tempfile

# Change to a temp directory for test outputs
test_dir = tempfile.mkdtemp()
os.chdir(test_dir)

from pydantic import BaseModel
from pivot import stage
from pivot.registry import REGISTRY
from pivot import executor


class Params(BaseModel):
    var_three: str = "value3"


# Create test input files
for f in ["foo.csv", "bar.csv", "value5.csv", "value6.csv"]:
    with open(f, "w") as fp:
        fp.write(f"content of {f}\n")


print("=" * 60)
print("TEST: User's exact syntax with Pivot")
print("=" * 60)

# Clear any previous registrations
REGISTRY._stages.clear()

base_deps = ["foo.csv", "bar.csv"]
for var_one, var_deps in [
    ("value1", base_deps),
    ("value2", [*base_deps, "value5.csv", "value6.csv"]),
]:
    for var_two in ["value3", "value4"]:
        # Capture loop variables in default args!
        @stage(
            name=f"process_{var_one}_{var_two}",
            deps=var_deps,
            outs=[f"{var_one}_{var_two}_output.txt"],
        )
        def process(var_one=var_one, var_two=var_two):
            # Default args capture VALUE at definition time
            output_file = f"{var_one}_{var_two}_output.txt"
            with open(output_file, "w") as f:
                f.write(f"Processed {var_one} + {var_two}\n")
            print(f"  Executed: process_{var_one}_{var_two}")


print(f"\nRegistered stages: {list(REGISTRY._stages.keys())}")

# Check fingerprints are different
print("\nFingerprint check:")
fingerprints = {}
for name, spec in REGISTRY._stages.items():
    fp = spec['fingerprint']  # It's a dict with 'code' and 'deps' keys
    fp_str = str(fp)[:40]
    fingerprints[name] = fp_str
    print(f"  {name}: {fp_str}...")

unique_fps = len(set(fingerprints.values()))
print(f"\nUnique fingerprints: {unique_fps}/{len(fingerprints)}")

# Run the pipeline
print("\n" + "=" * 60)
print("Running pipeline with Pivot executor...")
print("=" * 60)

try:
    results = executor.run()
    print(f"\nResults: {len(results)} stages executed")
    for name, result in results.items():
        print(f"  {name}: {result.status}")
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()

# Check outputs
print("\n" + "=" * 60)
print("Checking outputs:")
print("=" * 60)
for f in os.listdir(test_dir):
    if f.endswith("_output.txt"):
        with open(f) as fp:
            content = fp.read().strip()
        print(f"  {f}: {content}")

print(f"\nTest directory: {test_dir}")
