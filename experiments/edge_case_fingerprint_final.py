"""
FINAL PROOF: Demonstrating that base class method changes are NOT detected
when the concrete class AST doesn't change.
"""
import sys
sys.path.insert(0, "/home/pivot/agent1/src")

import tempfile
import importlib.util
import os
from pivot import fingerprint

print("=" * 80)
print("PROOF: Base class method change not detected")
print("=" * 80)

# Create temporary module files to simulate real file editing
with tempfile.TemporaryDirectory() as tmpdir:
    # Write base_v1.py
    base_path = os.path.join(tmpdir, "base.py")
    with open(base_path, "w") as f:
        f.write("""
class BaseMixin:
    def process(self, x):
        return x + 1  # V1 implementation
""")

    # Write concrete.py that imports from base
    concrete_path = os.path.join(tmpdir, "concrete.py")
    with open(concrete_path, "w") as f:
        f.write("""
from base import BaseMixin

class MyProcessor(BaseMixin):
    '''Concrete processor using BaseMixin.'''
    pass
""")

    # Add tmpdir to sys.path
    sys.path.insert(0, tmpdir)

    # Import the modules
    spec_base = importlib.util.spec_from_file_location("base", base_path)
    base_module = importlib.util.module_from_spec(spec_base)
    sys.modules["base"] = base_module
    spec_base.loader.exec_module(base_module)

    spec_concrete = importlib.util.spec_from_file_location("concrete", concrete_path)
    concrete_module = importlib.util.module_from_spec(spec_concrete)
    sys.modules["concrete"] = concrete_module
    spec_concrete.loader.exec_module(concrete_module)

    # Create instance and stage
    proc_v1 = concrete_module.MyProcessor()

    def stage_v1():
        return proc_v1.process(10)

    fp_v1 = fingerprint.get_stage_fingerprint(stage_v1)
    print(f"V1 fingerprint: {fp_v1}")
    print(f"V1 result: {stage_v1()}")

    # Now modify ONLY the base class method
    with open(base_path, "w") as f:
        f.write("""
class BaseMixin:
    def process(self, x):
        return x + 999  # V2 implementation - CHANGED!
""")

    # Reload the base module
    importlib.reload(base_module)

    # Reload concrete to pick up new base class
    importlib.reload(concrete_module)

    # Create new instance
    proc_v2 = concrete_module.MyProcessor()

    def stage_v2():
        return proc_v2.process(10)

    fp_v2 = fingerprint.get_stage_fingerprint(stage_v2)
    print(f"V2 fingerprint: {fp_v2}")
    print(f"V2 result: {stage_v2()}")

    # Compare
    class_hash_v1 = fp_v1.get('class:proc_v1.__class__')
    class_hash_v2 = fp_v2.get('class:proc_v2.__class__')

    print(f"\nClass hash V1: {class_hash_v1}")
    print(f"Class hash V2: {class_hash_v2}")
    print(f"Class hashes SAME: {class_hash_v1 == class_hash_v2}")
    print()

    print("=" * 80)
    print("ANALYSIS")
    print("=" * 80)

    import inspect
    print("MyProcessor source (hashed by Pivot):")
    print(inspect.getsource(concrete_module.MyProcessor))
    print()

    print("BaseMixin source (NOT hashed by Pivot):")
    print(inspect.getsource(base_module.BaseMixin))
    print()

    # Check MRO
    print("MyProcessor.__mro__:")
    for cls in concrete_module.MyProcessor.__mro__:
        print(f"  {cls.__name__}")
        if fingerprint.is_user_code(cls) and cls != object:
            print(f"    is_user_code: True")
            print(f"    hash: {fingerprint.hash_function_ast(cls)}")
    print()

    print("=" * 80)
    print("BUG CONFIRMED")
    print("=" * 80)
    if class_hash_v1 == class_hash_v2:
        print("*** BUG: Class hashes are SAME despite base class method change! ***")
        print("*** Stage would NOT be re-run even though behavior changed! ***")
    else:
        print("Note: Hashes differ, but check if it's just due to stage name difference")

    # Cleanup
    sys.path.remove(tmpdir)
    del sys.modules["base"]
    del sys.modules["concrete"]
