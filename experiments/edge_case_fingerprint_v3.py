"""
Final precise test: Confirming inherited method changes are NOT detected.
"""
import sys
sys.path.insert(0, "/home/pivot/agent1/src")

from pivot import fingerprint
import inspect

print("=" * 80)
print("PRECISE TEST: Same concrete class, different inherited method implementation")
print("=" * 80)

# This simulates what happens when you edit a base class method
# The concrete class doesn't change at all, but the method it uses does

# First, let's verify the hash only covers the class source, not inherited methods
class BaseV1:
    def inherited_method(self, x):
        return x + 1  # V1 implementation

class ConcreteA(BaseV1):
    def own_method(self, x):
        return x * 2

instance_a = ConcreteA()

def stage_a():
    return instance_a.inherited_method(10)

fp_a = fingerprint.get_stage_fingerprint(stage_a)
print(f"Stage A fingerprint: {fp_a}")
print(f"\nConcreteA source (what gets hashed):")
print(inspect.getsource(ConcreteA))

# Now create a class with identical source but different parent
class BaseV2:
    def inherited_method(self, x):
        return x + 999  # V2 implementation - DIFFERENT!

# ConcreteB has IDENTICAL source code to ConcreteA
# (same class body, just different parent)
class ConcreteB(BaseV2):
    def own_method(self, x):
        return x * 2

instance_b = ConcreteB()

def stage_b():
    return instance_b.inherited_method(10)

fp_b = fingerprint.get_stage_fingerprint(stage_b)
print(f"\nStage B fingerprint: {fp_b}")
print(f"\nConcreteB source (what gets hashed):")
print(inspect.getsource(ConcreteB))

# The key comparison: do the class hashes differ?
hash_a = fp_a.get('class:instance_a.__class__')
hash_b = fp_b.get('class:instance_b.__class__')
print(f"\nConcreteA hash: {hash_a}")
print(f"ConcreteB hash: {hash_b}")
print(f"Hashes same: {hash_a == hash_b}")
print(f"BUT inherited_method implementations are DIFFERENT!")
print()

print("=" * 80)
print("VERIFICATION: What hash_function_ast actually hashes")
print("=" * 80)

hash_concrete_a = fingerprint.hash_function_ast(ConcreteA)
hash_concrete_b = fingerprint.hash_function_ast(ConcreteB)
hash_base_v1 = fingerprint.hash_function_ast(BaseV1)
hash_base_v2 = fingerprint.hash_function_ast(BaseV2)

print(f"hash(ConcreteA): {hash_concrete_a}")
print(f"hash(ConcreteB): {hash_concrete_b}")
print(f"hash(BaseV1): {hash_base_v1}")
print(f"hash(BaseV2): {hash_base_v2}")
print()

# The concrete classes have different hashes because their SOURCE includes
# the base class name in the class definition line!
print("Note: ConcreteA and ConcreteB have different hashes because")
print("the AST includes 'BaseV1' vs 'BaseV2' in the class definition.")
print("BUT the actual BASE CLASS CODE (BaseV1/BaseV2 bodies) is NOT hashed.")
print()

print("=" * 80)
print("REAL-WORLD SCENARIO: Editing a mixin/base class method")
print("=" * 80)

# In real-world usage, you wouldn't rename the base class
# You'd edit a method in the SAME base class file
# Let's simulate this more precisely

# Create a module where we can modify the base class
import types

base_module = types.ModuleType("base_module")
base_module.__file__ = "/home/pivot/agent1/src/base_module.py"
sys.modules["base_module"] = base_module

# Define V1 of the base class
exec("""
class BaseMixin:
    def process(self, x):
        return x + 1
""", base_module.__dict__)

# Import from the module
from base_module import BaseMixin as BaseMixinV1

# Create concrete class
class MyProcessor(BaseMixinV1):
    """Concrete processor using BaseMixin."""
    pass

proc = MyProcessor()

def stage_real():
    return proc.process(10)

fp1 = fingerprint.get_stage_fingerprint(stage_real)
print(f"Fingerprint before base class edit: {fp1}")

# Now simulate editing the base class file (method changes)
exec("""
class BaseMixin:
    def process(self, x):
        return x + 999  # CHANGED!
""", base_module.__dict__)

# The instance `proc` still uses the OLD BaseMixin because Python
# doesn't hot-reload. But let's test if fingerprinting would catch
# the change if we created a new instance.

from base_module import BaseMixin as BaseMixinV2

class MyProcessorV2(BaseMixinV2):
    """Same concrete class, different base."""
    pass

proc_v2 = MyProcessorV2()

def stage_real_v2():
    return proc_v2.process(10)

fp2 = fingerprint.get_stage_fingerprint(stage_real_v2)
print(f"Fingerprint after base class edit: {fp2}")
print(f"Fingerprints same: {fp1 == fp2}")
print()

print("=" * 80)
print("DEMONSTRATING THE EXACT BUG")
print("=" * 80)

# The bug is clearer with explicit MRO
print(f"MyProcessor MRO: {[c.__name__ for c in MyProcessor.__mro__]}")
print(f"MyProcessor.process is actually: {MyProcessor.process}")
print(f"Defined in: {MyProcessor.process.__qualname__}")
print()

# Only MyProcessor's source is hashed
print("Only MyProcessor's source is hashed:")
print(inspect.getsource(MyProcessor))
print()

# BaseMixin source is NOT hashed, despite providing the process method!
print("BaseMixinV1's source is NOT included in the hash:")
print(inspect.getsource(BaseMixinV1))
print()

print("=" * 80)
print("BUG SEVERITY: HIGH")
print("=" * 80)
print("""
IMPACT:
- Any method inherited from a base class/mixin is NOT tracked
- Editing a base class method will NOT trigger stage re-execution
- This is common in real codebases (Strategy pattern, Mixins, ABC, etc.)

EXAMPLE SCENARIO:
1. User has `class BasePreprocessor` with `clean_data()` method
2. Stage uses `class MyPreprocessor(BasePreprocessor)`
3. User fixes a bug in `BasePreprocessor.clean_data()`
4. Pivot fingerprint doesn't change because only MyPreprocessor is hashed
5. Stage is NOT re-run, user gets stale results from buggy code

WORKAROUND (for users):
- Override inherited methods in concrete class even if just calling super()
- This forces the method to appear in the concrete class's AST

FIX (for Pivot):
- When hashing a class, traverse the MRO and hash all user-defined base classes
- Only include base classes where `is_user_code(base)` is True
""")
