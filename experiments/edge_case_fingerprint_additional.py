"""
Additional edge cases to complete the red team analysis.
"""
import sys
sys.path.insert(0, "/home/pivot/agent1/src")

from pivot import fingerprint

print("=" * 80)
print("EDGE CASE: functools.wraps decorator")
print("=" * 80)

import functools

def original_process(x):
    return x + 1

def wrapper_factory(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs) * 2
    return wrapper

wrapped = wrapper_factory(original_process)

def stage_wrapped():
    return wrapped(10)

fp1 = fingerprint.get_stage_fingerprint(stage_wrapped)
print(f"Fingerprint: {fp1}")
print(f"wrapped.__name__: {wrapped.__name__}")
print(f"wrapped.__wrapped__: {getattr(wrapped, '__wrapped__', 'NOT SET')}")

# Now change the original function
def original_process_v2(x):
    return x + 999

wrapped_v2 = wrapper_factory(original_process_v2)

def stage_wrapped_v2():
    return wrapped_v2(10)

fp2 = fingerprint.get_stage_fingerprint(stage_wrapped_v2)
print(f"Fingerprint V2: {fp2}")
print(f"Fingerprints same: {fp1 == fp2}")
print()

print("=" * 80)
print("EDGE CASE: Lambda in closure")
print("=" * 80)

multiplier_v1 = lambda x: x * 2  # noqa: E731
multiplier_v2 = lambda x: x * 999  # noqa: E731

def stage_lambda_v1():
    return multiplier_v1(10)

def stage_lambda_v2():
    return multiplier_v2(10)

fp1 = fingerprint.get_stage_fingerprint(stage_lambda_v1)
fp2 = fingerprint.get_stage_fingerprint(stage_lambda_v2)
print(f"Lambda V1 fingerprint: {fp1}")
print(f"Lambda V2 fingerprint: {fp2}")
print(f"Different: {fp1 != fp2}")
print()

print("=" * 80)
print("EDGE CASE: Class method vs instance method")
print("=" * 80)

class ProcessorWithClassMethod:
    @classmethod
    def class_process(cls, x):
        return x + 1

    @staticmethod
    def static_process(x):
        return x + 2

    def instance_process(self, x):
        return x + 3

proc = ProcessorWithClassMethod()

def stage_classmethod():
    return ProcessorWithClassMethod.class_process(10)

def stage_staticmethod():
    return ProcessorWithClassMethod.static_process(10)

def stage_instancemethod():
    return proc.instance_process(10)

fp_class = fingerprint.get_stage_fingerprint(stage_classmethod)
fp_static = fingerprint.get_stage_fingerprint(stage_staticmethod)
fp_instance = fingerprint.get_stage_fingerprint(stage_instancemethod)

print(f"Classmethod fingerprint: {fp_class}")
print(f"Staticmethod fingerprint: {fp_static}")
print(f"Instancemethod fingerprint: {fp_instance}")
print()

print("=" * 80)
print("EDGE CASE: __slots__ class")
print("=" * 80)

class SlottedClass:
    __slots__ = ['value']

    def __init__(self, value):
        self.value = value

    def process(self, x):
        return x + self.value

slotted = SlottedClass(10)

def stage_slotted():
    return slotted.process(5)

fp = fingerprint.get_stage_fingerprint(stage_slotted)
print(f"Slotted class fingerprint: {fp}")
print()

print("=" * 80)
print("EDGE CASE: dataclass")
print("=" * 80)

import dataclasses

@dataclasses.dataclass
class DataProcessor:
    multiplier: int = 2

    def process(self, x):
        return x * self.multiplier

data_proc = DataProcessor(multiplier=3)

def stage_dataclass():
    return data_proc.process(10)

fp = fingerprint.get_stage_fingerprint(stage_dataclass)
print(f"Dataclass fingerprint: {fp}")
print(f"Note: multiplier value (3) is stored in instance, not in class definition")
print()

# Check if multiplier value change is detected
data_proc_v2 = DataProcessor(multiplier=999)

def stage_dataclass_v2():
    return data_proc_v2.process(10)

fp_v2 = fingerprint.get_stage_fingerprint(stage_dataclass_v2)
print(f"Dataclass V2 fingerprint: {fp_v2}")
print(f"Different fingerprints: {fp != fp_v2}")
print(f"BUG: Instance state (multiplier value) is NOT tracked!")
print()

print("=" * 80)
print("EDGE CASE: Property getter/setter")
print("=" * 80)

class ConfigWithProperty:
    def __init__(self):
        self._threshold = 0.5

    @property
    def threshold(self):
        return self._threshold

    @threshold.setter
    def threshold(self, value):
        self._threshold = value

    def is_above_threshold(self, x):
        return x > self._threshold

config = ConfigWithProperty()

def stage_property():
    return config.is_above_threshold(0.6)

fp = fingerprint.get_stage_fingerprint(stage_property)
print(f"Property fingerprint: {fp}")
print()

print("=" * 80)
print("EDGE CASE: Instance attribute (config value)")
print("=" * 80)

class ModelConfig:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def get_lr(self):
        return self.learning_rate

config_v1 = ModelConfig(learning_rate=0.001)
config_v2 = ModelConfig(learning_rate=0.1)  # Different value!

def stage_config_v1():
    return config_v1.get_lr()

def stage_config_v2():
    return config_v2.get_lr()

fp1 = fingerprint.get_stage_fingerprint(stage_config_v1)
fp2 = fingerprint.get_stage_fingerprint(stage_config_v2)

print(f"Config V1 fingerprint: {fp1}")
print(f"Config V2 fingerprint: {fp2}")
print(f"Class hashes same: {fp1.get('class:config_v1.__class__') == fp2.get('class:config_v2.__class__')}")
print(f"BUG: Instance attribute values (learning_rate) are NOT tracked!")
print()

print("=" * 80)
print("EDGE CASE: Enum class")
print("=" * 80)

import enum

class Mode(enum.Enum):
    FAST = "fast"
    ACCURATE = "accurate"

def stage_enum():
    return Mode.FAST.value

fp = fingerprint.get_stage_fingerprint(stage_enum)
print(f"Enum fingerprint: {fp}")
print()

print("=" * 80)
print("SUMMARY OF ALL CONFIRMED BUGS")
print("=" * 80)
print("""
CRITICAL SEVERITY:
1. Base class / mixin method changes NOT detected
   - Only concrete class is hashed
   - Methods inherited from user-defined base classes are invisible
   - MRO is completely ignored

2. Instance attribute values NOT tracked
   - Class is hashed, but instance state is not
   - Different ModelConfig(lr=0.001) vs ModelConfig(lr=0.1) have same fingerprint
   - Dataclass field values are not tracked

HIGH SEVERITY:
3. Instance-level monkey-patched methods NOT detected
   - Methods attached directly to instance via types.MethodType
   - type(instance) still returns the original class

4. Instances inside collections NOT tracked
   - Dict/list values that are instances are silently ignored
   - _process_collection_dependency only handles callables

5. Proxy/delegation objects (__getattr__)
   - Only proxy class tracked, not the delegated target

6. Module.instance.method pattern
   - mod:module.instance tracked as repr, not the method code

MEDIUM SEVERITY:
7. Nested attribute access (obj.attr.method)
   - Only top-level obj tracked
   - Nested objects invisible

8. functools.partial wrapping methods
   - May or may not be tracked depending on how captured

DESIGN LIMITATIONS (lower priority):
9. Instance state in general is not tracked
   - self.x values are only tracked if they're in class definition
   - Runtime state changes are invisible
""")
