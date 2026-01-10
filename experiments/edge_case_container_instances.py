"""
Detailed test for instances inside containers.
"""
import sys
sys.path.insert(0, "/home/pivot/agent1/src")

from pivot import fingerprint

print("=" * 80)
print("BUG PROOF: Instances inside containers are NOT tracked")
print("=" * 80)

class WorkerV1:
    def work(self, x):
        return x + 1  # V1: adds 1

class WorkerV2:
    def work(self, x):
        return x + 999  # V2: adds 999

# Test 1: Instance in dict
print("\n--- Test 1: Instance in dict ---")
workers_dict = {"main": WorkerV1()}

def stage_dict():
    return workers_dict["main"].work(10)

fp1 = fingerprint.get_stage_fingerprint(stage_dict)
print(f"Fingerprint with WorkerV1 in dict: {fp1}")
print(f"Result: {stage_dict()}")

# Replace with different implementation
workers_dict["main"] = WorkerV2()

fp2 = fingerprint.get_stage_fingerprint(stage_dict)
print(f"Fingerprint with WorkerV2 in dict: {fp2}")
print(f"Result: {stage_dict()}")
print(f"Fingerprints SAME (BUG!): {fp1 == fp2}")

# Test 2: Instance in list
print("\n--- Test 2: Instance in list ---")
workers_list = [WorkerV1()]

def stage_list():
    return workers_list[0].work(10)

fp1 = fingerprint.get_stage_fingerprint(stage_list)
print(f"Fingerprint with WorkerV1 in list: {fp1}")
print(f"Result: {stage_list()}")

workers_list[0] = WorkerV2()

fp2 = fingerprint.get_stage_fingerprint(stage_list)
print(f"Fingerprint with WorkerV2 in list: {fp2}")
print(f"Result: {stage_list()}")
print(f"Fingerprints SAME (BUG!): {fp1 == fp2}")

# Test 3: Instance in tuple (immutable, so different tuples)
print("\n--- Test 3: Instance in tuple ---")
workers_tuple_v1 = (WorkerV1(),)
workers_tuple_v2 = (WorkerV2(),)

def stage_tuple_v1():
    return workers_tuple_v1[0].work(10)

def stage_tuple_v2():
    return workers_tuple_v2[0].work(10)

fp1 = fingerprint.get_stage_fingerprint(stage_tuple_v1)
fp2 = fingerprint.get_stage_fingerprint(stage_tuple_v2)
print(f"Fingerprint V1 tuple: {fp1}")
print(f"Fingerprint V2 tuple: {fp2}")
print(f"Fingerprints SAME (BUG!): {fp1 == fp2}")

# Test 4: Nested dict with instance
print("\n--- Test 4: Nested dict with instance ---")
config_v1 = {"processing": {"worker": WorkerV1()}}
config_v2 = {"processing": {"worker": WorkerV2()}}

def stage_nested_v1():
    return config_v1["processing"]["worker"].work(10)

def stage_nested_v2():
    return config_v2["processing"]["worker"].work(10)

fp1 = fingerprint.get_stage_fingerprint(stage_nested_v1)
fp2 = fingerprint.get_stage_fingerprint(stage_nested_v2)
print(f"Fingerprint V1 nested: {fp1}")
print(f"Fingerprint V2 nested: {fp2}")
print(f"Results differ: {stage_nested_v1()} vs {stage_nested_v2()}")
print(f"Fingerprints SAME (BUG!): {fp1 == fp2}")

print("\n" + "=" * 80)
print("ANALYSIS")
print("=" * 80)
print("""
The _process_collection_dependency function only processes CALLABLE items:

    for i, value in enumerate(items):
        if callable(value) and is_user_code(value):
            _add_callable_to_manifest(...)

Instance objects are NOT callable (unless they have __call__), so they are
completely ignored. This means any instance stored in a dict/list/tuple
will not have its class tracked in the fingerprint.

IMPACT:
- Pipeline configurations that store worker instances in dicts/lists
- Strategy pattern implementations where strategies are in a registry
- Plugin systems where plugins are stored in collections
""")
