import ast
import hashlib
import inspect
import textwrap
from types import ModuleType

import user_utils


def stage_google_style(data):
    return user_utils.helper_b(data) + user_utils.CONSTANT_A


class ModuleAttrExtractor(ast.NodeVisitor):
    def __init__(self):
        self.attrs = []

    def visit_Attribute(self, node):
        if isinstance(node.value, ast.Name):
            self.attrs.append((node.value.id, node.attr))
        self.generic_visit(node)


def extract_module_attrs(func):
    source = textwrap.dedent(inspect.getsource(func))
    tree = ast.parse(source)
    extractor = ModuleAttrExtractor()
    extractor.visit(tree)
    return extractor.attrs


def is_user_module(mod):
    if not hasattr(mod, "__file__") or mod.__file__ is None:
        return False
    path = mod.__file__
    if "site-packages" in path or "dist-packages" in path:
        return False
    if "/usr/lib/python" in path:
        return False
    return True


def get_fingerprint(func, visited=None):
    if visited is None:
        visited = set()

    if id(func) in visited:
        return {}
    visited.add(id(func))

    manifest = {}

    try:
        source = textwrap.dedent(inspect.getsource(func))
        tree = ast.parse(source)
        func_hash = hashlib.md5(ast.dump(tree).encode()).hexdigest()[:8]
        manifest[f"self:{func.__name__}"] = func_hash
    except OSError:
        manifest[f"self:{func.__name__}"] = "no-source"

    cv = inspect.getclosurevars(func)
    all_refs = {**cv.globals, **cv.nonlocals}

    for name, val in all_refs.items():
        if callable(val) and hasattr(val, "__code__"):
            manifest[f"func:{name}"] = get_func_hash(val)
            manifest.update(get_fingerprint(val, visited))

        elif isinstance(val, (bool, int, float, str, bytes, type(None))):
            manifest[f"const:{name}"] = repr(val)

        elif isinstance(val, ModuleType) and is_user_module(val):
            attrs = extract_module_attrs(func)
            for mod_name, attr_name in attrs:
                if mod_name == name:
                    attr_val = getattr(val, attr_name, None)
                    if callable(attr_val) and hasattr(attr_val, "__code__"):
                        manifest[f"mod:{name}.{attr_name}"] = get_func_hash(attr_val)
                        manifest.update(get_fingerprint(attr_val, visited))
                    elif attr_val is not None:
                        manifest[f"const:{name}.{attr_name}"] = repr(attr_val)

    return manifest


def get_func_hash(func):
    try:
        source = textwrap.dedent(inspect.getsource(func))
        tree = ast.parse(source)
        return hashlib.md5(ast.dump(tree).encode()).hexdigest()[:8]
    except OSError:
        return "no-source"


def test_google_style_with_user_module():
    print("=" * 60)
    print("TEST: Google-style import with user module")
    print("=" * 60)

    cv = inspect.getclosurevars(stage_google_style)
    print(f"Globals: {list(cv.globals.keys())}")

    for name, val in cv.globals.items():
        if isinstance(val, ModuleType):
            print(f"  {name} is a module: {val.__file__}")
            print(f"  is_user_module: {is_user_module(val)}")

    attrs = extract_module_attrs(stage_google_style)
    print(f"\nModule.attr usage in AST: {attrs}")

    manifest = get_fingerprint(stage_google_style)
    print("\nFull manifest:")
    for k, v in sorted(manifest.items()):
        print(f"  {k}: {v}")

    expected_functions = [
        "self:stage_google_style",
        "mod:user_utils.helper_b",
        "self:helper_b",
        "self:helper_a",
        "self:leaf_func",
    ]

    expected_constants = ["const:user_utils.CONSTANT_A"]

    print("\n--- Verification ---")
    for exp in expected_functions:
        if exp in manifest:
            print(f"  FOUND: {exp}")
        else:
            print(f"  MISSING: {exp}")

    for exp in expected_constants:
        if exp in manifest:
            print(f"  FOUND: {exp}")
        else:
            print(f"  MISSING: {exp}")

    if "unused_func" not in str(manifest):
        print("  GOOD: unused_func is NOT in manifest (precise tracking!)")
    else:
        print("  BAD: unused_func is in manifest (spurious dependency)")


if __name__ == "__main__":
    test_google_style_with_user_module()
