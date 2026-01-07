import yaml

# Use union types to avoid type: ignore on fallback assignment
Loader: type[yaml.SafeLoader] | type[yaml.CSafeLoader]
Dumper: type[yaml.SafeDumper] | type[yaml.CSafeDumper]

try:
    Loader = yaml.CSafeLoader
    Dumper = yaml.CSafeDumper
except AttributeError:
    # CSafeLoader unavailable (no libyaml); SafeLoader is API-compatible
    Loader = yaml.SafeLoader
    Dumper = yaml.SafeDumper
