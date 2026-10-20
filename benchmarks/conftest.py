from __future__ import annotations

import pathlib
import sys

# The harness is a directory of scripts, not a package; its tests import them by module name.
_benchmarks_dir = pathlib.Path(__file__).parent
if str(_benchmarks_dir) not in sys.path:
    sys.path.append(str(_benchmarks_dir))
