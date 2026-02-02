---
tags: [python, multiprocessing, containers, cgroups]
category: gotcha
module: executor
symptoms: ["OOM in containers", "too many workers spawned", "container CPU throttling"]
---

# Use loky.cpu_count() for Container-Aware Worker Limits

## Problem

`os.cpu_count()` returns the host machine's CPU count, ignoring container resource limits:

```python
import os

# On a 64-core host with container limited to 4 CPUs:
os.cpu_count()  # Returns 64, not 4
```

This causes multiprocessing code to spawn far more workers than the container can handle:

1. **CPU throttling** - Container exceeds its CPU quota, scheduler throttles all processes
2. **OOM kills** - Each worker consumes memory; 64 workers on a 4-CPU container exhausts RAM
3. **Cascading failures** - Throttled workers time out, triggering retries that make it worse

The issue is pervasive in containerized environments (Docker, Kubernetes, CI runners) where cgroups v1/v2 limit CPU shares or cores.

```python
# Naive approach - spawns too many workers in containers
from concurrent.futures import ProcessPoolExecutor
import os

with ProcessPoolExecutor(max_workers=os.cpu_count()) as pool:
    # On 4-CPU container: spawns 64 workers, gets throttled/killed
    results = pool.map(heavy_compute, data)
```

## Solution

Use `loky.cpu_count()` which reads cgroup CPU limits:

```python
import loky

def compute_max_workers(stage_count: int, override: int | None = None) -> int:
    """Compute worker count respecting container limits."""
    cpu_count = loky.cpu_count() or 1  # Container-aware
    max_workers = override if override is not None else cpu_count
    return max(1, min(max_workers, stage_count))
```

loky inspects multiple cgroup interfaces:

1. **cgroups v2** (`/sys/fs/cgroup/cpu.max`) - Modern unified hierarchy
2. **cgroups v1** (`/sys/fs/cgroup/cpu/cpu.cfs_quota_us`) - Legacy hierarchy
3. **Fallback** - `os.cpu_count()` when not in a cgroup

This works transparently across:
- Docker with `--cpus=4` or `--cpu-quota`
- Kubernetes CPU limits (`resources.limits.cpu`)
- systemd slices with CPU quotas
- CI runners (GitHub Actions, GitLab CI) with resource limits

### Verifying Container Limits

To debug CPU limit detection:

```python
import loky
import os

print(f"os.cpu_count(): {os.cpu_count()}")
print(f"loky.cpu_count(): {loky.cpu_count()}")

# Check cgroup directly (Linux)
try:
    # cgroups v2
    with open("/sys/fs/cgroup/cpu.max") as f:
        quota, period = f.read().strip().split()
        if quota != "max":
            print(f"cgroup v2 limit: {int(quota) / int(period)} CPUs")
except FileNotFoundError:
    pass

try:
    # cgroups v1
    with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us") as f:
        quota = int(f.read().strip())
    with open("/sys/fs/cgroup/cpu/cpu.cfs_period_us") as f:
        period = int(f.read().strip())
    if quota > 0:
        print(f"cgroup v1 limit: {quota / period} CPUs")
except FileNotFoundError:
    pass
```

### Alternative: joblib

If you're using joblib directly (not through Pivot), it also handles cgroups:

```python
from joblib import cpu_count

# Returns container-aware count
workers = cpu_count()
```

Both loky and joblib use similar detection logic since loky is joblib's process backend.

## Key Insight

Container resource limits are enforced by the kernel, not visible to userspace syscalls like `sysconf(_SC_NPROCESSORS_ONLN)` that `os.cpu_count()` uses. Applications must actively read cgroup interfaces to discover their actual limits.

This is a common source of container misbehavior:

| Environment | `os.cpu_count()` | `loky.cpu_count()` | Workers Spawned |
|-------------|------------------|---------------------|-----------------|
| Bare metal (8 cores) | 8 | 8 | 8 (correct) |
| Docker `--cpus=2` | 8 | 2 | 8 vs 2 |
| K8s `limits.cpu: 500m` | 64 | 1 | 64 vs 1 |
| GitHub Actions | 2-4 | 2-4 | Usually correct |

The principle: **never trust `os.cpu_count()` in code that might run in containers**. Use container-aware libraries (loky, joblib) or read cgroups directly.
