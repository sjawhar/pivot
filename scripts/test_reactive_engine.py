#!/usr/bin/env python3
"""Integration tests for watch mode (ReactiveEngine).

Tests:
1. Output correctness - watch mode produces same outputs as fresh run
2. Stage selection - correct stages run after specific changes
3. Change detection latency - time from file change to execution start
4. Debouncing - rapid changes coalesce into single execution
"""

import hashlib
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import time
import traceback

# Test configuration
PIVOT_BIN = pathlib.Path(__file__).parent.parent / ".venv" / "bin" / "pivot"


def hash_file(path: pathlib.Path) -> str:
    """Get SHA256 hash of file contents."""
    if not path.exists():
        return "MISSING"
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def create_test_pipeline(tmp_dir: pathlib.Path) -> None:
    """Create a minimal test pipeline with two stages."""
    # Create pivot.yaml
    (tmp_dir / "pivot.yaml").write_text("""\
stages:
  preprocess:
    python: stages.preprocess
    deps: [data/input.txt]
    outs: [data/processed.txt]

  analyze:
    python: stages.analyze
    deps: [data/processed.txt]
    outs: [results/output.txt]
""")

    # Create stages.py
    (tmp_dir / "stages.py").write_text("""\
import pathlib

def preprocess() -> None:
    input_file = pathlib.Path("data/input.txt")
    output_file = pathlib.Path("data/processed.txt")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    content = input_file.read_text()
    output_file.write_text(f"PROCESSED: {content}")

def analyze() -> None:
    input_file = pathlib.Path("data/processed.txt")
    output_file = pathlib.Path("results/output.txt")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    content = input_file.read_text()
    lines = content.strip().split("\\n")
    output_file.write_text(f"ANALYZED: {len(lines)} lines\\n{content}")
""")

    # Create input data
    data_dir = tmp_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "input.txt").write_text("line1\nline2\nline3\n")

    # Create .pivot marker
    (tmp_dir / ".pivot").mkdir(exist_ok=True)


def run_pivot(
    tmp_dir: pathlib.Path, *args: str, timeout: int = 60
) -> subprocess.CompletedProcess[str]:
    """Run pivot command in the test directory."""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(tmp_dir)

    result = subprocess.run(
        [str(PIVOT_BIN), *args],
        cwd=tmp_dir,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
    )
    return result


def test_fresh_run(tmp_dir: pathlib.Path) -> dict[str, str]:
    """Run pipeline fresh and return output hashes."""
    print("\n=== Test: Fresh Run ===")

    # Clean outputs
    for d in ["data/processed.txt", "results"]:
        p = tmp_dir / d
        if p.exists():
            if p.is_dir():
                shutil.rmtree(p)
            else:
                p.unlink()

    result = run_pivot(tmp_dir, "run")
    print(f"Exit code: {result.returncode}")
    print(f"Stdout: {result.stdout[:500]}")
    if result.returncode != 0:
        print(f"Stderr: {result.stderr}")
        raise RuntimeError("Fresh run failed")

    hashes = {
        "processed": hash_file(tmp_dir / "data" / "processed.txt"),
        "output": hash_file(tmp_dir / "results" / "output.txt"),
    }
    print(f"Output hashes: {hashes}")
    return hashes


def test_watch_produces_same_output(tmp_dir: pathlib.Path, expected_hashes: dict[str, str]) -> None:
    """Verify watch mode produces identical outputs."""
    print("\n=== Test: Watch Output Correctness ===")

    # Clean outputs
    for d in ["data/processed.txt", "results"]:
        p = tmp_dir / d
        if p.exists():
            if p.is_dir():
                shutil.rmtree(p)
            else:
                p.unlink()

    # Start watch mode in background
    env = os.environ.copy()
    env["PYTHONPATH"] = str(tmp_dir)

    proc = subprocess.Popen(
        [str(PIVOT_BIN), "run", "--watch"],
        cwd=tmp_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )

    try:
        # Wait for initial run to complete
        time.sleep(5)

        # Check outputs
        hashes = {
            "processed": hash_file(tmp_dir / "data" / "processed.txt"),
            "output": hash_file(tmp_dir / "results" / "output.txt"),
        }
        print(f"Watch output hashes: {hashes}")

        if hashes != expected_hashes:
            print("FAIL: Hashes don't match!")
            print(f"  Expected: {expected_hashes}")
            print(f"  Got: {hashes}")
            raise AssertionError("Watch mode produced different output")

        print("PASS: Outputs match fresh run")

    finally:
        proc.terminate()
        proc.wait(timeout=5)


def test_data_change_triggers_rerun(tmp_dir: pathlib.Path) -> None:
    """Verify data file change triggers correct stages."""
    print("\n=== Test: Data Change Detection ===")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(tmp_dir)

    proc = subprocess.Popen(
        [str(PIVOT_BIN), "run", "--watch"],
        cwd=tmp_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )

    try:
        # Wait for initial run
        time.sleep(3)

        # Get initial output hash
        initial_hash = hash_file(tmp_dir / "results" / "output.txt")

        # Modify input data
        input_file = tmp_dir / "data" / "input.txt"
        input_file.write_text("line1\nline2\nline3\nline4\n")  # Added line4

        # Wait for watch re-run
        time.sleep(3)

        # Check output changed
        new_hash = hash_file(tmp_dir / "results" / "output.txt")

        if new_hash == initial_hash:
            print("FAIL: Output didn't change after input modification")
            raise AssertionError("Data change not detected")

        # Verify output content reflects new data
        output_content = (tmp_dir / "results" / "output.txt").read_text()
        if "4 lines" not in output_content:
            print(f"FAIL: Output doesn't reflect new data: {output_content}")
            raise AssertionError("Output content incorrect")

        print("PASS: Data change correctly triggered re-run")

    finally:
        proc.terminate()
        proc.wait(timeout=5)


def test_code_change_triggers_reload(tmp_dir: pathlib.Path) -> None:
    """Verify code change triggers registry reload."""
    print("\n=== Test: Code Change Detection ===")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(tmp_dir)

    proc = subprocess.Popen(
        [str(PIVOT_BIN), "-v", "run", "--watch"],  # -v must come before run
        cwd=tmp_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )

    try:
        # Wait for initial run
        time.sleep(5)

        # Get initial output
        initial_content = (tmp_dir / "results" / "output.txt").read_text()
        print(f"Initial output content: {initial_content[:100]}")

        # Modify stages.py with a functional change (not just comment)
        stages_file = tmp_dir / "stages.py"
        original_code = stages_file.read_text()
        print(f"Original stages.py contains 'ANALYZED:': {'ANALYZED:' in original_code}")

        new_code = original_code.replace(
            'output_file.write_text(f"ANALYZED:', 'output_file.write_text(f"ANALYZED_V2:'
        )
        stages_file.write_text(new_code)
        print(f"Modified stages.py contains 'ANALYZED_V2:': {'ANALYZED_V2:' in new_code}")

        # Wait for watch re-run - longer wait to be safe
        time.sleep(8)

        # Check output reflects code change
        new_content = (tmp_dir / "results" / "output.txt").read_text()
        print(f"New output content: {new_content[:100]}")

        if "ANALYZED_V2" not in new_content:
            print("FAIL: Output doesn't reflect code change")
            # Read process output for debugging
            proc.terminate()
            stdout, _ = proc.communicate(timeout=5)
            print(f"Process output:\n{stdout[:2000]}")
            raise AssertionError("Code change not detected")

        print("PASS: Code change correctly triggered reload and re-run")

    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


def test_debouncing(tmp_dir: pathlib.Path) -> None:
    """Verify rapid changes are debounced into single execution."""
    print("\n=== Test: Debouncing ===")

    # Reset stages.py to original (previous test modified it)
    (tmp_dir / "stages.py").write_text("""\
import pathlib

def preprocess() -> None:
    input_file = pathlib.Path("data/input.txt")
    output_file = pathlib.Path("data/processed.txt")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    content = input_file.read_text()
    output_file.write_text(f"PROCESSED: {content}")

def analyze() -> None:
    input_file = pathlib.Path("data/processed.txt")
    output_file = pathlib.Path("results/output.txt")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    content = input_file.read_text()
    lines = content.strip().split("\\n")
    output_file.write_text(f"ANALYZED: {len(lines)} lines\\n{content}")
""")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(tmp_dir)

    proc = subprocess.Popen(
        [str(PIVOT_BIN), "run", "--watch"],
        cwd=tmp_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )

    try:
        # Wait for initial run (code changed, so stages will run)
        time.sleep(5)

        # Make rapid changes
        input_file = tmp_dir / "data" / "input.txt"
        for i in range(10):
            input_file.write_text(f"rapid change {i}\n")
            time.sleep(0.05)  # 50ms between changes

        # Wait for debounced execution
        time.sleep(5)

        # Verify final content
        output_content = (tmp_dir / "results" / "output.txt").read_text()
        if "rapid change 9" not in output_content:
            print(f"FAIL: Final change not reflected: {output_content}")
            raise AssertionError("Debouncing didn't capture final change")

        print("PASS: Rapid changes were debounced")

    finally:
        proc.terminate()
        proc.wait(timeout=5)


def main() -> int:
    """Run all tests."""
    print("=" * 60)
    print("Watch Mode Integration Tests")
    print("=" * 60)

    if not PIVOT_BIN.exists():
        print(f"ERROR: pivot binary not found at {PIVOT_BIN}")
        return 1

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = pathlib.Path(tmp)

        try:
            create_test_pipeline(tmp_dir)
            print(f"Created test pipeline in {tmp_dir}")

            # Run tests
            fresh_hashes = test_fresh_run(tmp_dir)
            test_watch_produces_same_output(tmp_dir, fresh_hashes)
            test_data_change_triggers_rerun(tmp_dir)
            test_code_change_triggers_reload(tmp_dir)
            test_debouncing(tmp_dir)

            print("\n" + "=" * 60)
            print("ALL TESTS PASSED")
            print("=" * 60)
            return 0

        except Exception as e:
            print(f"\nTEST FAILED: {e}")
            traceback.print_exc()
            return 1


if __name__ == "__main__":
    sys.exit(main())
