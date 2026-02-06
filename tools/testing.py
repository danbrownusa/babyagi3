"""
Self-Testing Tool

Allows the agent to run its own test suite and inspect results.
Useful for verifying changes, catching regressions, and self-improvement.
"""

import subprocess
import os

from tools import tool


@tool
def run_tests(path: str = "", keyword: str = "", max_failures: int = 0) -> dict:
    """Run the project's pytest test suite and return results.

    Use this to verify code changes, check for regressions, or inspect
    the health of the codebase. Returns structured pass/fail counts
    and failure details.

    Args:
        path: Specific test file or directory to run (e.g. "tests/test_agent.py"). Defaults to all tests.
        keyword: Only run tests matching this keyword expression (pytest -k)
        max_failures: Stop after this many failures (0 = no limit)
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    test_dir = os.path.join(project_root, "tests")

    # Build pytest command
    cmd = ["python", "-m", "pytest", "--tb=short", "-q"]

    if max_failures > 0:
        cmd.append(f"--maxfail={max_failures}")

    if keyword:
        cmd.extend(["-k", keyword])

    # Determine what to test
    if path:
        target = os.path.join(project_root, path) if not os.path.isabs(path) else path
    else:
        target = test_dir

    cmd.append(target)

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
            cwd=project_root,
        )

        # Parse the summary line (e.g. "14 passed, 2 failed in 1.23s")
        output = result.stdout + result.stderr
        passed = failed = errors = 0
        for line in output.splitlines():
            line = line.strip()
            if "passed" in line or "failed" in line or "error" in line:
                # Look for the pytest summary line like "10 passed, 2 failed"
                import re
                p = re.search(r"(\d+) passed", line)
                f = re.search(r"(\d+) failed", line)
                e = re.search(r"(\d+) error", line)
                if p:
                    passed = int(p.group(1))
                if f:
                    failed = int(f.group(1))
                if e:
                    errors = int(e.group(1))

        # Collect failure details (lines starting with FAILED)
        failures = [
            line.strip()
            for line in output.splitlines()
            if line.strip().startswith("FAILED") or line.strip().startswith("ERROR")
        ]

        return {
            "success": result.returncode == 0,
            "passed": passed,
            "failed": failed,
            "errors": errors,
            "failures": failures[:20],  # Cap to avoid huge output
            "output": output[-3000:] if len(output) > 3000 else output,  # Tail of output
        }

    except subprocess.TimeoutExpired:
        return {"success": False, "error": "Tests timed out after 120 seconds"}
    except FileNotFoundError:
        return {"success": False, "error": "pytest not found. Install with: pip install pytest"}
