#!/usr/bin/env python3
"""Debug: print sys.path and env when run via systemd."""
import sys
import os

print("=== DEBUG SYSTEMD ===")
print(f"EXE: {sys.executable}")
print(f"CWD: {os.getcwd()}")
print(f"HOME: {os.environ.get('HOME', '')}")
print(f"PATH: {os.environ.get('PATH', '')}")
print(f"VIRTUAL_ENV: {os.environ.get('VIRTUAL_ENV', '')}")
print(f"PYTHONPATH: {os.environ.get('PYTHONPATH', '')}")
print(f"sys.path:")
for p in sys.path:
    print(f"  {p}")

print("\n=== IMPORT TESTS ===")
try:
    import py_clob_client
    print(f"py_clob_client: OK ({py_clob_client.__file__})")
except ImportError as e:
    print(f"py_clob_client: FAIL ({e})")

try:
    import stable_baselines3
    print(f"stable_baselines3: OK ({stable_baselines3.__file__})")
except ImportError as e:
    print(f"stable_baselines3: FAIL ({e})")

try:
    import numpy
    print(f"numpy: OK ({numpy.__version__})")
except ImportError as e:
    print(f"numpy: FAIL ({e})")

print("=== END DEBUG ===")
