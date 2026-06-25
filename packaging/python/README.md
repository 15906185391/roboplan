# Python packaging

This directory contains the PyPI/cibuildwheel-specific packaging helpers for RoboPlan.
The normal source-tree CMake build remains in the package directories.
`pyproject.toml` points `scikit-build-core` at this directory with `tool.scikit-build.cmake.source-dir`.

The helper CMake files cover three packaging-only concerns:

- expose `cmeel`'s isolated native dependency prefix to `scikit-build-core` builds;
- install the runtime native libraries that wheels need at import time;
- repair installed RPATHs with `patchelf` using checked-in CMake script templates rather than large embedded `install(CODE ...)` strings.

Release wheels are built with `cibuildwheel` in the [`build-pypi-wheels.yml`](../../.github/workflows/build-pypi-wheels.yml) CI workflow, and are smoke-tested by importing the `roboplan` namespace and every compiled submodule.
That workflow runs on pull requests so wheel build breakage is caught before release.
The [`release.yml`](../../.github/workflows/release.yml) workflow invokes the same build workflow from the tagged commit, downloads those fresh artifacts, and publishes them through PyPI trusted publishing.


## Local checks

Run commands from the repository root. Keep native parallelism capped on small machines, matching CI:

```bash
export CMAKE_BUILD_PARALLEL_LEVEL=2
export MAKEFLAGS=-j2
export NINJAFLAGS=-j2
```

Source-build and import-test the unified wheel path:

```bash
uv venv --seed /tmp/roboplan-wheel-check
uv pip install -v --python /tmp/roboplan-wheel-check/bin/python --no-cache .
/tmp/roboplan-wheel-check/bin/python - <<'PY'
import roboplan
import roboplan.core
import roboplan.filters
import roboplan.example_models
import roboplan.simple_ik
import roboplan.optimal_ik
import roboplan.rrt
import roboplan.toppra
import roboplan.cartesian_planning
print("roboplan imports ok")
PY
```

Check the structural packaging contract:

```bash
uvx pytest tests/unified_python_package_test.py -q
```

## Run an example against the wheel

To prove the wheel is usable end-to-end, install it into a fresh environment and run one of the examples.
The examples are not part of the wheel, so they pull in a few extra runtime dependencies on top of `roboplan` itself.

```bash
# 1. Create an isolated environment and install the wheel we just built.
uv venv --seed /tmp/roboplan-wheel-check
uv pip install --python /tmp/roboplan-wheel-check/bin/python --no-cache .

# 2. Install the extra dependencies the examples need.
uv pip install --python /tmp/roboplan-wheel-check/bin/python pycollada tyro xacro viser

# 3. Run an example.
/tmp/roboplan-wheel-check/bin/python roboplan_examples/python/example_ik.py
```
