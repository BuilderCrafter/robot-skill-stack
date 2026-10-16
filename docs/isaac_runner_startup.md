# Isaac regression startup: assessment and direct-script correction

## Evidence and limits

The workstation report is that the GUI runs Home/Move/Pick/Place, while all eight
new runner scenarios segfault immediately after printing NumPy 1.26.4's correct
path. This establishes successful NumPy pre-import, not successful native Isaac
startup. The cloud machine has no usable Isaac/GPU installation, so the native
crash and its exact cause cannot be reproduced or established here.

The common dispatch changed between Phase 1 and initial Phase 2:

| Property | Phase 1 | Initial Phase 2 | Corrected path |
|---|---|---|---|
| Python invocation | `python.sh path/to/test.py ...` | `python.sh -c CODE tests.group.module ...` | `python.sh path/to/test.py ...` |
| Main module | Actual scenario script | `-c` launcher; runpy uses a separate globals dictionary | Actual scenario script |
| `sys.modules['__main__'].__file__` | Script path | Absent | Script path |
| Scenario globals are the registered main namespace | Yes | No (`alter_sys=False`, the runpy default) | Yes |
| `sys.argv[0]` | Script path | Manually set dotted module name | Script path |
| Scenario `__spec__` | `None` | Module spec | `None` |
| Explicit NumPy check before SDK import | No | In launcher string | At top of each scenario file |

A plain-Python subprocess probe confirmed the namespace, file, argument and spec
differences. In the old path, the executed scenario has a valid `__file__`, but
the separately registered `__main__` does not. Direct execution also restores
normal script-directory `sys.path[0]`; the repository remains on `PYTHONPATH`
through the unchanged wrapper. The result/artifact argument values themselves
were already present in Phase 1 and were not new to Phase 2.

The altered execution context is a plausible common trigger for native startup,
but no evidence identifies which context property, if any, Isaac inspects or
which native component faults. Default standalone `SimulationApp` startup also
differs from the working GUI's explicit full experience and ROS overrides.
GUI success alone does not rule out a standalone SDK/plugin problem. No change
to the experience, SDK, dependencies or lifecycle is justified by this evidence.

## Minimal correction

- Remove the runner's inline entry string and `runpy.run_module()` dispatch.
- Invoke all eight existing test files directly through `run_isaac_python.sh`.
- Move the existing NumPy import to the beginning of each file, check exact
  version 1.26.4 and print its path before `from isaacsim import SimulationApp`.
- Preserve every scenario body, its arguments, result checks and artifacts.

The small repeated startup guard keeps each scenario a conventional standalone
script, including when run manually; it introduces no bootstrap abstraction.
`run_isaac_python.sh`, `.deps` precedence, GUI launch, dependency pins, robot code,
perception, tracking, planners and Isaac lifecycle implementations are unchanged.
No unsupported launcher argument is added.

## Cloud validation

```bash
.venv/bin/python -m unittest tests.unit.test_project_runners -v
.venv/bin/python scripts/run_all_regressions.py --unit-only \
    --artifacts-dir outputs/isaac_startup_fix
git diff --check
```

Runner tests verify direct file dispatch and selection of all eight scenarios,
argument forwarding with spaces, normal main-module metadata, fresh result paths,
and NumPy pre-import before a fake SDK is loaded. Each real scenario is exercised
up to a fake `SimulationApp` construction, which exits without running simulation.
Wrong NumPy is rejected before SDK import. Crash return codes `-11` and `139`
remain failures even if a PASS JSON exists, and later selected scenarios still run.
These tests do not validate real native SDK startup or physical manipulation.

Cloud results: 18 runner tests PASS; complete offline run PASS with 173 discovery
tests, all eight standalone unit scripts, and architecture/syntax checks.
`git diff --check` passed. Logs are retained under the ignored
`outputs/isaac_startup_fix/run-py24kqso/` directory. No SDK or dependency changes
were made during investigation.

## Workstation validation and minimal diagnosis

From the repository root, using the existing installation and dependencies:

```bash
# First isolate one direct-script entry point with its usual result arguments.
mkdir -p outputs/isaac_startup_probe
./run_isaac_python.sh tests/isaac/test_perception_discovery.py \
    --result outputs/isaac_startup_probe/discovery.json \
    --artifacts-dir outputs/isaac_startup_probe/discovery

# Then validate every retained scenario through the corrected runner.
python3 scripts/run_all_regressions.py --group isaac \
    --artifacts-dir outputs/isaac_startup_fix
python3 scripts/run_all_regressions.py --group regression \
    --artifacts-dir outputs/isaac_startup_fix
```

Inspect the new run directories' logs, actual exit codes and scenario JSON.
All two integration and six regression scenarios must execute and report PASS;
the NumPy line or an open GUI alone is not a passing regression result.

If the first direct invocation still segfaults, isolate SDK startup with this
temporary file; do not install packages or change the working environment:

```bash
cat > /tmp/robot_isaac_startup_probe.py <<'PY'
import faulthandler
import sys
faulthandler.enable(all_threads=True)
import numpy as np
assert np.__version__ == '1.26.4', (np.__version__, np.__file__)
print('numpy:', np.__version__, np.__file__, flush=True)
print('entry:', __file__, __spec__, sys.argv, flush=True)
print('before isaacsim import', flush=True)
from isaacsim import SimulationApp
print('after import; before SimulationApp construction', flush=True)
app = SimulationApp({'headless': False})
print('constructed SimulationApp', flush=True)
app.close()
print('closed SimulationApp', flush=True)
PY
./run_isaac_python.sh /tmp/robot_isaac_startup_probe.py \
    > outputs/isaac_startup_probe/sdk.log 2>&1
printf 'probe exit: %s\n' "$?"
```

Record the last marker, SDK log and any fatal stack/core information. If the
probe fails too, direct script mode is insufficient and the fault precedes
scenario logic. If it succeeds, repeat it with the original `--result` and
`--artifacts-dir` arguments appended to distinguish argument-sensitive startup
from later scenario imports/setup. Compare with the successful GUI's experience
and startup log before proposing further changes. Do not treat a Python-only
metadata difference as a proven explanation for a native segmentation fault.
