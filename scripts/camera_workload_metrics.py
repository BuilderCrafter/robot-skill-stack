"""SDK-free measurements and declared gates for the camera workload smoke test.

CPU counters describe the system exposed by /proc, while process visibility may
be restricted. Missing values stay missing; no CPU result substitutes for an
unavailable native GPU-backed run. Thresholds are smoke-test screening criteria,
not a benchmark confidence interval or a physical robot safety claim.
"""
from collections.abc import Mapping
import math
from pathlib import Path
import time


def _number(value):
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def distribution(values):
    """Finite samples with linear-interpolated percentiles; never invent zeros."""
    values = sorted(value for raw in values if (value := _number(raw)) is not None)
    if not values:
        return dict(count=0)

    def percentile(fraction):
        at = (len(values)-1)*fraction
        low = int(at)
        return values[low] + (values[min(low+1, len(values)-1)]-values[low])*(at-low)

    return dict(count=len(values), mean=sum(values)/len(values), p50=percentile(.5),
                p95=percentile(.95), p99=percentile(.99), worst=values[-1])


def cpu_snapshot(proc_root='/proc'):
    """Read aggregate CPU jiffies and available RAM without external packages.

    Linux guest/guest_nice counters are already included in user/nice, so only
    the first eight CPU fields contribute to the total. Idle includes iowait.
    A denied or absent interface is recorded independently for CPU and RAM.
    """
    root = Path(proc_root)
    result = dict(at=time.monotonic(), cpu_total=None, cpu_idle=None,
                  available_ram_bytes=None, total_ram_bytes=None, errors=[])
    try:
        cpu = next(line.split()[1:] for line in (root/'stat').read_text().splitlines()
                   if line.startswith('cpu '))
        fields = [int(value) for value in cpu[:8]]
        if len(fields) < 5:
            raise ValueError('Incomplete aggregate CPU counters')
        result.update(cpu_total=sum(fields), cpu_idle=fields[3]+fields[4])
    except (OSError, ValueError, StopIteration) as exc:
        result['errors'].append('cpu: '+type(exc).__name__)
    try:
        memory = {}
        for line in (root/'meminfo').read_text().splitlines():
            key, value = line.split(':', 1)
            if key in ('MemTotal', 'MemAvailable'):
                memory[key] = int(value.split()[0])*1024
        result.update(available_ram_bytes=memory.get('MemAvailable'),
                      total_ram_bytes=memory.get('MemTotal'))
        if result['available_ram_bytes'] is None:
            result['errors'].append('ram: MemAvailable unavailable')
    except (OSError, ValueError, IndexError) as exc:
        result['errors'].append('ram: '+type(exc).__name__)
    return result


def cpu_interval(before, after):
    """CPU busy percent across an interval, excluding idle and iowait."""
    values = [_number(snapshot.get(key)) for snapshot in (before, after)
              for key in ('cpu_total', 'cpu_idle')]
    if any(value is None for value in values):
        return None
    total = values[2]-values[0]
    idle = values[3]-values[1]
    if total <= 0 or idle < 0 or idle > total:
        return None
    return 100.*(total-idle)/total


def counter_deltas(before, after):
    """Nested numeric counter differences, retaining absence/reset as None."""
    result = {}
    for key in sorted(set(before or {}) | set(after or {})):
        first, last = (before or {}).get(key), (after or {}).get(key)
        if isinstance(first, Mapping) or isinstance(last, Mapping):
            result[key] = counter_deltas(first if isinstance(first, Mapping) else {},
                                         last if isinstance(last, Mapping) else {})
        else:
            first, last = _number(first), _number(last)
            result[key] = None if first is None or last is None or last < first else last-first
    return result


def _stat(summary, name, statistic):
    value = summary.get(name)
    if not isinstance(value, Mapping) or (_number(value.get('count')) or 0) <= 0:
        return None
    return _number(value.get(statistic))


def compare_cases(baseline, candidate):
    """Compare native result dictionaries, including a repeated baseline.

    Required input: top-level status='PASS', errors=[], and summary containing
    distributions, publication rates, motion_complete=True, frame_checks=True.
    Missing required evidence yields NOT_MEASURED; an observed failure yields
    FAIL even if other measurements are unavailable. CPU/GPU telemetry is not a
    pass prerequisite. Call this same function for end-baseline drift.
    """
    first, last = baseline.get('summary') or {}, candidate.get('summary') or {}
    gates, changes = {}, {}

    def gate(name, status, **details):
        gates[name] = dict(status=status, **details)

    for label, result in [('baseline', baseline), ('candidate', candidate)]:
        status = result.get('status')
        errors = result.get('errors')
        if status in ('BLOCKED', 'NOT_MEASURED'):
            gate(label+'_native_run', 'NOT_MEASURED', reason='Native workload unavailable; no performance result')
        elif status == 'FAIL' or errors:
            gate(label+'_native_run', 'FAIL', reason='Native run failed or reported errors')
        elif status != 'PASS' or errors is None:
            gate(label+'_native_run', 'NOT_MEASURED', reason='Completed error-free native run not demonstrated')
        else:
            gate(label+'_native_run', 'PASS')
        summary = result.get('summary') or {}
        if 'runtime_errors' in summary:
            runtime_errors = summary['runtime_errors']
            if isinstance(runtime_errors, (tuple, list)):
                gate(label+'_runtime_errors', 'FAIL' if runtime_errors else 'PASS',
                     count=len(runtime_errors))
            else:
                count = _number(runtime_errors)
                gate(label+'_runtime_errors', 'NOT_MEASURED' if count is None or count < 0
                     else 'FAIL' if count > 0 else 'PASS', count=count)

    for field in ('motion_complete', 'frame_checks'):
        values = [first.get(field), last.get(field)]
        status = 'FAIL' if False in values else 'PASS' if all(v is True for v in values) else 'NOT_MEASURED'
        gate(field, status, baseline=values[0], candidate=values[1])

    for field in ('kit_update_ms', 'control_interval_ms'):
        original, measured = _stat(first, field, 'p99'), _stat(last, field, 'p99')
        if original is None or measured is None or original <= 0 or measured < 0:
            gate(field, 'NOT_MEASURED', baseline=original, candidate=measured)
            continue
        limit = max(original*1.25, original+5.)
        gate(field, 'PASS' if measured <= limit else 'FAIL', baseline=original,
             candidate=measured, limit=limit, criterion='p99 <= max(baseline*1.25, baseline+5ms)')
        changes[field] = dict(absolute_ms=measured-original,
                              percent=100.*(measured/original-1.))

    for field, requested in [('semantic_hz', 5.), ('mapping_hz', 2.)]:
        original, measured = _number(first.get(field)), _number(last.get(field))
        if original is None or measured is None or original <= 0 or measured < 0:
            gate(field, 'NOT_MEASURED', baseline=original, candidate=measured)
            continue
        limit = max(original*.8, requested*.8)
        # A degraded initial baseline cannot establish a healthy comparison.
        passing = original >= requested*.8 and measured >= limit
        gate(field, 'PASS' if passing else 'FAIL', baseline=original,
             candidate=measured, minimum=limit, requested=requested,
             criterion='both baseline and candidate >=80% requested; candidate >=80% baseline')
        changes[field] = dict(absolute_hz=measured-original,
                              percent=100.*(measured/original-1.))

    for label, summary in [('baseline', first), ('candidate', last)]:
        age = _stat(summary, 'mapping_source_age_ms', 'worst')
        gate(label+'_source_age', 'NOT_MEASURED' if age is None else
             'PASS' if 0 <= age <= 1000. else 'FAIL', worst_ms=age, maximum_ms=1000.,
             criterion='Original source age, without timestamp rewriting')
        for field in ('physics_interval_ms', 'physics_step_ms', 'mapping_compute_ms',
                      'mapping_queue_ms', 'mapping_publish_age_ms'):
            value = _stat(summary, field, 'p99')
            gate(label+'_'+field+'_measured', 'PASS' if value is not None and value >= 0
                 else 'NOT_MEASURED', p99=value)

    states = [item['status'] for item in gates.values()]
    status = 'FAIL' if 'FAIL' in states else 'NOT_MEASURED' if 'NOT_MEASURED' in states else 'PASS'
    return dict(status=status, gates=gates, relative_changes=changes,
                scope='Single-run smoke-test screening; CPU/GPU telemetry is optional, not a native-workload substitute.')


def markdown_table(results):
    """Compact comparative table; absent native readings display an em dash."""
    def number(value, digits=2):
        value = _number(value)
        return '—' if value is None else f'{value:.{digits}f}'

    def safe(value):
        return str(value).replace('|', '/').replace('\n', ' ')

    lines = [
        '| Case | Third camera | Kit p99/worst ms | Physics p99 ms | Control interval p99 ms | Semantic / mapping Hz | Source age worst ms | CPU mean % | RAM available p50 GiB | Status |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---|',
    ]
    for result in results:
        summary = result.get('summary') or {}
        resolution = result.get('resolution')
        if isinstance(resolution, (tuple, list)):
            resolution = '×'.join(str(item) for item in resolution)
        resolution = resolution or '—'
        resolution = str(resolution).replace('x', '×')
        label = result.get('label', result.get('scenario', '?'))
        baseline_case = label in ('A', 'A_repeat')
        case_label = f'{label} / {resolution} group' if baseline_case and resolution != '—' else label
        third_camera = 'none (two cameras)' if baseline_case else resolution
        memory = _stat(summary, 'available_ram_bytes', 'p50')
        status = (result.get('comparison') or {}).get('status', result.get('status', 'NOT_MEASURED'))
        row = [case_label, third_camera,
               number(_stat(summary, 'kit_update_ms', 'p99'))+'/'+number(_stat(summary, 'kit_update_ms', 'worst')),
               number(_stat(summary, 'physics_step_ms', 'p99')),
               number(_stat(summary, 'control_interval_ms', 'p99')),
               number(summary.get('semantic_hz'))+' / '+number(summary.get('mapping_hz')),
               number(_stat(summary, 'mapping_source_age_ms', 'worst'), 1),
               number(_stat(summary, 'cpu_utilization_percent', 'mean'), 1),
               number(None if memory is None else memory/(1024**3)), status]
        lines.append('| '+' | '.join(safe(value) for value in row)+' |')
    return '\n'.join(lines)+'\n'
