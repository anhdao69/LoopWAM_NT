"""Conservative admission and throughput selection for the unscored sweep."""
import math

def assert_fresh_sweep(base):
    existing = [p.name for p in base.glob('speed_*') if p.is_dir()]
    if existing:
        raise RuntimeError('Refusing to retime cached/partial calibration. Preserve and archive the unfinished sweep before restarting: ' + ', '.join(sorted(existing)))

def select_profile(profiles, available_mib=None):
    eligible = [p for p in profiles if p['passed'] and p['backend'] == 'cuda-graph'
                and p['memory_margin_mib'] >= 12000 and p['episodes'] > 0
                and (available_mib is None or required_memory(p['workers'], profiles) <= available_mib)]
    if not eligible:
        raise ValueError('No successful CUDA-graph profile with 12000 MiB headroom that fits final admission')
    return min(eligible, key=lambda p: p['seconds'] / p['episodes'])

def required_memory(workers, profiles):
    if workers not in (4, 8, 10, 12):
        raise ValueError('Unsupported worker count')
    measured = [p for p in profiles if p['passed'] and p['workers'] == workers]
    floor = {4: 60000, 8: 85000}.get(workers, 0)
    if measured:
        return max(floor, max(p['peak_added_mib'] for p in measured) + 10000)
    if workers in (4, 8):
        return floor
    points = {p['workers']: p for p in profiles
              if p['passed'] and p['backend'] == 'cuda-graph' and p['memory_margin_mib'] >= 8000}
    if 4 not in points or 8 not in points:
        raise ValueError('Need successful four- and eight-worker memory measurements')
    slope = max(4000, (points[8]['peak_added_mib'] - points[4]['peak_added_mib']) / 4)
    return math.ceil(points[8]['peak_added_mib'] + (workers - 8) * slope * 1.15 + 10000)
