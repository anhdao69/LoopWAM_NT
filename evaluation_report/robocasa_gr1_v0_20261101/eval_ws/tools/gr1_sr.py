#!/usr/bin/env python3
"""Current success rate of GR1 eval dirs: official (any-step) SR from finished task JSONs, and episode progress."""
import glob, json, os, re, sys
for O in sys.argv[1:]:
    if not os.path.isdir(O): continue
    js = [json.load(open(f)) for f in glob.glob(f'{O}/n12/*.json')]
    s = sum(sum(d['successes']) for d in js); n = sum(len(d['successes']) for d in js)
    vids = [v for v in glob.glob(f'{O}/videos/n12/*/*.mp4') if re.search(r'_success[01]\.mp4$', v)]
    fs = sum(v.endswith('_success1.mp4') for v in vids)
    print(f'{os.path.basename(O)}: {len(js)}/24 tasks finished, official SR {100 * s / max(1, n):.2f}% ({s}/{n}); '
          f'{len(vids)}/1200 episodes finished (final-step success {fs}, {100 * fs / max(1, len(vids)):.1f}% - lower bound)')
