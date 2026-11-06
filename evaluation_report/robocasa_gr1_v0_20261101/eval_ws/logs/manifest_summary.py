import json, sys, hashlib, collections
c = collections.defaultdict(lambda: [0, 0, hashlib.sha256()])
for l in open(sys.argv[1]):
    p, sz, h = json.loads(l); k = '/'.join(p.split('/')[:int(sys.argv[2])])
    e = c[k]; e[0] += 1; e[1] += sz; e[2].update(l.encode())
for k in sorted(c): print(f'{k}\t{c[k][0]}\t{c[k][1]}\t{c[k][2].hexdigest()[:16]}')
