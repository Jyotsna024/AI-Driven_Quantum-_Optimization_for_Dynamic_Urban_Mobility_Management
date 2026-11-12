import xml.etree.ElementTree as ET
from pathlib import Path

net = Path('sumo_files/hsr.net.xml')
print('Net size:', net.stat().st_size, 'bytes')
root = ET.parse(net).getroot()
junctions = [j for j in root.findall('.//junction') if not j.get('id','').startswith(':')]
tls = root.findall('.//tlLogic')
edges = [e for e in root.findall('.//edge') if e.get('function') != 'internal']
print(f'Junctions (non-internal): {len(junctions)}')
print(f'TLS programs: {len(tls)}')
print(f'Edges (non-internal): {len(edges)}')
for j in junctions:
    jid = j.get('id')
    jtype = j.get('type')
    tl = j.get('tl','')
    x = float(j.get('x', 0))
    y = float(j.get('y', 0))
    print(f'  junc: {jid}  type={jtype}  tl={tl}  x={x:.1f}  y={y:.1f}')
for t in tls:
    phases = t.findall('phase')
    tid = t.get('id')
    ttype = t.get('type')
    print(f'  tls: {tid}  phases={len(phases)}  type={ttype}')
for e in edges[:8]:
    print(f'  edge: {e.get("id")}  from={e.get("from")}  to={e.get("to")}')
