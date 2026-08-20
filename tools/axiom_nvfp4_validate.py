#!/usr/bin/env python3
"""Validate a downloaded DeepSeek-V4-Flash-NVFP4 model dir: parse every shard header,
aggregate dtypes, confirm the routed-expert NVFP4 (U8) count == layers*experts*3, and
report the layer range. Reads only headers (fast). Usage: python axiom_nvfp4_validate.py <dir>
"""
import json, struct, collections, glob, re, sys
d = sys.argv[1] if len(sys.argv) > 1 else "."
files = sorted(glob.glob(d + "/*.safetensors"))
dt = collections.Counter(); total = 0; u8_expert = 0; bad = 0; layers = set()
for p in files:
    try:
        f = open(p, "rb"); n = struct.unpack("<Q", f.read(8))[0]; h = json.loads(f.read(n)); f.close()
    except Exception as ex:
        print("BAD header", p, ex); bad += 1; continue
    h.pop("__metadata__", None)
    for k, v in h.items():
        dt[v["dtype"]] += 1; total += 1
        if v["dtype"] == "U8" and "experts." in k and k.endswith(".weight"): u8_expert += 1
        m = re.search(r"layers\.(\d+)\.", k)
        if m: layers.add(int(m.group(1)))
NL = len(layers)
print("shards=%d parsed_ok=%d bad=%d tensors=%d" % (len(files), len(files) - bad, bad, total))
print("dtypes", dict(dt))
print("routed-expert U8 weights=%d  layers=%d range=%d..%d" % (u8_expert, NL, min(layers), max(layers)))
print("expected experts*3*layers=%d  match=%s" % (256 * 3 * NL, u8_expert == 256 * 3 * NL))
