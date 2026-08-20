import struct, sys
GGML={0:"F32",1:"F16",8:"Q8_0",10:"Q2_K",12:"Q4_K",14:"Q6_K",15:"Q8_K"}
f=open(sys.argv[1],"rb")
def u32(): return struct.unpack("<I",f.read(4))[0]
def u64(): return struct.unpack("<Q",f.read(8))[0]
def s():
    n=u64(); return f.read(n).decode("utf-8","replace")
magic=u32(); ver=u32(); nt=u64(); nkv=u64()
def skip(t):
    if t==8: n=u64(); f.read(n)
    elif t in (6,4,5): f.read(4)
    elif t in (10,11,12): f.read(8)
    elif t in (0,1,7): f.read(1)
    elif t in (2,3): f.read(2)
    elif t==9:
        et=u32(); c=u64()
        for _ in range(c):
            if et==8: ln=u64(); f.read(ln)
            else: skip(et)
    else: raise ValueError(t)
for _ in range(nkv):
    k=s(); t=u32(); skip(t)
want=[l for l in sys.argv[2:]]
for _ in range(nt):
    name=s(); nd=u32(); dims=[u64() for _ in range(nd)]; tt=u32(); off=u64()
    for w in want:
        if name.startswith(w):
            print(f"  {name:38s} dims={str(dims):20s} {GGML.get(tt,tt)}")
            break
