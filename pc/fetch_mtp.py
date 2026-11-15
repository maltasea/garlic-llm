"""Only the MTP head (the mtp.* tensors) of a Hugging Face Qwen3.5 checkpoint, fetched with HTTP range requests from
its safetensors files, into models/<name>-mtp/model.safetensors (+ config.json), for convert_model.py --mtp=
(a GGUF has no MTP head: llama.cpp leaves it out).
  python pc/fetch_mtp.py Qwen/Qwen3.5-4B        -> models/Qwen3.5-4B-mtp/
(On a Mac: /usr/bin/python3, or any python: only the standard library.)"""
import json
import struct
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def get(url, a=None, b=None):
    req = urllib.request.Request(url, headers={"Range": f"bytes={a}-{b - 1}"} if a is not None else {})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def main():
    repo = sys.argv[1]
    base = f"https://huggingface.co/{repo}/resolve/main/"
    out = ROOT / "models" / (repo.split("/")[-1] + "-mtp")
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_bytes(get(base + "config.json"))
    wmap = json.loads(get(base + "model.safetensors.index.json"))["weight_map"]
    names = sorted(k for k in wmap if k.startswith("mtp."))
    head, blobs, off = {}, [], 0
    for f in sorted({wmap[k] for k in names}):
        n = struct.unpack("<Q", get(base + f, 0, 8))[0]
        h = json.loads(get(base + f, 8, 8 + n))
        for k in names:
            if wmap[k] != f:
                continue
            a, b = h[k]["data_offsets"]
            blobs.append(get(base + f, 8 + n + a, 8 + n + b))
            head[k] = {"dtype": h[k]["dtype"], "shape": h[k]["shape"], "data_offsets": [off, off + b - a]}
            off += b - a
            print(f"{k} {h[k]['shape']}", flush=True)
    hb = json.dumps(head).encode()
    hb += b" " * (-len(hb) % 8)
    with open(out / "model.safetensors", "wb") as w:
        w.write(struct.pack("<Q", len(hb)) + hb)
        for x in blobs:
            w.write(x)
    print(f"{out}: {len(names)} tensors, {off / 2**20:.0f} MiB")


if __name__ == "__main__":
    main()
