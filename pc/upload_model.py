"""Upload big model files to the PS4 (as NAME.part, renamed when complete; LIST checks the size, since GoldHEN's SIZE
truncates to 32 bits). A file goes to /data; a folder (pc/convert_model.py's models/ps4/<id>) to /data/models/<id>/.
  python upload_model.py models/a.bin [models/ps4/qwen3.5-2b ...]
(On a Mac: /usr/bin/python3, which macOS lets reach the LAN.)"""
import ftplib
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import build as b  # noqa: E402


def upload(p, dest):
    t0 = time.time()
    ftp = b.ps4_ftp(timeout=60)
    with open(p, "rb") as f:
        ftp.storbinary(f"STOR {dest}/{p.name}.part", f, blocksize=4 << 20)
    lines = []
    ftp.retrlines(f"LIST {dest}/{p.name}.part", lines.append)
    size = int(lines[0].split()[4]) if lines else -1
    if size != p.stat().st_size:
        sys.exit(f"{p.name}: uploaded {size} bytes of {p.stat().st_size}")
    try:
        ftp.delete(f"{dest}/{p.name}")   # (the older copy of the same file, replaced by the new one)
    except ftplib.error_perm:
        pass
    ftp.rename(f"{dest}/{p.name}.part", f"{dest}/{p.name}")
    ftp.quit()
    dt = max(time.time() - t0, 1e-3)
    print(f"{dest}/{p.name}: {size / 2**20:.0f} MiB in {dt:.0f} s ({size / 2**20 / dt:.1f} MiB/s)", flush=True)


def mkdirs(path):
    ftp = b.ps4_ftp()
    for d in ("/data/models", path):
        try:
            ftp.mkd(d)
        except ftplib.error_perm:   # (there already)
            pass
    ftp.quit()


for p in map(Path, sys.argv[1:]):
    if p.is_dir():
        dest = f"/data/models/{p.name}"
        mkdirs(dest)
        # model.idx last: the server lists a folder once its model.idx is there
        for f in sorted(p.iterdir(), key=lambda f: f.name == "model.idx"):
            if f.is_file():
                upload(f, dest)
    else:
        upload(p, "/data")
