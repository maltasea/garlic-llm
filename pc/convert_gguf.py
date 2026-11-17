"""A Qwen3 model (chat, or Qwen3-Embedding) or an LFM2 chat model (LFM2.5) in GGUF (F32 / F16 / BF16 / Q8_0 / K-quants)
-> a model folder for the PS4 (chat/server.h MODELS_DIR). The embedder is served at /v1/embeddings next to the chat
model; LFM2 is a chat model ("PML2": its short-conv layers, 64-dim heads, a CHAT_SEQ context):
  weights.bin   PMDL; the embedder's matrices in q8 (fmt 9: GGUF's Q8_0 values as they are), a chat model's in q4;
                tied embedding, no lm_head of its own
  model.idx     PMDL, MAX_SEQ (the longest input) = EMB_SEQ, the weights a bare name: the folder's weights.bin
  kernels.bin   shaders/qwen3.cl with MAX_SEQ = EMB_SEQ (head_dim 128) and the batched prompt (PREFILL)
  tokenizer.bin from the GGUF's vocabulary and merges (as pc/export_chat.py export_tokenizer)
  embedding.txt "last": pooling (the final norm's output at the last token, then L2 norm); marks the folder
A chat model's matrices: Q8_0 is unpacked exactly (int8 * fp16 scale), then quantized to q4 with the least-error scale
per block (pc/q4.py quantize_q4_mse). The embedder's: Q8_0 copied (int8 rows, then the fp16 scales), anything else
quantized to Q8_0 as llama.cpp does (d = max |w| / 127).

  .venv/bin/python pc/convert_gguf.py ~/models/Qwen3-Embedding-0.6B-Q8_0.gguf [--id qwen3-embedding-0.6b]
  .venv/bin/python pc/convert_gguf.py ~/models/LFM2.5-2.6B-QAD-Q4_0.gguf [--id=lfm2.5-2.6b] [--ctx=2048]
  .venv/bin/python pc/convert_gguf.py ~/models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf --id=qwen3-4b-2507 [--ctx=4096]
      -> models/ps4/<id>/
"""
import json
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from q4 import interleave_q4, quantize_q4_mse  # noqa: E402

CHAT_SEQ = 4096   # a chat model's context, unless --ctx= (fp16 KV cache: LFM2.5 16 KiB per token, Qwen3-4B 144 KiB)
EMB_SEQ = 2048   # tokens per input (the KV cache: 28 layers x 2 x 2048 x 8 heads x 128 x fp16 = 224 MiB)


def gguf(path):
    """(metadata dict, {name: (dims, type, absolute offset)}, memmap) of a GGUF v2/v3 file."""
    f = open(path, "rb")
    assert f.read(4) == b"GGUF"
    _ver, nt, nkv = struct.unpack("<IQQ", f.read(20))
    fm = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}

    def s():
        return f.read(struct.unpack("<Q", f.read(8))[0]).decode("utf-8", "surrogateescape")

    def val(t):
        if t in fm:
            return struct.unpack("<" + fm[t], f.read(struct.calcsize(fm[t])))[0]
        if t == 8:
            return s()
        at, n = struct.unpack("<IQ", f.read(12))
        return [val(at) for _ in range(n)]

    meta = {}
    for _ in range(nkv):
        k = s()
        meta[k] = val(struct.unpack("<I", f.read(4))[0])
    tensors = {}
    for _ in range(nt):
        name = s()
        nd = struct.unpack("<I", f.read(4))[0]
        dims = struct.unpack("<" + "Q" * nd, f.read(8 * nd))
        ty, off = struct.unpack("<IQ", f.read(12))
        tensors[name] = (dims, ty, off)
    align = meta.get("general.alignment", 32)
    base = (f.tell() + align - 1) // align * align
    tensors = {k: (d, t, base + o) for k, (d, t, o) in tensors.items()}
    return meta, tensors, np.memmap(path, np.uint8, "r")


def tensor(tensors, mm, name):
    """float32 numpy, shape (rows, cols) = GGUF dims reversed."""
    dims, ty, off = tensors[name]
    n = int(np.prod(dims))
    shape = tuple(reversed(dims))
    if ty == 0:   # F32
        return mm[off:off + 4 * n].view(np.float32).reshape(shape).copy()
    if ty == 1:   # F16
        return mm[off:off + 2 * n].view(np.float16).astype(np.float32).reshape(shape)
    if ty == 30:  # BF16
        return (mm[off:off + 2 * n].view(np.uint16).astype(np.uint32) << 16).view(np.float32).reshape(shape)
    if ty == 8:   # Q8_0: blocks of 32: fp16 d, 32 int8
        b = mm[off:off + n // 32 * 34].reshape(-1, 34)
        d = b[:, :2].copy().view(np.float16).astype(np.float32)
        q = b[:, 2:].view(np.int8).astype(np.float32)
        return (q * d).reshape(shape)
    import gguf   # the K-quants (Q4_K, Q5_K, Q6_K, ...): llama.cpp's own unpacking (pip install gguf)
    qt = gguf.GGMLQuantizationType(ty)
    bs, ts = gguf.GGML_QUANT_SIZES[qt]
    return gguf.quants.dequantize(mm[off:off + n // bs * ts], qt).astype(np.float32).reshape(shape)


def bytes_to_unicode():
    """GPT-2's byte <-> printable character table (as pc/export_chat.py): {character: byte}."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


def tokenizer(meta, out):
    """tokenizer.bin (PTOK, as export_chat.py export_tokenizer) from tokenizer.ggml.tokens / token_type / merges."""
    toks, types, merges = meta["tokenizer.ggml.tokens"], meta["tokenizer.ggml.token_type"], meta["tokenizer.ggml.merges"]
    dec = bytes_to_unicode()
    vocab = {t: i for i, t in enumerate(toks)}
    raw, special = [], []
    for t, ty in zip(toks, types):
        sp = ty in (3, 4)   # control / user-defined: matched as a whole, its text as it is
        raw.append(t.encode("utf-8", "surrogateescape") if sp else bytes(dec[ch] for ch in t))
        special.append(1 if sp else 0)
    byte_id = [vocab[next(ch for ch, b in dec.items() if b == x)] for x in range(256)]
    mrec = []
    for m in merges:
        a, b = m.split(" ")
        mrec.append(struct.pack("<III", vocab[a], vocab[b], vocab[a + b]))
    magic = b"PTK3" if meta.get("tokenizer.ggml.pre") == "lfm2" else b"PTOK"   # (PTK3: numbers in runs of up to 3)
    data = struct.pack("<4sII", magic, len(toks), len(mrec)) + struct.pack("<256I", *byte_id)
    data += b"".join(struct.pack("<BH", special[i], len(r)) + r for i, r in enumerate(raw))
    data += b"".join(mrec)
    (out / "tokenizer.bin").write_bytes(data)
    print(f"tokenizer.bin: {len(toks)} tokens, {len(mrec)} merges")


def kernels(out, defines):
    """shaders/qwen3.cl with these -D's (MAX_SEQ, HEAD_DIM, PREFILL) -> kernels.bin (as export_chat.py)."""
    import gpu
    obj = out / "qwen3.o"
    flags = [f"-D{d}" for d in defines]
    subprocess.run(gpu.CLANG + flags + ["-c", str(gpu.SHADERS / "qwen3.cl"), "-o", str(obj)], check=True)
    syms, sections = gpu.elf_symbols(obj.read_bytes())
    obj.unlink()
    text = sections[".text"]
    recs = []
    for sym, (sec, value, _size) in sorted(syms.items()):
        if sym.endswith(".kd"):
            name = sym[:-3]
            entry = syms[name][1]
            assert entry % 256 == 0, name
            k = gpu.Kernel(name, entry, sections[sec][value: value + 64])
            recs.append(struct.pack("<32sIIIII", name.encode(), entry, k.rsrc1, k.rsrc2, k.user_sgprs, k.kernarg_sgpr))
    (out / "kernels.bin").write_bytes(struct.pack("<4sII", b"PKRN", len(recs), len(text)) + b"".join(recs) + text)
    print(f"kernels.bin: {len(recs)} kernels, {len(text)} bytes of code")


def main():
    src = Path([a for a in sys.argv[1:] if not a.startswith("--")][0]).expanduser()
    meta, tensors, mm = gguf(src)
    arch = meta["general.architecture"]
    assert arch in ("qwen3", "lfm2"), f"{arch}: only Qwen3 (chat or Qwen3-Embedding) and LFM2 so far"
    lfm = arch == "lfm2"
    chat = lfm or "qwen3.pooling_type" not in meta   # (an embedding model has a pooling type)
    mid = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--id=")),
               src.stem.lower().replace("-q8_0", "").replace("-q4_k_m", "").replace("-f16", "").replace("-bf16", ""))
    out = ROOT / "models" / "ps4" / mid
    out.mkdir(parents=True, exist_ok=True)
    A = arch + "."
    n_layers, hidden = meta[A + "block_count"], meta[A + "embedding_length"]
    n_heads, kvs = meta[A + "attention.head_count"], meta[A + "attention.head_count_kv"]
    kvs = kvs if isinstance(kvs, list) else [kvs] * n_layers   # (LFM2: 0 = a short-conv layer)
    n_kv, full_mask = max(kvs), sum(1 << i for i, k in enumerate(kvs) if k)
    head_dim = meta.get(A + "attention.key_length", hidden // n_heads)
    get = lambda name: tensor(tensors, mm, name)  # noqa: E731

    w = open(out / "weights.bin", "wb")
    recs = {}

    def put(key, data, rows=0, cols=0, scales=0, fmt=0):
        w.write(b"\0" * ((-w.tell()) % 256))
        recs[key] = (w.tell(), len(data), rows, cols, scales, fmt)
        w.write(data)

    def f32(key, a):
        put(key, np.ascontiguousarray(a, np.float32).tobytes())

    copied = []

    def q4(key, *names):
        """The rows of these tensors, one after the other, in q4. Q4_0 tensors are copied as they are (q4 is Q4_0's
        block: fp16 d, then q[j] | q[j + 16] << 4), anything else is quantized (cosine 0.94 to the source, mse 0.95)."""
        if all(tensors[n][1] == 2 for n in names):
            parts = []
            for n in names:
                (cols, rows), _, off = tensors[n]
                b = mm[off:off + rows * cols // 32 * 18].reshape(rows, cols // 32, 18)
                parts.append((b[..., 2:].reshape(rows, cols // 2), b[..., :2].copy().view(np.float16)[..., 0]))
            packed, d16 = np.concatenate([a for a, _ in parts]), np.concatenate([d for _, d in parts])
            copied.append(key)
        else:
            packed, d16 = quantize_q4_mse(np.concatenate([get(n) for n in names]))
        if chat:   # q4i (the same values, rows in tiles of 64): the batched prompt's matrix kernels read it
            nib, sc = interleave_q4(packed, d16)
            put(key, nib + sc, rows=packed.shape[0], cols=packed.shape[1] * 2, scales=len(nib), fmt=3)
        else:
            put(key, packed.tobytes() + d16.tobytes(), rows=packed.shape[0], cols=packed.shape[1] * 2,
                scales=packed.nbytes, fmt=2)

    def q8(key, *names):
        """The rows of these tensors in q8 (fmt 9): int8 [rows][cols], then fp16 scales [rows][cols / 32]."""
        qs, ds = [], []
        for n in names:
            (cols, rows), ty, off = tensors[n]
            if ty == 8:   # Q8_0: blocks of 34 bytes, fp16 d then 32 int8
                b = mm[off:off + rows * cols // 32 * 34].reshape(rows, cols // 32, 34)
                qs.append(b[..., 2:].reshape(rows, cols)), ds.append(b[..., :2].copy().view(np.float16)[..., 0])
            else:
                b = get(n).reshape(rows, cols // 32, 32)
                d = np.abs(b).max(axis=2, keepdims=True) / 127
                q = np.rint(np.divide(b, d, out=np.zeros_like(b), where=d > 0))
                qs.append(q.astype(np.int8).view(np.uint8).reshape(rows, cols))
                d = d[..., 0]
                ds.append(d.astype(np.float16))
        q, d16 = np.concatenate(qs), np.concatenate(ds)
        if all(tensors[n][1] == 8 for n in names):
            copied.append(key)
        put(key, q.tobytes() + d16.tobytes(), rows=q.shape[0], cols=q.shape[1], scales=q.nbytes, fmt=9)

    mat = q4 if chat else q8   # (the embedder in 8 bits: its vectors as llama.cpp gives them from the Q8_0 GGUF)
    mat("embed", "token_embd.weight")
    f32("final_norm", get("token_embd_norm.weight" if lfm else "output_norm.weight"))
    order = ["embed", "final_norm"]
    for i in range(n_layers):
        p = f"blk.{i}."
        f32(f"{i}.attn_norm", get(p + "attn_norm.weight"))
        f32(f"{i}.mlp_norm", get(p + "ffn_norm.weight"))
        if kvs[i]:
            f32(f"{i}.q_norm", get(p + "attn_q_norm.weight"))
            f32(f"{i}.k_norm", get(p + "attn_k_norm.weight"))
            mat(f"{i}.wqkv", p + "attn_q.weight", p + "attn_k.weight", p + "attn_v.weight")
            mat(f"{i}.wo", p + "attn_output.weight")
        else:   # LFM2 short conv, in the same 8 slots: conv weight [hidden][3], (empty), in_proj (B | C | x), out_proj
            f32(f"{i}.q_norm", get(p + "shortconv.conv.weight"))
            f32(f"{i}.k_norm", np.zeros(64))
            mat(f"{i}.wqkv", p + "shortconv.in_proj.weight")
            mat(f"{i}.wo", p + "shortconv.out_proj.weight")
        mat(f"{i}.wgu", p + "ffn_gate.weight", p + "ffn_up.weight")
        mat(f"{i}.wdown", p + "ffn_down.weight")
        order += [f"{i}.{k}" for k in ("attn_norm", "mlp_norm", "q_norm", "k_norm", "wqkv", "wo", "wgu", "wdown")]
        print(f"layer {i}: {w.tell() / 2**20:.0f} MiB", end="\r", flush=True)
    size = w.tell()
    w.close()
    eos = meta.get("tokenizer.ggml.eos_token_id", 151643)
    toks = meta["tokenizer.ggml.tokens"]
    # eos = the end of a turn, eos2 = the end of the text
    chat_seq = int(next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--ctx=")), CHAT_SEQ))
    assert chat_seq % 64 == 0, "--ctx: a multiple of 64 (shaders/attn.h ATT_BLOCK)"
    ends, seq = ((toks.index("<|im_end|>"), toks.index("<|endoftext|>")), chat_seq) if chat else ((151645, eos), EMB_SEQ)
    head = struct.pack("<4s10I2f128sQI", b"PML2" if lfm else b"PMDL", n_layers, hidden, meta[A + "feed_forward_length"],
                       n_heads, n_kv, head_dim, tensors["token_embd.weight"][0][1], seq, ends[0], ends[1],
                       meta[A + "attention.layer_norm_rms_epsilon"], meta[A + "rope.freq_base"], b"weights.bin", size,
                       len(order))
    if lfm:   # full_mask, im_start, bos, think, think_end (chat/engine.c load_index)
        head += struct.pack("<8I", full_mask, toks.index("<|im_start|>"), meta["tokenizer.ggml.bos_token_id"],
                            toks.index("<think>"), toks.index("</think>"), 0, 0, 0)
    (out / "model.idx").write_bytes(head + b"".join(struct.pack("<QQIIQI", *recs[k]) for k in order))
    if not chat:
        (out / "embedding.txt").write_text(json.dumps({"pooling": "last", "eos": eos, "max_tokens": EMB_SEQ}))
    print(f"\n{len(copied)} of {1 + 4 * n_layers} matrices copied as they are ({'Q4_0' if chat else 'Q8_0'})")
    print(f"\n{mid}/weights.bin: {size / 2**20:.0f} MiB ({n_layers} layers, hidden {hidden}, heads {n_heads}/{n_kv} x "
          f"{head_dim})")
    tokenizer(meta, out)
    kernels(out, [f"MAX_SEQ={seq}", "PREFILL"] + ([f"HEAD_DIM={head_dim}"] if head_dim != 128 else []))


if __name__ == "__main__":
    main()
