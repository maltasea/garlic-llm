"""Any Qwen3.5 size (0.8B-9B, Hugging Face bf16 safetensors) -> a model folder for the PS4 (chat/server.h MODELS_DIR):
  weights.bin  every matrix in q4s (shaders/spec.h; as convert_9b_q4s.py), lm_head (or the tied embedding) in the
               keep list's row order, then the input embedding rows (q4row, read by the CPU), then the ids record
  model.idx    PMQ5 (as convert_9b_q4s.py write_index), the weights path a bare name: the folder's weights.bin
  mtp.bin/.idx the MTP head when the checkpoint has one (as convert_9b_q4s.py --mtp)
All q4 (pc/q4.py), no q3. Needs numpy only; reads the safetensors files with numpy.memmap.

  .venv/bin/python pc/convert_model.py models/Qwen3.5-2B [--id qwen3.5-2b]    -> models/ps4/<id>/
  .venv/bin/python pc/convert_model.py ~/models/Qwen3.5-4B-Q5_K_M.gguf --id=qwen3.5-4b [--mtp=models/Qwen3.5-4B-mtp]
      (a GGUF: GGUFCheckpoint; its MTP head from pc/fetch_mtp.py)
"""
import json
import struct
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from q4 import pad128, q4_of, quantize_q4, to_q4s  # noqa: E402

FMT = {"q4s": 6, "ids": 7, "q4row": 4}   # (as convert_9b_q4s.py FMT)

KEEP = HERE / "vocab_order.npy"   # every token id, most frequent first (lm_head rows in this order)
ND = 32768                        # the MTP draft's rows: the first ND of lm_head
IM_START, IM_END, EOT = 248045, 248046, 248044


class Checkpoint:
    """The tensors of a safetensors checkpoint folder, as float32 numpy arrays (bf16 / f32 sources)."""

    def __init__(self, folder):
        self.folder = Path(folder)
        idx = self.folder / "model.safetensors.index.json"
        files = sorted({*json.loads(idx.read_text())["weight_map"].values()}) if idx.exists() else ["model.safetensors"]
        self.where = {}
        for f in files:
            p = self.folder / f
            with open(p, "rb") as fh:
                n = struct.unpack("<Q", fh.read(8))[0]
                head = json.loads(fh.read(n))
            mm = np.memmap(p, np.uint8, "r")
            for k, e in head.items():
                if k != "__metadata__":
                    self.where[k] = (mm, 8 + n, e)

    def __contains__(self, name):
        return name in self.where

    def __getitem__(self, name):
        mm, base, e = self.where[name]
        a, b = e["data_offsets"]
        raw = mm[base + a: base + b]
        if e["dtype"] == "BF16":
            return (raw.view(np.uint16).astype(np.uint32) << 16).view(np.float32).reshape(e["shape"])
        if e["dtype"] == "F32":
            return raw.view(np.float32).reshape(e["shape"]).copy()
        raise ValueError(f"{name}: dtype {e['dtype']}")


class GGUFCheckpoint:
    """A Qwen3.5 GGUF (llama.cpp's "qwen35", any of its quant types) as Checkpoint gives the safetensors: the Hugging
    Face names and layout. llama.cpp stores the norms as 1 + w, A_log as -exp(A_log), and, when there are more value
    heads than key heads, the value heads tiled (GGUF head j = HF head (j % n_kh) * r + j // n_kh, r = n_vh / n_kh);
    all of that is undone here. .config: the config.json fields main() reads."""

    def __init__(self, path):
        import convert_gguf
        self.meta, self.t, self.mm = convert_gguf.gguf(path)
        self.tensor = convert_gguf.tensor
        m, A = self.meta, "qwen35."
        nl = m[A + "block_count"] - m.get(A + "nextn_predict_layers", 0)
        hd, nk, nv = m[A + "attention.key_length"], m[A + "ssm.group_count"], m[A + "ssm.time_step_rank"]
        self.config = dict(
            model_type="qwen3_5_text", num_hidden_layers=nl, hidden_size=m[A + "embedding_length"],
            intermediate_size=m[A + "feed_forward_length"], num_attention_heads=m[A + "attention.head_count"],
            num_key_value_heads=m[A + "attention.head_count_kv"], head_dim=hd,
            rms_norm_eps=m[A + "attention.layer_norm_rms_epsilon"],
            rope_parameters=dict(rope_theta=m[A + "rope.freq_base"],
                                 partial_rotary_factor=m[A + "rope.dimension_count"] / hd),
            linear_num_key_heads=nk, linear_num_value_heads=nv, linear_key_head_dim=m[A + "ssm.state_size"],
            linear_value_head_dim=m[A + "ssm.inner_size"] // nv, linear_conv_kernel_dim=m[A + "ssm.conv_kernel"],
            layer_types=["full_attention" if (i + 1) % m[A + "full_attention_interval"] == 0 else "linear_attention"
                         for i in range(nl)],
            mtp_num_hidden_layers=m.get(A + "nextn_predict_layers", 0))
        self.nl, self.nk, self.nv = nl, nk, nv
        self.dk, self.dv = self.config["linear_key_head_dim"], self.config["linear_value_head_dim"]
        r = nv // nk   # HF value head h = GGUF head perm[h]
        self.perm = np.array([(h % r) * nk + h // r for h in range(nv)])

    def where(self, name):
        """(GGUF name, what to undo): "norm" (1 + w), "A" (-exp), "v_rows" / "v_cols" / "v" (value head order)."""
        P, L = "model.language_model.", "model.language_model.layers."
        if name == P + "embed_tokens.weight": return "token_embd.weight", None
        if name == P + "norm.weight": return "output_norm.weight", "norm"
        if name == "lm_head.weight": return "output.weight", None
        if name.startswith("mtp."):   # the MTP head: llama.cpp's "nextn" layer after the model's own
            b = f"blk.{self.nl}."
            top = {"mtp.fc.weight": ("nextn.eh_proj.weight", None),
                   "mtp.pre_fc_norm_embedding.weight": ("nextn.enorm.weight", "norm"),
                   "mtp.pre_fc_norm_hidden.weight": ("nextn.hnorm.weight", "norm"),
                   "mtp.norm.weight": ("nextn.shared_head_norm.weight", "norm")}
            if name in top: return b + top[name][0], top[name][1]
            name, i = L + "x." + name[len("mtp.layers.0."):], self.nl
        else:
            i = int(name[len(L):].split(".")[0])
        k = name[len(L):].split(".", 1)[1]
        b = f"blk.{i}."
        m = {"input_layernorm.weight": ("attn_norm.weight", "norm"),
             "post_attention_layernorm.weight": ("post_attention_norm.weight", "norm"),
             "mlp.gate_proj.weight": ("ffn_gate.weight", None), "mlp.up_proj.weight": ("ffn_up.weight", None),
             "mlp.down_proj.weight": ("ffn_down.weight", None),
             "self_attn.q_proj.weight": ("attn_q.weight", None), "self_attn.k_proj.weight": ("attn_k.weight", None),
             "self_attn.v_proj.weight": ("attn_v.weight", None), "self_attn.o_proj.weight": ("attn_output.weight", None),
             "self_attn.q_norm.weight": ("attn_q_norm.weight", "norm"),
             "self_attn.k_norm.weight": ("attn_k_norm.weight", "norm"),
             "linear_attn.A_log": ("ssm_a", "A"), "linear_attn.dt_bias": ("ssm_dt.bias", "v"),
             "linear_attn.in_proj_a.weight": ("ssm_alpha.weight", "v"),
             "linear_attn.in_proj_b.weight": ("ssm_beta.weight", "v"),
             "linear_attn.in_proj_qkv.weight": ("attn_qkv.weight", "qkv"),
             "linear_attn.in_proj_z.weight": ("attn_gate.weight", "v_rows"),
             "linear_attn.conv1d.weight": ("ssm_conv1d.weight", "qkv"),
             "linear_attn.norm.weight": ("ssm_norm.weight", None),
             "linear_attn.out_proj.weight": ("ssm_out.weight", "v_cols")}[k]
        return b + m[0], m[1]

    def __contains__(self, name):
        try:
            return self.where(name)[0] in self.t
        except (KeyError, ValueError):
            return False

    def __getitem__(self, name):
        g, undo = self.where(name)
        w = self.tensor(self.t, self.mm, g)
        if undo == "norm":
            return w - 1.0
        if undo == "A":
            return np.log(-w[self.perm])
        if undo == "v":   # one row per value head
            return w[self.perm]
        heads = lambda n: (self.perm[:, None] * self.dv + np.arange(self.dv)).reshape(-1) + n   # noqa: E731
        if undo == "v_rows":
            return w[heads(0)]
        if undo == "v_cols":
            return w[:, heads(0)]
        if undo == "qkv":   # q | k rows as they are, then the value heads
            qk = 2 * self.nk * self.dk
            return np.concatenate([w[:qk], w[heads(qk)]])   # (conv1d: [channels][taps], main() reshapes it)
        return w


class Both:
    """Tensors from a, else from b."""

    def __init__(self, a, b):
        self.a, self.b = a, b

    def __contains__(self, name):
        return name in self.a or name in self.b

    def __getitem__(self, name):
        return self.a[name] if name in self.a else self.b[name]


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    src = Path(args[0]).expanduser()
    mid = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--id=")), src.name.lower())
    out_dir = ROOT / "models" / "ps4" / mid   # (not next to the source: a Mac disk ignores case)
    out_dir.mkdir(parents=True, exist_ok=True)
    if src.suffix == ".gguf":
        ck = GGUFCheckpoint(src)
        c = ck.config
        mtp_dir = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--mtp=")), None)
        if mtp_dir:   # the MTP head from pc/fetch_mtp.py's folder, the rest from the GGUF
            ck = Both(Checkpoint(Path(mtp_dir).expanduser()), ck)
            c = dict(c, mtp_num_hidden_layers=1)
    else:
        cfg = json.loads((src / "config.json").read_text())
        c = cfg.get("text_config", cfg)
        ck = Checkpoint(src)
    assert c["model_type"] == "qwen3_5_text", c["model_type"]
    keep = np.load(KEEP).astype(np.int64)
    P = "model.language_model."
    nh, hd = c["num_attention_heads"], c["head_dim"]

    out = open(out_dir / "weights.bin", "wb")
    tensors = {}

    def put(key, data, **meta):
        tensors[key] = dict(offset=out.tell(), size=len(data), **meta)
        out.write(data)
        out.write(b"\0" * ((-out.tell()) % 256))

    def f32(key, w):
        put(key, np.ascontiguousarray(w, np.float32).tobytes(), cols=int(np.size(w)))

    def q4s(key, w, rows=None):
        q, d = pad128(*q4_of(np.ascontiguousarray(w, np.float32)))
        parts = [to_q4s(q[r:r + 8192], d[r:r + 8192]) for r in range(0, q.shape[0], 8192)]
        wb, sb = b"".join(p[0] for p in parts), b"".join(p[1] for p in parts)
        put(key, wb + sb, rows=q.shape[0], cols=w.shape[1], scales=len(wb), fmt="q4s")

    def wqkv(a):   # q_proj per head [query | gate] -> q | k | v | gate rows
        qg = ck[a + "q_proj.weight"].reshape(nh, 2, hd, -1)
        q, gate = qg[:, 0].reshape(nh * hd, -1), qg[:, 1].reshape(nh * hd, -1)
        return np.concatenate([q, ck[a + "k_proj.weight"], ck[a + "v_proj.weight"], gate])

    f32("final_norm", 1.0 + ck[P + "norm.weight"])
    for i, kind in enumerate(c["layer_types"]):
        p = f"{P}layers.{i}."
        f32(f"{i}.attn_norm", 1.0 + ck[p + "input_layernorm.weight"])
        f32(f"{i}.mlp_norm", 1.0 + ck[p + "post_attention_layernorm.weight"])
        if kind == "full_attention":
            a = p + "self_attn."
            f32(f"{i}.q_norm", 1.0 + ck[a + "q_norm.weight"])
            f32(f"{i}.k_norm", 1.0 + ck[a + "k_norm.weight"])
            q4s(f"{i}.wqkv", wqkv(a))
            q4s(f"{i}.wo", ck[a + "o_proj.weight"])
        else:
            a = p + "linear_attn."
            q4s(f"{i}.w_in", np.concatenate([ck[a + "in_proj_qkv.weight"], ck[a + "in_proj_z.weight"],
                                             ck[a + "in_proj_b.weight"], ck[a + "in_proj_a.weight"]]))
            f32(f"{i}.conv", ck[a + "conv1d.weight"].reshape(-1, c["linear_conv_kernel_dim"]))
            f32(f"{i}.A", -np.exp(ck[a + "A_log"]))
            f32(f"{i}.dt_bias", ck[a + "dt_bias"])
            f32(f"{i}.gnorm", ck[a + "norm.weight"])
            q4s(f"{i}.wout", ck[a + "out_proj.weight"])
        q4s(f"{i}.wgu", np.concatenate([ck[p + "mlp.gate_proj.weight"], ck[p + "mlp.up_proj.weight"]]))
        q4s(f"{i}.wdown", ck[p + "mlp.down_proj.weight"])
        print(f"layer {i} ({kind}): {out.tell() / 2**20:.0f} MiB", flush=True)
    emb = ck[P + "embed_tokens.weight"]
    head = emb if "lm_head.weight" not in ck else ck["lm_head.weight"]   # (tied: the embedding)
    q4s("lm_head", head[keep])
    tensors["lm_head"]["vocab"] = len(keep)
    gpu_size = out.tell()
    rows, cols = emb.shape   # input embedding, a record per row: nibbles then fp16 scales
    rowbytes = cols // 2 + cols // 32 * 2
    start = out.tell()
    for r0 in range(0, rows, 8192):
        packed, d16 = quantize_q4(emb[r0:r0 + 8192])
        out.write(np.concatenate([packed, d16.view(np.uint8)], axis=1).tobytes())
    tensors["embed"] = dict(offset=start, size=rows * rowbytes, rows=rows, cols=cols, scales=cols // 2,
                            rowbytes=rowbytes, fmt="q4row")
    out.write(b"\0" * ((-out.tell()) % 256))
    ids = keep.astype(np.uint32).tobytes()
    put("ids", ids, rows=len(keep), cols=1, scales=0, fmt="ids")
    size = out.tell()
    out.close()
    write_index(out_dir, dict(c, vocab_size=len(keep)), tensors, gpu_size)
    print(f"{mid}/weights.bin: {size / 2**20:.0f} MiB, GPU part {gpu_size / 2**20:.0f} MiB")
    if c.get("mtp_num_hidden_layers") and "mtp.fc.weight" in ck:
        mtp(out_dir, ck, c, keep, tensors["embed"], wqkv)


def write_index(out_dir, c, t, gpu_size):
    """model.idx: as convert_9b_q4s.py write_index, the weights a bare name (the folder's weights.bin)."""
    order, mask = ["embed", "final_norm"], 0
    for i, kind in enumerate(c["layer_types"]):
        full = kind == "full_attention"
        mask |= full << i
        mixer = ("q_norm", "k_norm", "wqkv", "wo") if full else ("w_in", "conv", "A", "dt_bias", "gnorm", "wout")
        order += [f"{i}.{k}" for k in ("attn_norm", "mlp_norm", *mixer, "wgu", "wdown")]
    order += ["lm_head", "ids"]
    rp = c["rope_parameters"]
    rope_dim = int(c["head_dim"] * rp.get("partial_rotary_factor", 1.0))
    head = struct.pack("<4s10I2f128sQI", b"PMQ5", c["num_hidden_layers"], c["hidden_size"], c["intermediate_size"],
                       c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"], c["vocab_size"], 8192,
                       IM_END, EOT, c["rms_norm_eps"], rp["rope_theta"], b"weights.bin", gpu_size, len(order))
    head += struct.pack("<8I", c["linear_num_key_heads"], c["linear_num_value_heads"], c["linear_key_head_dim"],
                        c["linear_value_head_dim"], rope_dim, mask, IM_START, 1 | 2 | 4)
    recs = [struct.pack("<QQIIQI", t[k]["offset"], t[k]["size"], t[k].get("rows", 0), t[k].get("cols", 0),
                        t[k].get("scales", 0), FMT.get(t[k].get("fmt"), 0)) for k in order]
    (out_dir / "model.idx").write_bytes(head + b"".join(recs))


MTP_ORDER = ["pre_e", "pre_h", "fc", "attn_norm", "mlp_norm", "q_norm", "k_norm", "wqkv", "wo", "wgu", "wdown", "norm"]


def mtp(out_dir, ck, c, keep, embed, wqkv):
    """mtp.bin + mtp.idx: as convert_9b_q4s.py mtp(), the draft rows from this folder's weights.bin."""
    out = open(out_dir / "mtp.bin", "wb")
    recs = {}

    def put(key, data, rows=0, cols=0, scales=0, fmt=0):
        recs[key] = (out.tell(), len(data), rows, cols, scales, fmt)
        out.write(data)
        out.write(b"\0" * ((-out.tell()) % 256))

    def f32(key, w):
        put(key, np.ascontiguousarray(w, np.float32).tobytes(), cols=w.size)

    def q4s(key, w):
        q, d = pad128(*q4_of(np.ascontiguousarray(w, np.float32)))
        wb, sb = to_q4s(q, d)
        put(key, wb + sb, rows=q.shape[0], cols=w.shape[1], scales=len(wb), fmt=6)

    P, L = "mtp.", "mtp.layers.0."
    f32("pre_e", 1.0 + ck[P + "pre_fc_norm_embedding.weight"])
    f32("pre_h", 1.0 + ck[P + "pre_fc_norm_hidden.weight"])
    q4s("fc", ck[P + "fc.weight"])
    f32("attn_norm", 1.0 + ck[L + "input_layernorm.weight"])
    f32("mlp_norm", 1.0 + ck[L + "post_attention_layernorm.weight"])
    f32("q_norm", 1.0 + ck[L + "self_attn.q_norm.weight"])
    f32("k_norm", 1.0 + ck[L + "self_attn.k_norm.weight"])
    q4s("wqkv", wqkv(L + "self_attn."))
    q4s("wo", ck[L + "self_attn.o_proj.weight"])
    q4s("wgu", np.concatenate([ck[L + "mlp.gate_proj.weight"], ck[L + "mlp.up_proj.weight"]]))
    q4s("wdown", ck[L + "mlp.down_proj.weight"])
    f32("norm", 1.0 + ck[P + "norm.weight"])
    wsize = out.tell()
    rb = embed["rowbytes"]
    with open(out_dir / "weights.bin", "rb") as f:   # the draft rows' input embeddings, in lm_head row order
        for tokid in keep[:ND]:
            f.seek(embed["offset"] + int(tokid) * rb)
            out.write(f.read(rb))
    tsize = out.tell() - wsize
    out.close()
    head = struct.pack("<4sI128sQQI", b"PMTP", ND, b"mtp.bin", wsize, tsize, len(MTP_ORDER))
    (out_dir / "mtp.idx").write_bytes(head + b"".join(struct.pack("<QQIIQI", *recs[k]) for k in MTP_ORDER))
    print(f"mtp.bin: tensors {wsize / 2**20:.1f} MiB + draft table {tsize / 2**20:.1f} MiB")


if __name__ == "__main__":
    main()
