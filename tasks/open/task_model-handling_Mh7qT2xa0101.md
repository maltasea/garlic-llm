# Task: Model handling

  id: Mh7qT2xa0101  
  status: open  

## Goal
Models are folders on the PS4 that Ben converts on the Mac and uploads; the server lists them and loads the one a
request names. No rebuild, no reinstall to change or add a model.

## Plan
1. **Model folders.** `/data/models/<id>/`: `model.idx`, `weights.bin` (the idx's path becomes relative to the
   folder), optional `mtp.idx` + `mtp.bin` (the MTP head), optional `tokenizer.bin` (else the package's). The app
   starts with the model named in `/data/models/default.txt`, else the first folder, else the built-in one (today's
   `/data/qwen3.5-9b-q4s.bin` with the package's idx files), so the current install keeps working.
2. **Reload without restart.** `engine_load(e, dir)`: keeps the GPU block, the kernels and the self-test; drops
   weights, caches and the MTP head and loads the new folder (~40 s for the 9B, less for small ones); engine_ctx and
   spec_start again. On failure the server falls back to the model it had.
3. **API.** `/v1/models` lists the folders (the loaded one first). A request whose `model` is another folder loads
   it first (the screen says so); an unknown id gives 404 `model_not_found`. No `model` = the loaded one.
4. **Converter on the Mac.** `pc/convert_model.py <HF repo or local folder> [--id name]`: any Qwen3.5 size
   (0.8B–9B) from the bf16 safetensors, read tensor by tensor; every matrix q4s (as `convert_9b_q4s.py`), embedding
   rows on disk (q4row), lm_head trimmed by the keep list, MTP head when the checkpoint has one. Writes
   `models/<id>/` with the files of step 1. Needs numpy in a project venv (`.venv`, gitignored).
5. **Upload.** `pc/upload_model.py <folder>`: the folder's files to `/data/models/<id>/` over FTP (`.part`, renamed
   when complete; size checked with LIST, as today).

## Gate
Convert Qwen3.5-2B on the Mac, upload it, a request with `"model": "qwen3.5-2b"` answers from it, the next one with
`"model": "qwen3.5-9b"` from the 9B again, without restarting the app; `/v1/models` lists both; the contract tests in
`wrk/openflow/.../project/tests` pass.

## Open
- Done on the PS4 (2026-11-14): folders, /v1/models, switching 9B -> 2B, the embedder (pc/convert_gguf.py,
  /v1/embeddings). Not yet: switching back 2B -> 9B crashes after the load (the log stops after "GPU memory used").
- Whether every Qwen3.5 size fits the kernels' shapes (head_dim 256, DeltaNet head counts): checked per size on the
  console in step 4's first run.
