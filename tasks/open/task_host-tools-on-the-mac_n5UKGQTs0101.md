# Task: Host tools on the Mac

  id: n5UKGQTs0101  
  status: open  

## Goal
`python3 build_chat.py` run from a Mac (or Linux) checkout with a `toolchain/` (or `GARLIC_TOOLCHAIN`) in place builds exactly as it does on Windows today: it resolves `zig`, `dotnet`, and the OpenOrbis `bin/<os>` tools for the host, runs PkgTool through dotnet, and stops before touching `build/` with one clear message naming each missing tool and its expected path. `LIBS` gains `-lSceNet` and `-lSceNetCtl`.

## Plan

1. **`build.py` — host resolution.** Keep `ROOT`, `TOOLCHAIN`, `OO`, `CFLAGS`, `ENV`, `run`, `pkgtool`, `ps4_ftp`, `PS4_IP` as they are. Add `HOST` (`"windows"` on `sys.platform == "win32"`, `"macos"` on `darwin`, else `"linux"`), `EXE` (`".exe"` only on Windows), then `ZIG = TOOLCHAIN/"zig"/("zig"+EXE)`, `DOTNET = TOOLCHAIN/"dotnet"/("dotnet"+EXE)`, `BIN = OO/"bin"/HOST`. On Windows every value stays byte-identical to today.
2. **`build.py` — `require_tools(*paths)`.** New function: exit non-zero with one message listing each missing path, the toolchain root it looked under, and "set GARLIC_TOOLCHAIN". No other behaviour change; `pkgtool` keeps calling `PkgTool.Core.dll` through `DOTNET` (decision 2, decisions.md).
3. **`build_chat.py` — tool names.** Use `b.BIN / f"create-fself{b.EXE}"` and `b.BIN / f"create-gp4{b.EXE}"` (lines 106, 127). Call `b.require_tools(b.ZIG, b.DOTNET, ...create-fself..., ...create-gp4...)` as the first statement of `main()`, before `shutil.rmtree(PKGDIR)` and any `mkdir`, so a missing tool leaves `build/` untouched. Add `"-lSceNet", "-lSceNetCtl"` to `LIBS`.
4. **Decided gaps.** No fallback from `bin/macos` to `bin/linux`: a wrong-ABI `create-fself` is worse than a clear error (decision 2's "if OpenOrbis ships no macOS binary … say so"). Compile order, object caching, assets, PKG names, `--4b`, `--full`, `--upload`, CFLAGS and every C source stay as they are. No C file, spec, test or README changes here; the full build with the real toolchain is the merge check, not this gate.
5. **What must not change:** the exported names above, `LIBS` order outside the two additions, Windows paths, and `build/chat/` output layout. Toolchain and models stay gitignored.

## Gate
```sh
python3 -m py_compile build.py build_chat.py
python3 -c "import build as b; assert b.HOST in ('windows','macos','linux'); assert b.EXE == ('.exe' if b.HOST=='windows' else ''); assert b.OO == b.TOOLCHAIN/'OpenOrbis'/'PS4Toolchain'; assert b.BIN == b.OO/'bin'/b.HOST; assert b.ZIG == b.TOOLCHAIN/'zig'/('zig'+b.EXE); assert b.DOTNET == b.TOOLCHAIN/'dotnet'/('dotnet'+b.EXE)"
python3 -c "import build_chat as c; assert {'-lSceNet','-lSceNetCtl'} <= set(c.LIBS)"
GARLIC_TOOLCHAIN="$PWD/.absent-toolchain" python3 build_chat.py 2>&1 | grep -q "GARLIC_TOOLCHAIN"
```

## Answers
- OpenOrbis v0.5.4 (llvm-18) ships `bin/macos`, and its tools run on this Mac (x86_64, checked against `$OO_PS4_TOOLCHAIN`). On macOS the fself tool is `create-fself-macos`, not `create-fself`, so `build_chat.py` picks that name when `HOST == "macos"`. The no-fallback rule stays.
