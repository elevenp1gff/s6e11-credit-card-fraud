# Environment pitfalls that cost real time

Everything here is something that silently broke a run. None of it is obvious from
an error message you would recognise.

---

## Windows / PowerShell

### `Out-File -Encoding utf8` writes a BOM
UTF-8 with BOM (`EF BB BF`) makes Python fail in ways that look like a logic bug.

```powershell
# WRONG
Set-Content -Path $f -Value $txt -Encoding UTF8

# RIGHT
[IO.File]::WriteAllText($f, $txt, (New-Object Text.UTF8Encoding($false)))
```

I found **35 files** in one project carrying a BOM. Detect:

```powershell
$b = [IO.File]::ReadAllBytes($p) | Select-Object -First 3
if ($b[0] -eq 239 -and $b[1] -eq 187 -and $b[2] -eq 191) { "has BOM" }
```

### `NO_PROXY` containing `[::1]` breaks httpx
```
httpx.InvalidURL: Invalid port: ':1]'
```
This aborts every HuggingFace download, including `from_pretrained("gpt2")`. The value
must be `::1`, not `[::1]`. Deleting the variable entirely also works.

### `Start-Process` fails when `NO_PROXY` and `no_proxy` both exist
```
ArgumentException: 已添加项。字典中的关键字:"NO_PROXY"所添加的关键字:"no_proxy"
```
Use `cmd /c "start /B python script.py > out.txt 2>&1"` instead.

### `timeout` in PowerShell is the Windows command, not coreutils
`timeout 300 python x.py` → `ERROR: Invalid syntax. Default option is not allowed more than '1' time(s).`

### `scp` silently truncates text files
I have seen `StandardScaler` arrive on a server as `StandardScale`. There is no error.
Always transfer source via base64 and verify line count:

```powershell
$b64 = [Convert]::ToBase64String([IO.File]::ReadAllBytes($local))
$b64 | Out-File -Encoding ASCII -NoNewline x.b64
scp x.b64 user@host:/path/
```
```bash
base64 -d x.b64 > target.py && wc -l target.py && grep -c "<function_name>" target.py
```

### Inline SSH commands mangle quotes and parentheses
```powershell
ssh host 'python -c "print(1)"'          # breaks
ssh host 'grep -c "def foo" x.py'        # breaks
```
Write the command to a file, scp it, then execute. This was the cause of roughly a third
of my failed attempts.

---

## Python / numpy

### `bottleneck` and `numexpr` compiled against numpy 1.x
Under numpy 2.x they raise:
```
AttributeError: _ARRAY_API not found
```
and if imported indirectly by pandas, **the exception can kill a long training run**.
Guard before importing pandas:

```python
import sys
sys.modules["bottleneck"] = None
sys.modules["numexpr"] = None
import pandas as pd
```

This must be in every entry point, including `python -c` one-liners.

### pandas uses `StringDtype`, not `object`
```python
[c for c in df.columns if df[c].dtype == object]     # misses strings
[c for c in df.columns if df[c].dtype.kind in "OUST"]  # correct
```

### Truncated `.npy` raises on load, not on access
`ValueError: Failed to read all data for array`. Always wrap loads:

```python
def safe_load(p, n):
    try:
        a = np.load(p)
        return a if a.shape[0] == n else None
    except Exception:
        return None
```

---

## Huawei Ascend NPU

### `import torch_npu` must be explicit
Without it, `torch.device("npu")` is rejected. And `source /usr/local/Ascend/ascend-toolkit/set_env.sh`
must run in every new shell.

### `nn.Linear(n, 1)` raises a oneDNN error on CPU
```
RuntimeError: could not create a primitive descriptor for a matmul primitive
```
in torch 2.1.0 aarch64. Width ≥ 2 works. Workaround for CPU fallback: make the final
layer width 2 and slice `[:, 0]`.

### `Conv2d` on CPU is ~15× slower than `Conv1d`
3.5 s vs 0.24 s per step on the same workload. Use `Conv1d` even for 2-D structure.

### `conv` over the wrong axis is a silent, total failure
`"a shared CNN reads decimal digits for 13 numeric fields"` means convolve **along the
digit axis, shared across fields**. I first convolved *across fields*, whose order is
arbitrary — the layer learned a bias over meaningless ordering and modelled no digit
pattern at all. The model still trained to 0.944, so nothing looked broken.

### A learning-rate schedule that never anneals
Every early run built `OneCycleLR(total_steps=60 epochs)` but early-stopped at epoch 11–19.
OneCycle peaks at 30% of budget, so stopping fired **near peak LR** and the network was
never evaluated at low LR. My claim of "60 epochs, patience 10" was false — the run never
approached epoch 60.

If val AUC peaks at epoch 6 and decays to 0.898 by epoch 51, that is memorisation of
11,689 positives, and no amount of dropout or schedule tuning fixes it.

### Multi-process writes to the same `.npz` corrupt it
`zipfile.BadZipFile: File is not a zip file`. Build caches serially before forking workers.

### N processes each loading a 441 MB design matrix will OOM the box
Batch concurrent workers to ≤ 4 on a 32 GB machine, or use a memmap.

### `nohup bash -c "nohup ..."` double-wrapping kills the process instantly
Use a runner script with an exported function:

```bash
run_one() {
  local F=$1
  python script.py "$F" > "log_$F.txt" 2>&1
}
export -f run_one
for f in 0 1 2; do run_one "$f" & done
wait
```

### `pkill` returning non-zero aborts the whole SSH command
`pkill -f x; true` or `pkill -f x || true`.

### `$(date)` inside a quoted SSH string is evaluated by the local shell
Use fixed filenames.

---

## Kaggle API

### `IntrospectToken` goes down for minutes at a time
```
SSLError: EOF occurred in violation of protocol
MaxRetryError: /v1/security.OAuthService/IntrospectToken
```
Every authenticated action fails, including `datasets download` and `competitions submit`.
It recovers on its own. Wrap in a retry loop of 8+ attempts with 10 s sleeps; do not
conclude the credentials are broken.

### Dataset downloads corrupt silently
`kaggle datasets download -d <slug>` for a large dataset can produce 0 bytes or 4 MB
instead of 260 MB, with no error. Download individual files and verify size:

```powershell
kaggle datasets download -d <slug> -f <file> -p <dir>
if ((Get-Item $p).Length -lt $threshold) { retry }
```

### Submission success is not always printed
Verify with `kaggle competitions submissions -c <comp>` rather than trusting the CLI's
stdout.

### The leaderboard you cache is a lie
I once reported rank 82 from an 11-02 file while the competition had 886 teams. Always
re-download:

```powershell
kaggle competitions leaderboard -c <comp> --download -p <fresh-dir>
```

The CLI syntax for discussions is also non-obvious: it is
`kaggle competitions topics show <competition> <topic-id>` (two positional args), not `-c`.

---

## A methodological trap worth more than any of the above

On a dataset with 11,689 positives, **OOF differences of 0.0001–0.0003 are sampling
noise**. Fitting weights, selecting members, or choosing a blend on that signal overfits
reliably. I confirmed this ten times:

| approach | OOF | public LB |
|---|---|---|
| 151-member pool (self-built) | 0.952661 | 0.95234 |
| pure public stack | 0.952504 | 0.95267 |
| 0.6·mine + 0.4·public | 0.952765 | 0.95285 |

The highest OOF produced the worst leaderboard score. Every fitted-weight scheme
(NNLS, coordinate ascent, greedy, LR stacking, residual regression) lost to a single
member at a fixed weight.

Use OOF to reject broken members, never to rank healthy ones.

---

## Making figures that survive an audit

I had my figures professionally audited and the first version failed 6 of 8 checks. The
failures were not exotic; they are what happens when plots are built ad hoc.

### A single font-size ladder, or nothing
The first version used **13 different sizes for one semantic role** (annotations from 5.6
to 10.2 pt). Readers cannot tell whether a size difference means "less important" or "the
author nudged it". Put every size in one dictionary and never write a literal:

`python
SIZE = dict(panel=9.5, title=9.0, axis=8.5, tick=7.5,
            legend=7.0, annot=6.5, footer=6.5)
`

### ontweight="bold" is a silent no-op with SimSun
SimSun (宋体) has **no bold face**. indfont warns once per (family, weight) pair and
falls back to weight 400 — the text renders identically. I verified this by counting ink
pixels: normal and bold gave the same 2016/2016 count.

If you need real emphasis with Chinese text, use a family that *has* a heavy weight:

`python
plt.rcParams["font.sans-serif"] = ["SimSun"]        # body
ax.annotate("a", fontfamily="SimHei", ...)          # panel letters: real bold
`

### Reserve a band for the caption; do not overlay it
ig.text(y=0.01, ...) lands on top of the axes unless the gridspec reserves room:

`python
gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.28], hspace=0.07)
# then grow figsize by the same fraction so the axes keep their height
`

I tried removing the band to "save whitespace" and the captions landed on the data.

### Position labels in points, not data units
xytext with 	extcoords="offset points" survives axis rescaling; data-coordinate text
drifts into marks as soon as a limit changes.

`python
ax.annotate("label", (x, y), xytext=(11, -3), textcoords="offset points")
`

### Panel letters at a fixed offset
Axes-fraction offsets (dx=-0.15) put the letters in different places per figure — the
audit measured a **77.9 pt spread**. Anchoring to (0,1) in axes coords with a fixed point
offset gives 0.00 pt spread:

`python
ax.annotate("a", xy=(0, 1), xycoords="axes fraction",
            xytext=(-22, 12), textcoords="offset points", ...)
`

### Fewer callouts, not cleverer offsets
Four labels under a rising curve cannot be separated by tuning offsets. If labels collide,
**delete one** — the figure usually says more with two well-placed labels than four
cramped ones.

### Count the labels before you trust a number in prose
A caption said "950 levels, 949 thresholds" while the data had **949 levels, 948
thresholds**. The number had been typed by hand. Derive it:

`python
NLEV = len(np.unique(np.round(values, 1)))
ax.annotate(f"{NLEV} levels → {NLEV-1} thresholds", ...)
`

### Watch for phantom tick labels
matplotlib creates tick labels outside the axis limits that are never drawn. Any bbox-based
overlap checker will report them as collisions. Filter by a paint test (toggle the artist
off, diff the canvas) before believing an overlap count.
