#!/usr/bin/env python3
"""guard_preflight.py - shared, non-intrusive pre-push safety check for all Grok bots.

What it checks (on the STAGED changes, `git diff --cached`):
  (a) only A (added) entries; at most --max-files N files (default 1)
  (b) no staged path under .github/
  (c) origin is not a protected repo (JuniorSolana, cloudcover95/cloudcover95)
  (d) added content has no '0.0.0.0' bind and no obvious secrets
      (ghp_, gho_, github_pat_, AKIA..., -----BEGIN ... PRIVATE KEY)
  (e) no staged file > 1 MB and no .gguf/.safetensors/.bin model files
  (f) branch up to date with upstream after `git fetch` (WARN only)
  (g) --rate-check: added files mentioning 'rate' should use 0.05; WARN otherwise

What it does NOT touch: it never stages, commits, pushes, edits, or deletes
anything (only `git fetch` updates remote-tracking refs). It never blocks by
itself: it prints PASS/FAIL/WARN lines and exits 0 on pass, 1 on any FAIL.
The calling bot decides to stop and report.

How to run (from the repo root, before `git push`):
  python3 grok_bot/guard_preflight.py [--max-files N] [--rate-check]
  python3 grok_bot/guard_preflight.py --selftest
Stdlib only.
"""
import argparse, os, re, subprocess, sys, tempfile

PROTECTED = ("juniorsolana", "cloudcover95/cloudcover95")
SECRET_RE = re.compile(r"(ghp_[A-Za-z0-9]{20,}|gho_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}"
                       r"|AKIA[0-9A-Z]{16}|-----BEGIN .*PRIVATE KEY)")
BIND_RE = re.compile(r"0\.0\.0\.0")
MODEL_EXT = (".gguf", ".safetensors", ".bin")
MAX_BYTES = 1024 * 1024
RATE_RE = re.compile(r"rate[^\n]{0,40}?(\d+\.\d+)|(\d+\.\d+)[^\n]{0,20}?rate", re.I)


def git(args, cwd):
    p = subprocess.run(["git"] + args, cwd=cwd, capture_output=True, text=True)
    return p.returncode, p.stdout, p.stderr


def run(cwd=".", max_files=1, rate_check=False, fetch=True, out=print):
    fails = 0

    def res(ok, name, detail=""):
        nonlocal fails
        if not ok:
            fails += 1
        out(f"{'PASS' if ok else 'FAIL'} {name}{': ' + detail if detail else ''}")

    def warn(name, detail):
        out(f"WARN {name}: {detail}")

    rc, ns, err = git(["diff", "--cached", "--name-status"], cwd)
    if rc:
        res(False, "git", err.strip())
        return 1
    entries = [l.split("\t") for l in ns.splitlines() if l.strip()]
    non_add = [e for e in entries if e[0] != "A"]
    paths = [e[-1] for e in entries]
    added = [e[1] for e in entries if e[0] == "A"]
    res(bool(entries), "a.staged-nonempty", f"{len(entries)} entries")
    res(not non_add, "a.additive-only", "; ".join("\t".join(e) for e in non_add) or "all A")
    res(len(entries) <= max_files, "a.max-files", f"{len(entries)}/{max_files}")
    gh = [p for p in paths if p == ".github" or p.startswith(".github/")]
    res(not gh, "b.no-.github", ", ".join(gh) or "none")
    rc, url, _ = git(["remote", "get-url", "origin"], cwd)
    u = url.strip().lower().rstrip("/")
    if u.endswith(".git"):
        u = u[:-4]
    bad = [p for p in PROTECTED if u.endswith(p) or ("/" not in p and u.rsplit("/", 1)[-1] == p)]
    res(not bad, "c.not-protected-repo", url.strip() or "(no origin)")
    sec, bind, big, model, rate = [], [], [], [], []
    for p in added:
        rc, blob, _ = git(["cat-file", "-s", f":{p}"], cwd)
        size = int(blob.strip() or 0)
        if size > MAX_BYTES:
            big.append(f"{p}({size}B)")
        if p.lower().endswith(MODEL_EXT):
            model.append(p)
        rc, txt, _ = git(["show", f":{p}"], cwd)
        for i, line in enumerate(txt.splitlines(), 1):
            if SECRET_RE.search(line):
                sec.append(f"{p}:{i}")
            if BIND_RE.search(line):
                bind.append(f"{p}:{i}")
            if rate_check:
                for m in RATE_RE.finditer(line):
                    v = m.group(1) or m.group(2)
                    if v != "0.05":
                        rate.append(f"{p}:{i}={v}")
    # skip self-reference: this file legitimately contains the patterns it scans for
    me = "grok_bot/guard_preflight.py"
    sec = [s for s in sec if not s.startswith(me + ":")]
    bind = [s for s in bind if not s.startswith(me + ":")]
    rate = [s for s in rate if not s.startswith(me + ":")]
    res(not bind, "d.no-0.0.0.0-bind", ", ".join(bind) or "none")
    res(not sec, "d.no-secrets", ", ".join(sec) or "none")
    res(not big, "e.size<=1MB", ", ".join(big) or "ok")
    res(not model, "e.no-model-files", ", ".join(model) or "none")
    if fetch:
        rc, _, err = git(["fetch", "--quiet"], cwd)
        if rc:
            warn("f.upstream", "fetch failed: " + err.strip()[:200])
        else:
            rc, cnt, _ = git(["rev-list", "--left-right", "--count", "HEAD...@{u}"], cwd)
            if rc:
                warn("f.upstream", "no upstream configured")
            else:
                ahead, behind = cnt.split()
                if behind != "0":
                    warn("f.upstream", f"behind upstream by {behind} (ahead {ahead}); pull before push")
                else:
                    out(f"PASS f.upstream: up to date (ahead {ahead})")
    if rate_check:
        if rate:
            warn("g.rate-check", "non-0.05 rate literal near 'rate': " + ", ".join(rate))
        else:
            out("PASS g.rate-check: no non-0.05 rate literals")
    out(f"RESULT {'PASS' if not fails else 'FAIL'} ({fails} fail)")
    return 0 if not fails else 1


def selftest():
    def mk(origin="https://github.com/cloudcover95/JuniorLLM.git"):
        d = tempfile.mkdtemp(prefix="guardst_")
        for a in (["init", "-q"], ["config", "user.email", "t@t"], ["config", "user.name", "t"],
                  ["remote", "add", "origin", origin]):
            git(a, d)
        with open(os.path.join(d, "base.txt"), "w") as f:
            f.write("base\n")
        git(["add", "base.txt"], d)
        git(["commit", "-q", "--no-gpg-sign", "-m", "base"], d)
        return d

    def stage(d, name, content):
        p = os.path.join(d, name)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w" if isinstance(content, str) else "wb") as f:
            f.write(content)
        git(["add", name], d)

    cases = []
    d = mk(); stage(d, "ok.txt", "hello\n"); cases.append(("pass: one added file", d, {}, 0))
    d = mk(); stage(d, "a.txt", "1\n"); stage(d, "b.txt", "2\n"); cases.append(("fail: 2 files > max 1", d, {}, 1))
    d = mk(); stage(d, "a.txt", "1\n"); stage(d, "b.txt", "2\n"); cases.append(("pass: 2 files, --max-files 2", d, {"max_files": 2}, 0))
    d = mk(); stage(d, "base.txt", "changed\n"); cases.append(("fail: modified file", d, {}, 1))
    d = mk(); git(["rm", "-q", "base.txt"], d); cases.append(("fail: deleted file", d, {}, 1))
    d = mk(); stage(d, ".github/workflows/x.yml", "on: push\n"); cases.append(("fail: .github path", d, {}, 1))
    d = mk("git@github.com:cloudcover95/JuniorSolana.git"); stage(d, "ok.txt", "x\n"); cases.append(("fail: JuniorSolana", d, {}, 1))
    d = mk("https://github.com/cloudcover95/cloudcover95"); stage(d, "ok.txt", "x\n"); cases.append(("fail: cloudcover95/cloudcover95", d, {}, 1))
    d = mk(); stage(d, "s.py", "app.run(host='0.0.0.0')\n"); cases.append(("fail: 0.0.0.0 bind", d, {}, 1))
    d = mk(); stage(d, "s.env", "T=ghp_" + "a" * 36 + "\n"); cases.append(("fail: ghp_ token", d, {}, 1))
    d = mk(); stage(d, "k.pem", "-----BEGIN RSA " + "PRIVATE KEY-----\n"); cases.append(("fail: private key", d, {}, 1))
    d = mk(); stage(d, "big.txt", "x" * (MAX_BYTES + 10)); cases.append(("fail: >1MB", d, {}, 1))
    d = mk(); stage(d, "m.gguf", b"\0"); cases.append(("fail: .gguf model", d, {}, 1))
    d = mk(); stage(d, "r.py", "fee_rate = 0.07\n"); cases.append(("pass(warn): rate-check 0.07", d, {"rate_check": True}, 0))
    ok = True
    for name, d, kw, want in cases:
        lines = []
        got = run(d, fetch=False, out=lines.append, **kw)
        good = got == want
        if name.startswith("pass(warn)"):
            good = good and any(l.startswith("WARN g.rate-check") for l in lines)
        ok &= good
        print(f"{'PASS' if good else 'FAIL'} selftest {name} (exit {got}, want {want})")
        if not good:
            print("   " + "\n   ".join(lines))
    print(f"SELFTEST {'PASS' if ok else 'FAIL'} ({len(cases)} cases)")
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Read-only pre-push guard for Grok bots.")
    ap.add_argument("--max-files", type=int, default=1)
    ap.add_argument("--rate-check", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--no-fetch", action="store_true", help="skip check (f)")
    a = ap.parse_args()
    sys.exit(selftest() if a.selftest else run(".", a.max_files, a.rate_check, not a.no_fetch))
