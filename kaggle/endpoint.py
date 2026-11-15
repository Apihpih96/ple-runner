"""Find the live Kaggle Gemma endpoint, if any.

Reads the newest encrypted message on the ntfy topic, opens it with the private
key in env KAGGLE_LLM_PRIVKEY, checks the server answers, and prints shell
lines: KAGGLE_LLM_URL=..., KAGGLE_LLM_KEY=..., KAGGLE_LLM_CTX=...
(nothing when no live endpoint). With --github it writes them to $GITHUB_ENV
and masks them instead of printing. Exit code 0 = live endpoint found, 1 = none.
"""
import base64, json, os, sys, urllib.request

NTFY_TOPIC = "ple-gemma-672d4d5a44f2e35d847f320d"


def main():
    from nacl.public import PrivateKey, SealedBox
    priv = os.environ.get("KAGGLE_LLM_PRIVKEY", "").strip()
    if not priv:
        print("no KAGGLE_LLM_PRIVKEY", file=sys.stderr)
        return 1
    box = SealedBox(PrivateKey(base64.b64decode(priv)))
    raw = urllib.request.urlopen(f"https://ntfy.sh/{NTFY_TOPIC}/json?poll=1&since=12h", timeout=20).read().decode()
    msgs = [json.loads(l) for l in raw.splitlines() if l.strip()]
    msgs = [m for m in msgs if m.get("event") == "message"]
    if not msgs:
        print("no endpoint published in the last 12 h", file=sys.stderr)
        return 1
    m = max(msgs, key=lambda m: m["time"])
    info = json.loads(box.decrypt(base64.b64decode(m["message"])))
    if not info.get("url"):
        print("Kaggle server has stopped", file=sys.stderr)
        return 1
    try:
        req = urllib.request.Request(info["url"] + "/health")
        ok = json.loads(urllib.request.urlopen(req, timeout=20).read()).get("status") == "ok"
    except Exception as e:  # noqa
        ok = False
    if not ok:
        print("published Kaggle endpoint does not answer", file=sys.stderr)
        return 1
    out = {"KAGGLE_LLM_URL": info["url"], "KAGGLE_LLM_KEY": info["key"], "KAGGLE_LLM_CTX": str(info["ctx"])}
    print(f"Kaggle endpoint live: ctx {info['ctx']}, speed {info.get('speed')}", file=sys.stderr)
    if "--github" in sys.argv:
        for k in ("KAGGLE_LLM_URL", "KAGGLE_LLM_KEY"):
            print(f"::add-mask::{out[k]}")
        with open(os.environ["GITHUB_ENV"], "a") as f:
            for k, v in out.items():
                f.write(f"{k}={v}\n")
    else:
        for k, v in out.items():
            print(f"{k}={v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
