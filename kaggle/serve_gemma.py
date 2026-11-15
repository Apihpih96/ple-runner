"""Serve Gemma 4 26B (A4B MoE, Q4_0 QAT GGUF) on a Kaggle GPU T4 x2 notebook.

Run from a Kaggle notebook cell (GPU T4 x2, Internet on):
    !curl -fsSL https://raw.githubusercontent.com/Apihpih96/ple-runner/main/kaggle/serve_gemma.py -o serve.py && python serve.py

What it does:
  1. llama.cpp server (OpenAI-compatible /v1) with the model split over both T4s,
     the longest context that fits, and a random API key made here.
  2. A Cloudflare quick tunnel to it (https://<random>.trycloudflare.com).
  3. Publishes {url, key, ctx, speed} *encrypted* to the dev runner's public key
     (sealed box) on an ntfy.sh topic. Only the runner's private key (a GitHub
     secret) can read it. Nothing secret is printed or stored in this notebook.
  4. Stops by itself after IDLE_MIN minutes with no requests or MAX_HOURS total,
     so it does not burn the weekly GPU quota.
"""
import base64, json, os, re, subprocess, sys, tarfile, threading, time, urllib.request, secrets, shutil

RUNNER_PUBKEY = "FtKpERLC/1Lyj7YrW+qZYGWhk2kljuA0Vu9mRZvIoR0="
NTFY_TOPIC = "ple-gemma-672d4d5a44f2e35d847f320d"
LLAMA_TAG = os.environ.get("LLAMA_TAG", "b12405")
MODEL_REPO = "google/gemma-4-26B-A4B-it-qat-q4_0-gguf"
MODEL_FILE = "gemma-4-26B_q4_0-it.gguf"
CTX_TRY = [int(x) for x in os.environ.get("CTX_TRY", "131072,98304,65536,32768").split(",")]
IDLE_MIN = float(os.environ.get("IDLE_MIN", "75"))
MAX_HOURS = float(os.environ.get("MAX_HOURS", "11.6"))
PORT = 8080
WORK = "/tmp/gemma"
T0 = time.time()
os.makedirs(WORK, exist_ok=True)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def sh(cmd, **kw):
    return subprocess.run(cmd, shell=True, check=True, **kw)


def pip(*pkgs):
    sh(f"{sys.executable} -m pip install -q " + " ".join(pkgs))


# ---------- 1. llama.cpp ----------
def get_llama():
    bin_dir = f"{WORK}/llama"
    exe = shutil.which("llama-server", path=bin_dir) or ""
    if not exe:
        base = f"https://github.com/ggml-org/llama.cpp/releases/download/{LLAMA_TAG}"
        for name in (f"llama-{LLAMA_TAG}-bin-ubuntu-cuda-12.8-x64.tar.gz",
                     f"cudart-llama-{LLAMA_TAG}-bin-ubuntu-cuda-12.8-x64.tar.gz"):
            log("download", name)
            urllib.request.urlretrieve(f"{base}/{name}", f"{WORK}/{name}")
            os.makedirs(bin_dir, exist_ok=True)
            with tarfile.open(f"{WORK}/{name}") as t:
                t.extractall(bin_dir)
        found = [os.path.join(r, f) for r, _, fs in os.walk(bin_dir) for f in fs if f == "llama-server"]
        exe = found[0]
    d = os.path.dirname(exe)
    os.environ["LD_LIBRARY_PATH"] = d + ":" + os.environ.get("LD_LIBRARY_PATH", "")
    os.chmod(exe, 0o755)
    subprocess.run("nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader; "
                   "nvcc --version | tail -1; ls -d /usr/local/cuda* 2>/dev/null", shell=True)
    r = subprocess.run([exe, "--list-devices"], capture_output=True, text=True)
    out = r.stdout + r.stderr
    log("list-devices rc", r.returncode, "\n" + out[-1200:])
    if "CUDA0" in out:
        return exe
    # Prebuilt binary cannot see the T4s: build for sm_75 with Kaggle's nvcc.
    log("prebuilt CUDA build not usable, compiling llama.cpp for sm_75 (~10-15 min)")
    src = f"{WORK}/llama.cpp"
    if not os.path.exists(src):
        sh(f"git clone -q --depth 1 --branch {LLAMA_TAG} https://github.com/ggml-org/llama.cpp {src}")
    cuda = "/usr/local/cuda"
    # Kaggle's image has no libcuda.so in the toolkit dirs cmake searches; point it at the stubs
    # and the driver's lib so the CUDA::cuda_driver target resolves.
    stubs = f"{cuda}/lib64/stubs"
    drv = subprocess.run("ldconfig -p | grep -m1 'libcuda.so ' | awk '{print $NF}'", shell=True,
                         capture_output=True, text=True).stdout.strip()
    extra = f"-DCUDAToolkit_ROOT={cuda} -DCMAKE_CUDA_COMPILER={cuda}/bin/nvcc -DCMAKE_LIBRARY_PATH={stubs}"
    if drv:
        extra += f" -DCUDA_cuda_driver_LIBRARY={drv}"
    elif os.path.exists(f"{stubs}/libcuda.so"):
        extra += f" -DCUDA_cuda_driver_LIBRARY={stubs}/libcuda.so"
    sh(f"cd {src} && cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75 -DLLAMA_CURL=OFF "
       f"-DLLAMA_BUILD_TESTS=OFF -DCMAKE_BUILD_TYPE=Release {extra} 2>&1 | tail -25 "
       f"&& cmake --build build -j$(nproc) --target llama-server 2>&1 | grep -E 'error|Error' | head -20; "
       f"test -x build/bin/llama-server")
    return f"{src}/build/bin/llama-server"


# ---------- 2. model ----------
def get_model():
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    try:
        import hf_transfer  # noqa
    except ImportError:
        pip("hf_transfer")
    from huggingface_hub import hf_hub_download
    log("download model", MODEL_FILE)
    return hf_hub_download(MODEL_REPO, MODEL_FILE, local_dir=f"{WORK}/model")


def get(url, key=None, data=None, timeout=10):
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    if key:
        req.add_header("Authorization", f"Bearer {key}")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read() or b"{}")


def start_server(exe, model, key):
    for ctx in CTX_TRY:
        args = [exe, "-m", model, "--host", "127.0.0.1", "--port", str(PORT), "--api-key", key,
                "-ngl", "999", "--split-mode", "layer", "-c", str(ctx), "-np", "1",
                "-fa", "on", "--jinja", "--metrics", "--no-webui", "-a", "gemma-4-26b",
                "-b", "2048", "-ub", "512"]
        log(f"start llama-server ctx={ctx}")
        logf = open(f"{WORK}/server.log", "w")
        p = subprocess.Popen(args, stdout=logf, stderr=subprocess.STDOUT)
        for _ in range(600):
            if p.poll() is not None:
                break
            try:
                if get(f"http://127.0.0.1:{PORT}/health").get("status") == "ok":
                    log("server ready")
                    return p, ctx
            except Exception:
                pass
            time.sleep(2)
        tail = open(f"{WORK}/server.log").read()[-1500:]
        log("server failed at ctx", ctx, "\n", re.sub(r"[A-Za-z0-9_-]{40,}", "<redacted>", tail))
        p.kill()
        p.wait()
    raise SystemExit("llama-server could not start at any context size")


# ---------- 3. speed ----------
def bench(key):
    filler = ("The diaphragm transfers lateral wind load to the shear walls through the chord "
              "and collector elements; check the unit shear against the allowable value. ") * 220
    t = time.time()
    r = get(f"http://127.0.0.1:{PORT}/v1/chat/completions", key, json.dumps({
        "model": "gemma-4-26b", "max_tokens": 200, "temperature": 0,
        "messages": [{"role": "user", "content": filler + "\nSummarize the above in 150 words."}]}).encode(), timeout=900)
    tm = r.get("timings", {})
    res = {"prompt_tokens": r.get("usage", {}).get("prompt_tokens"),
           "prompt_tok_s": round(tm.get("prompt_per_second", 0), 1),
           "gen_tok_s": round(tm.get("predicted_per_second", 0), 1),
           "wall_s": round(time.time() - t, 1)}
    log("bench", res)
    return res


# ---------- 4. tunnel ----------
def start_tunnel():
    cf = f"{WORK}/cloudflared"
    if not os.path.exists(cf):
        urllib.request.urlretrieve(
            "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64", cf)
        os.chmod(cf, 0o755)
    p = subprocess.Popen([cf, "tunnel", "--no-autoupdate", "--url", f"http://127.0.0.1:{PORT}"],
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    url = None
    for line in p.stdout:
        m = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", line)
        if m:
            url = m.group(0)
            break
    threading.Thread(target=lambda: [None for _ in p.stdout], daemon=True).start()
    log("tunnel up (url sent encrypted)")
    return p, url


def publish(info):
    try:
        from nacl.public import PublicKey, SealedBox
    except ImportError:
        pip("pynacl")
        from nacl.public import PublicKey, SealedBox
    box = SealedBox(PublicKey(base64.b64decode(RUNNER_PUBKEY)))
    blob = base64.b64encode(box.encrypt(json.dumps(info).encode())).decode()
    req = urllib.request.Request(f"https://ntfy.sh/{NTFY_TOPIC}", data=blob.encode(),
                                 headers={"Title": "ple-gemma", "Cache": "yes"})
    urllib.request.urlopen(req, timeout=20).read()
    log("published endpoint (encrypted), ctx", info["ctx"])


def counters(key):
    try:
        req = urllib.request.Request(f"http://127.0.0.1:{PORT}/metrics", headers={"Authorization": f"Bearer {key}"})
        txt = urllib.request.urlopen(req, timeout=10).read().decode()
        tot = sum(float(v) for v in re.findall(r"^llamacpp:(?:prompt_tokens_total|tokens_predicted_total) (\S+)", txt, re.M))
        busy = sum(float(v) for v in re.findall(r"^llamacpp:requests_processing (\S+)", txt, re.M))
        return tot, busy
    except Exception:
        return None, 0


def main():
    key = "sk-" + secrets.token_urlsafe(32)
    exe = get_llama()
    model = get_model()
    srv, ctx = start_server(exe, model, key)
    speed = bench(key)
    tun, url = start_tunnel()
    if not url:
        raise SystemExit("tunnel gave no URL")
    info = {"url": url, "key": key, "ctx": ctx, "model": "gemma-4-26b", "speed": speed,
            "started": int(T0), "stops_by": int(T0 + MAX_HOURS * 3600), "idle_min": IDLE_MIN}
    publish(info)
    last_tot, last_act, last_pub = None, time.time(), time.time()
    while True:
        time.sleep(60)
        if srv.poll() is not None:
            log("llama-server exited", srv.returncode)
            break
        if tun.poll() is not None:
            log("tunnel dropped, reopening")
            tun, url = start_tunnel()
            info["url"] = url
            publish(info)
            last_pub = time.time()
        tot, busy = counters(key)
        if busy or (tot is not None and tot != last_tot):
            last_act = time.time()
        last_tot = tot
        if time.time() - last_pub > 3 * 3600:  # keep the newest message fresh in ntfy's 12 h cache
            publish(info)
            last_pub = time.time()
        idle = (time.time() - last_act) / 60
        if idle > IDLE_MIN:
            log(f"idle {idle:.0f} min, stopping to save GPU quota")
            break
        if time.time() - T0 > MAX_HOURS * 3600:
            log("max hours reached, stopping")
            break
    info["url"], info["key"] = None, None
    info["stopped"] = int(time.time())
    publish(info)
    for p in (tun, srv):
        p.terminate()


if __name__ == "__main__":
    main()
