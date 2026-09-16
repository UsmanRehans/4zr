"""DoseFuse HTTP API (FastAPI) - runs as a Vercel Python function.

Access: open, no sign-in. Every page open is still recorded (IP geolocation from Vercel headers) for
the visit log. The optional /api/login endpoint lets a visitor attach a name to their cookie.
"""
import hashlib
import hmac
import json
import os
import sys
import threading
import time
import urllib.parse
import urllib.request

_T0 = time.time()
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")  # writable font-cache dir on serverless
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import Depends, FastAPI, HTTPException, Request, Response  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from dosefuse import __version__, session  # noqa: E402

IMPORT_SECONDS = time.time() - _T0
COOKIE = "dosefuse_session"
SESSION_SECONDS = 7 * 24 * 3600


def _secret() -> str:
    return os.environ.get("DOSEFUSE_SECRET") or hashlib.sha256(b"dosefuse:dev").hexdigest()


def _sign(msg: str, secret: str) -> str:
    return hmac.new(secret.encode(), msg.encode(), hashlib.sha256).hexdigest()


def make_token(user: str, secret: str) -> str:
    msg = f"{user}:{int(time.time()) + SESSION_SECONDS}"
    return f"{msg}:{_sign(msg, secret)}"


def check_token(token: str | None) -> str | None:
    if not token:
        return None
    try:
        name, exp, sig = token.rsplit(":", 2)
    except ValueError:
        return None
    if not hmac.compare_digest(sig, _sign(f"{name}:{exp}", _secret())):
        return None
    if int(exp) < time.time() or not name:
        return None
    return name


def require_auth(request: Request) -> str:
    """Access is open; this only resolves an optional visitor name from the cookie."""
    return check_token(request.cookies.get(COOKIE)) or ""


app = FastAPI(title="DoseFuse API", version=__version__)

# ----------------------------------------------------------------------------- visit / login audit
_EVENTS: list[dict] = []          # per-instance fallback; Supabase (if configured) is the durable store
_EVENTS_LOCK = threading.Lock()


def _geo(request: Request) -> dict:
    h = request.headers
    ip = h.get("x-forwarded-for", request.client.host if request.client else "?").split(",")[0].strip()
    return {
        "ip": ip,
        "country": h.get("x-vercel-ip-country", ""),
        "region": h.get("x-vercel-ip-country-region", ""),
        "city": urllib.parse.unquote(h.get("x-vercel-ip-city", "")),
        "lat": h.get("x-vercel-ip-latitude", ""),
        "lon": h.get("x-vercel-ip-longitude", ""),
        "timezone": h.get("x-vercel-ip-timezone", ""),
        "user_agent": h.get("user-agent", "")[:200],
    }


def _record(kind: str, request: Request, user: str = "", ok: bool | None = None):
    """Log a visit/login event to stdout (Vercel runtime logs) and, if configured, to Supabase."""
    ev = {"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "kind": kind, "user": user,
          "ok": ok, **_geo(request)}
    print("AUDIT " + json.dumps(ev), flush=True)
    with _EVENTS_LOCK:
        _EVENTS.append(ev)
        del _EVENTS[:-500]
    url, key = os.environ.get("SUPABASE_URL", ""), os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
    if url and key:
        try:
            req = urllib.request.Request(f"{url.rstrip('/')}/rest/v1/visits", data=json.dumps(ev).encode(),
                                         headers={"apikey": key, "Authorization": f"Bearer {key}",
                                                  "Content-Type": "application/json", "Prefer": "return=minimal"},
                                         method="POST")
            urllib.request.urlopen(req, timeout=3).read()
        except Exception as e:  # noqa: BLE001
            print(f"AUDIT supabase insert failed: {e}", flush=True)


def _fetch_events(limit: int) -> tuple[list[dict], str]:
    url, key = os.environ.get("SUPABASE_URL", ""), os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
    if url and key:
        req = urllib.request.Request(f"{url.rstrip('/')}/rest/v1/visits?select=*&order=ts.desc&limit={int(limit)}",
                                     headers={"apikey": key, "Authorization": f"Bearer {key}"})
        return json.loads(urllib.request.urlopen(req, timeout=5).read()), "supabase"
    with _EVENTS_LOCK:
        return list(reversed(_EVENTS))[:limit], "memory (this function instance only)"


class Login(BaseModel):
    username: str
    password: str | None = None  # no longer required


@app.get("/api/health")
def health():
    return {"ok": True, "version": __version__, "login_mode": "open",
            "import_seconds": round(IMPORT_SECONDS, 2), "cpu_count": os.cpu_count(),
            "cache_files": sorted(os.listdir(session.CACHE_DIR)) if os.path.isdir(session.CACHE_DIR) else [],
            "sessions_in_memory": len(session._SESSIONS), "timings": session.TIMINGS[-20:]}


@app.post("/api/login")
def login(body: Login, request: Request, response: Response):
    name = " ".join(body.username.split())[:40].replace(":", " ")
    ok = len(name) >= 2
    _record("login", request, name, ok)
    if not ok:
        raise HTTPException(status_code=400, detail="Please enter your name")
    secure = request.headers.get("x-forwarded-proto", request.url.scheme) == "https"
    response.set_cookie(COOKIE, make_token(name, _secret()), max_age=SESSION_SECONDS, httponly=True,
                        secure=secure, samesite="lax", path="/")
    return {"ok": True, "user": name}


@app.post("/api/logout")
def logout(response: Response):
    response.delete_cookie(COOKIE, path="/")
    return {"ok": True}


@app.get("/api/me")
def me(request: Request):
    """Called on every page load: records the visit with geolocation; returns the optional visitor name."""
    name = check_token(request.cookies.get(COOKIE)) or ""
    _record("visit", request, name, True)
    return {"user": name}


@app.get("/api/visits")
def visits(limit: int = 100, _: str = Depends(require_auth)):
    """Recent page visits and sign-in attempts with IP geolocation (Vercel headers)."""
    events, source = _fetch_events(max(1, min(limit, 500)))
    return {"source": source, "events": events}


@app.get("/api/run")
def run(method: str = "bspline", grid: float = 40.0, iters: int = 20, _: str = Depends(require_auth)):
    t = time.time()
    out = session.get_session(method, grid, iters).summary()
    session.TIMINGS.append(f"run {method} {grid} {iters}: {time.time()-t:.2f}s")
    return out


@app.get("/api/slice")
def slice_png(kind: str = "fusion_dir", axis: str = "axial", idx: int | None = None, mode: str = "blend",
              alpha: float = 0.5, series: str | None = None, iso: str = "10,20,30,45,60,80,100",
              structs: int = 1, mov_structs: int = 0, dmax: float | None = None,
              method: str = "bspline", grid: float = 40.0, iters: int = 20, _: str = Depends(require_auth)):
    s = session.get_session(method, grid, iters)
    axis = axis if axis in ("axial", "coronal", "sagittal") else "axial"
    levels = []
    for tok in iso.replace(";", ",").split(","):
        try:
            levels.append(float(tok))
        except ValueError:
            pass
    t = time.time()
    png = s.render(kind, axis, idx, mode, alpha, series, levels, bool(structs), bool(mov_structs), dmax)
    dt = time.time() - t
    session.TIMINGS.append(f"render {kind}: {dt:.2f}s")
    return Response(png, media_type="image/png",
                    headers={"Cache-Control": "private, max-age=3600", "X-Render-Seconds": f"{dt:.2f}"})


@app.get("/api/export")
def export(which: str = "eqd2", method: str = "bspline", grid: float = 40.0, iters: int = 20,
           _: str = Depends(require_auth)):
    which = which if which in ("eqd2", "physical", "deformed") else "eqd2"
    data = session.get_session(method, grid, iters).export_rtdose(which)
    name = {"eqd2": "RTDOSE_sum_EQD2.dcm", "physical": "RTDOSE_sum_physical.dcm",
            "deformed": "RTDOSE_A_deformed_to_B.dcm"}[which]
    return Response(data, media_type="application/dicom",
                    headers={"Content-Disposition": f'attachment; filename="{name}"'})


# Local development only: serve the static viewer from the same process (Vercel serves /public itself).
_public = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "public")
if os.path.isdir(_public) and not os.environ.get("VERCEL"):
    app.mount("/", StaticFiles(directory=_public, html=True), name="static")
