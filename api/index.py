"""DoseFuse HTTP API (FastAPI) - runs as a Vercel Python function.

Auth: a single demo account from env vars DOSEFUSE_USER / DOSEFUSE_PASSWORD; sessions are
HMAC-signed HttpOnly cookies (secret DOSEFUSE_SECRET, or derived from the password).
"""
import hashlib
import hmac
import os
import sys
import time

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


def _creds():
    user = os.environ.get("DOSEFUSE_USER", "")
    pw = os.environ.get("DOSEFUSE_PASSWORD", "")
    secret = os.environ.get("DOSEFUSE_SECRET") or hashlib.sha256(("dosefuse:" + pw).encode()).hexdigest()
    return user, pw, secret


def _sign(msg: str, secret: str) -> str:
    return hmac.new(secret.encode(), msg.encode(), hashlib.sha256).hexdigest()


def make_token(user: str, secret: str) -> str:
    msg = f"{user}:{int(time.time()) + SESSION_SECONDS}"
    return f"{msg}:{_sign(msg, secret)}"


def check_token(token: str | None) -> str | None:
    user, pw, secret = _creds()
    if not token or not user:
        return None
    try:
        name, exp, sig = token.rsplit(":", 2)
    except ValueError:
        return None
    if not hmac.compare_digest(sig, _sign(f"{name}:{exp}", secret)):
        return None
    if int(exp) < time.time() or name != user:
        return None
    return name


def require_auth(request: Request) -> str:
    name = check_token(request.cookies.get(COOKIE))
    if not name:
        raise HTTPException(status_code=401, detail="Not signed in")
    return name


app = FastAPI(title="DoseFuse API", version=__version__)


class Login(BaseModel):
    username: str
    password: str


@app.get("/api/health")
def health():
    user, pw, _ = _creds()
    return {"ok": True, "version": __version__, "login_configured": bool(user and pw),
            "import_seconds": round(IMPORT_SECONDS, 2), "cpu_count": os.cpu_count(),
            "cache_files": sorted(os.listdir(session.CACHE_DIR)) if os.path.isdir(session.CACHE_DIR) else [],
            "sessions_in_memory": len(session._SESSIONS), "timings": session.TIMINGS[-20:]}


@app.post("/api/login")
def login(body: Login, request: Request, response: Response):
    user, pw, secret = _creds()
    if not (user and pw):
        raise HTTPException(status_code=503, detail="Login is not configured on the server")
    ok = hmac.compare_digest(body.username.strip(), user) & hmac.compare_digest(body.password, pw)
    ip = request.headers.get("x-forwarded-for", request.client.host if request.client else "?").split(",")[0].strip()
    print(f"AUDIT login {'OK' if ok else 'FAIL'} user={body.username.strip()!r} ip={ip} "
          f"ua={request.headers.get('user-agent', '?')[:80]!r}", flush=True)
    if not ok:
        time.sleep(0.8)
        raise HTTPException(status_code=401, detail="Wrong username or password")
    secure = request.headers.get("x-forwarded-proto", request.url.scheme) == "https"
    response.set_cookie(COOKIE, make_token(user, secret), max_age=SESSION_SECONDS, httponly=True,
                        secure=secure, samesite="lax", path="/")
    return {"ok": True, "user": user}


@app.post("/api/logout")
def logout(response: Response):
    response.delete_cookie(COOKIE, path="/")
    return {"ok": True}


@app.get("/api/me")
def me(user: str = Depends(require_auth)):
    return {"user": user}


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
