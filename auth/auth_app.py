"""Google sign-in gate for the CCTV stack (Caddy forward_auth pattern).

Caddy asks GET /auth/verify for every request; 2xx lets it through,
anything else is returned to the browser (a redirect to /login).
Google (OIDC) authenticates the user; an allow-list in config.local.yaml
decides who gets in; a signed cookie keeps the session.

By default LAN/localhost access (Host is an IP or "localhost") bypasses
login — Google forbids plain-http redirect URIs on private IPs, and the
LAN was always open; the login gate protects the public hostname
(Cloudflare tunnel). Set auth.skip_lan: false to enforce everywhere.
"""

import ipaddress
import json
import threading
import logging
import os
import secrets
import time
import urllib.parse
from datetime import datetime
import urllib.request
from pathlib import Path

import uvicorn
import yaml
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from itsdangerous import BadSignature, URLSafeTimedSerializer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
log = logging.getLogger("auth")

GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_USERINFO_URL = "https://openidconnect.googleapis.com/v1/userinfo"

CONFIG_PATH = os.environ.get("AUTH_CONFIG", "/config/config.local.yaml")
STATE_DIR = Path(os.environ.get("AUTH_STATE", "/state"))
INVITE_DAYS = int(os.environ.get("INVITE_DAYS", "30"))
COOKIE = "ccvt_session"


def load_cfg() -> dict:
    try:
        cfg = yaml.safe_load(Path(CONFIG_PATH).read_text()) or {}
    except OSError:
        cfg = {}
    return cfg.get("auth") or {}


CFG = load_cfg()
ENABLED = bool(CFG.get("enabled"))
MAX_AGE = int(CFG.get("session_max_age_sec", 604800))
SKIP_LAN = bool(CFG.get("skip_lan", True))


def _session_secret() -> str:
    path = STATE_DIR / "session_secret.key"
    try:
        existing = path.read_text().strip()
        if existing:
            return existing
    except OSError:
        pass
    secret = secrets.token_urlsafe(48)
    try:
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(secret)
        path.chmod(0o600)
    except OSError:
        pass
    return secret


signer = URLSafeTimedSerializer(_session_secret())
app = FastAPI(title="ccvt auth")


# ---- helpers ---------------------------------------------------------

def _session(request: Request) -> dict | None:
    raw = request.cookies.get(COOKIE)
    if not raw:
        return None
    try:
        return signer.loads(raw, max_age=MAX_AGE)
    except BadSignature:
        return None


# Access model (2026-07-11): open sign-in. ANYONE who authenticates with
# Google is admitted — the only question is their ROLE:
#   admin   — the owner(s) in config.local.yaml admin_emails
#   member  — anyone who opened an invite link the admin generated
#   user    — everyone else (the default)
# A blocked list still hard-denies, so a bad actor can be cut off.
MEMBERS_PATH = STATE_DIR / "members.json"
_members_lock = threading.Lock()


def _load_members() -> set:
    try:
        return {e.strip().lower()
                for e in json.loads(MEMBERS_PATH.read_text()) if e.strip()}
    except (OSError, ValueError):
        return set()


def _add_member(email: str) -> None:
    email = (email or "").strip().lower()
    if not email:
        return
    with _members_lock:
        m = _load_members()
        if email not in m:
            m.add(email)
            MEMBERS_PATH.write_text(json.dumps(sorted(m), indent=2))


def _remove_member(email: str) -> None:
    email = (email or "").strip().lower()
    with _members_lock:
        m = _load_members()
        if email in m:
            m.discard(email)
            MEMBERS_PATH.write_text(json.dumps(sorted(m), indent=2))


BLOCKED_PATH = STATE_DIR / "blocked.json"


def _load_blocked() -> set:
    try:
        return {e.strip().lower()
                for e in json.loads(BLOCKED_PATH.read_text()) if e.strip()}
    except (OSError, ValueError):
        return set()


def _block_email(email: str) -> None:
    email = (email or "").strip().lower()
    if not email:
        return
    b = _load_blocked(); b.add(email)
    BLOCKED_PATH.write_text(json.dumps(sorted(b), indent=2))


def _is_blocked(email: str) -> bool:
    email = (email or "").strip().lower()
    cfg_blocked = {e.strip().lower() for e in (CFG.get("blocked_emails") or []) if e.strip()}
    return bool(email) and (email in cfg_blocked or email in _load_blocked())


def _is_admin_email(email: str) -> bool:
    email = (email or "").strip().lower()
    admins = {e.strip().lower() for e in (CFG.get("admin_emails") or []) if e.strip()}
    return bool(email) and email in admins


ROSTER_PATH = STATE_DIR / "roster.json"
_roster_lock = threading.Lock()


def _load_roster() -> dict:
    try:
        return json.loads(ROSTER_PATH.read_text())
    except (OSError, ValueError):
        return {}


def _record_login(email, name, picture) -> None:
    email = (email or "").strip().lower()
    if not email:
        return
    with _roster_lock:
        r = _load_roster()
        e = r.get(email) or {"first_seen": _now_ms(), "seen_by_admin": False}
        e.update({"name": name or e.get("name") or email.split("@")[0],
                  "picture": picture or e.get("picture") or "",
                  "last_seen": _now_ms()})
        e.setdefault("first_seen", _now_ms())
        e.setdefault("seen_by_admin", False)
        if _is_admin_email(email):        # admins are never "new to review"
            e["seen_by_admin"] = True
        r[email] = e
        ROSTER_PATH.write_text(json.dumps(r, indent=2))


def _new_count() -> int:
    return sum(1 for em, e in _load_roster().items()
               if not e.get("seen_by_admin") and not _is_admin_email(em))


def _mark_roster_seen() -> None:
    with _roster_lock:
        r = _load_roster()
        for e in r.values():
            e["seen_by_admin"] = True
        ROSTER_PATH.write_text(json.dumps(r, indent=2))


def _now_ms() -> int:
    import time as _t
    return int(_t.time() * 1000)


def _role(email: str) -> str:
    email = (email or "").strip().lower()
    if _is_admin_email(email):
        return "admin"
    if email in _load_members():
        return "member"
    return "user"


# ---- access requests (unknown Google users ask; admin decides) --------

REQ_PATH = STATE_DIR / "access_requests.json"


def _load_requests() -> list[dict]:
    try:
        data = json.loads(REQ_PATH.read_text())
        return data if isinstance(data, list) else []
    except (OSError, ValueError):
        return []


def _save_requests(items: list[dict]) -> None:
    REQ_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = REQ_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(items, indent=2))
    tmp.replace(REQ_PATH)


def _request_status(email: str) -> str | None:
    email = (email or "").strip().lower()
    for it in _load_requests():
        if it.get("email") == email:
            return it.get("status")
    return None


def _add_request(email: str, name: str, picture: str) -> None:
    email = (email or "").strip().lower()
    items = _load_requests()
    if any(it.get("email") == email for it in items):
        return
    items.append({
        "email": email, "name": name, "picture": picture,
        "status": "pending", "requested_at": time.time(),
    })
    _save_requests(items)


def _set_request_status(email: str, status: str) -> None:
    email = (email or "").strip().lower()
    items = _load_requests()
    for it in items:
        if it.get("email") == email:
            it["status"] = status
            break
    _save_requests(items)


def _delete_request(email: str) -> None:
    email = (email or "").strip().lower()
    _save_requests([it for it in _load_requests() if it.get("email") != email])


def _client_ip(request: Request) -> str:
    """The address the request really came from.

    `cloudflared` runs on this host and connects to Caddy over loopback, so the
    peer address of an internet visitor is 127.0.0.1. Judging "is this the LAN"
    from the peer would hand the whole system to the internet. Cloudflare puts the
    true client address in CF-Connecting-IP and that header cannot survive the
    tunnel from a client — Cloudflare overwrites it — so it is the only address
    here that is worth anything when the tunnel is in front.

    Failing that, Caddy APPENDS the peer it saw to X-Forwarded-For, so the LAST
    entry is whoever actually connected to it; earlier entries are client-supplied
    and worthless. `request.client.host` is Caddy's own container address.
    """
    cf = request.headers.get("cf-connecting-ip", "").strip()
    if cf:
        return cf
    xff = request.headers.get("x-forwarded-for", "")
    if xff:
        return xff.split(",")[-1].strip()
    return (request.client.host if request.client else "")


def _is_lan_client(request: Request) -> bool:
    """Whether the CLIENT is on a private network.

    This used to read the Host header. The client chooses that header, so
    `curl -H 'Host: 10.0.0.1'` walked straight past authentication and saw every
    camera. Verified: Host `10.9.9.9` returned 200 where `evil.example.com`
    returned a login redirect. The client's address cannot be forged past Caddy.
    """
    ip = (_client_ip(request) or "").split("%")[0].strip("[]")
    if not ip:
        return False
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return False
    return addr.is_private or addr.is_loopback


def _public_base(request: Request) -> str:
    # Caddy (trusted_proxies) passes the visitor scheme; direct LAN = http.
    proto = request.headers.get("x-forwarded-proto", "http")
    host = request.headers.get("x-forwarded-host") or request.headers.get("host", "")
    return f"{proto}://{host}"


def _safe_next(value: str) -> str:
    if value and value.startswith("/") and not value.startswith("//"):
        return value
    return "/"


def _http_post_form(url: str, data: dict) -> dict:
    req = urllib.request.Request(
        url, data=urllib.parse.urlencode(data).encode(),
        headers={"Content-Type": "application/x-www-form-urlencoded", "Accept": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read().decode())


def _http_get_json(url: str, token: str) -> dict:
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read().decode())


# ---- pages (white industrial theme, matches the webapp) ---------------

def _center(msg: str) -> str:
    return f'<div class="mark"></div><h1>Notice</h1><div class="sub">{msg}</div>'


def _page(title: str, inner: str) -> str:
    return f"""<!DOCTYPE html><html lang="en"><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>{title} · CCTV Monitoring</title>
<link rel="icon" type="image/png" href="data:image/png;base64,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">
<style>
*{{margin:0;padding:0;box-sizing:border-box;}}
body{{min-height:100vh;display:flex;align-items:center;justify-content:center;padding:20px;
  background:#f4f5f7;color:#16181b;
  font-family:-apple-system,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif;font-size:14px;}}
.card{{width:100%;max-width:380px;background:#fff;border:1px solid #d3d7dc;padding:36px 32px 26px;}}
.mark{{width:34px;height:34px;margin:0 auto 14px;background:#16181b;
  -webkit-mask:url("data:image/png;base64,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") center/contain no-repeat;
  mask:url("data:image/png;base64,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") center/contain no-repeat;}}
h1{{font-size:15px;font-weight:700;text-transform:uppercase;letter-spacing:.08em;text-align:center;}}
.sub{{color:#676e76;font-size:11px;text-transform:uppercase;letter-spacing:.06em;
  text-align:center;margin:4px 0 26px;}}
.err{{border:1px solid #d92b2b;background:#fdf0f0;color:#a02020;font-size:12.5px;
  line-height:1.45;padding:10px 12px;margin-bottom:18px;}}
.em{{font-weight:700;}}
.gbtn{{display:flex;align-items:center;justify-content:center;gap:12px;width:100%;height:44px;
  background:#fff;color:#16181b;font-size:13px;font-weight:600;text-decoration:none;
  border:1px solid #9aa1a9;cursor:pointer;}}
.gbtn:hover{{border-color:#16181b;background:#f4f5f7;}}
.gbtn svg{{width:18px;height:18px;}}
.ghost{{display:block;text-align:center;margin-top:12px;padding:10px;color:#676e76;
  font-size:12px;text-decoration:none;border:1px solid #d3d7dc;}}
.ghost:hover{{color:#16181b;border-color:#9aa1a9;}}
.foot{{display:flex;align-items:center;gap:10px;color:#9aa1a9;font-size:10px;font-weight:600;
  letter-spacing:.08em;margin:24px 0 10px;text-transform:uppercase;}}
.foot::before,.foot::after{{content:'';flex:1;height:1px;background:#d3d7dc;}}
.note{{color:#676e76;font-size:11px;line-height:1.6;text-align:center;}}
</style></head><body><div class="card">{inner}</div></body></html>"""


def _google_svg() -> str:
    return (
        '<svg viewBox="0 0 18 18" aria-hidden="true">'
        '<path fill="#4285F4" d="M17.64 9.2c0-.637-.057-1.251-.164-1.84H9v3.481h4.844c-.209 1.125-.843 2.078-1.796 2.717v2.258h2.908c1.702-1.567 2.684-3.874 2.684-6.615z"/>'
        '<path fill="#34A853" d="M9 18c2.43 0 4.467-.806 5.956-2.18l-2.908-2.259c-.806.54-1.837.86-3.048.86-2.344 0-4.328-1.584-5.036-3.711H.957v2.332C2.438 15.983 5.482 18 9 18z"/>'
        '<path fill="#FBBC05" d="M3.964 10.71c-.18-.54-.282-1.117-.282-1.71s.102-1.17.282-1.71V4.958H.957C.347 6.173 0 7.548 0 9s.348 2.827.957 4.042l3.007-2.332z"/>'
        '<path fill="#EA4335" d="M9 3.58c1.321 0 2.508.454 3.44 1.345l2.582-2.58C13.463.891 11.426 0 9 0 5.482 0 2.438 2.017.957 4.958L3.964 7.29C4.672 5.163 6.656 3.58 9 3.58z"/></svg>'
    )


def _login_page(error: str = "", next_url: str = "/") -> str:
    err = f'<div class="err">{error}</div>' if error else ""
    nxt = urllib.parse.quote(_safe_next(next_url))
    inner = (
        '<div class="mark"></div><h1>CCTV Monitoring</h1>'
        '<div class="sub">Restricted surveillance system</div>'
        f'{err}'
        f'<a class="gbtn" href="/auth/google/login?next={nxt}">{_google_svg()}'
        '<span>Sign in with Google</span></a>'
        '<div class="foot">Secure access</div>'
        '<div class="note"><span class="em">Authorized accounts only.</span><br>'
        'All access is logged.</div>'
    )
    return _page("Sign in", inner)


def _denied_page(email: str, status: str) -> str:
    word = "blocked" if status == "blocked" else "denied"
    inner = (
        '<div class="mark"></div><h1>Access denied</h1>'
        f'<div class="sub">Account {word} by the administrator</div>'
        f'<div class="err"><span class="em">{email}</span> has been {word}. '
        'Contact the administrator if you believe this is a mistake.</div>'
        '<a class="ghost" href="/logout">Use a different account</a>'
    )
    return _page("Denied", inner)


def _request_page(email: str, status: str | None) -> str:
    if status == "pending":
        inner = (
            '<div class="mark"></div><h1>Request pending</h1>'
            '<div class="sub">Waiting for administrator approval</div>'
            f'<div class="note" style="margin-bottom:18px;"><span class="em">{email}</span><br>'
            'Your request has been sent. You can sign in once it is approved.</div>'
            '<a class="ghost" href="/logout">Use a different account</a>'
        )
        return _page("Pending", inner)
    inner = (
        '<div class="mark"></div><h1>Access needed</h1>'
        '<div class="sub">Not on the authorized list yet</div>'
        f'<div class="note" style="margin-bottom:18px;"><span class="em">{email}</span><br>'
        'Request access and the administrator will review it.</div>'
        '<form method="post" action="/auth/request-access">'
        '<button class="gbtn" type="submit">Request access</button></form>'
        '<a class="ghost" href="/logout">Use a different account</a>'
    )
    return _page("Request access", inner)


def _request_sent_page(email: str) -> str:
    inner = (
        '<div class="mark"></div><h1>Request sent</h1>'
        '<div class="sub">The administrator will review it</div>'
        f'<div class="note" style="margin-bottom:18px;">Your request for '
        f'<span class="em">{email}</span> was saved.<br>'
        'You will get in once it is approved.</div>'
        '<a class="ghost" href="/logout">Done</a>'
    )
    return _page("Sent", inner)


def _people_page(people: list[dict]) -> str:
    """Roster of everyone who has signed in, newest first, with their role and
    per-person actions. Access is open now; this lists, it does not gate."""
    if not people:
        rows = ('<tr><td colspan="4" style="text-align:center;color:var(--text-muted);'
                'padding:26px;">No one has signed in yet.</td></tr>')
    else:
        parts = []
        for p in people:
            email = p.get("email", "")
            name = p.get("name", "") or email.split("@")[0]
            role = p.get("role", "user")
            when = ""
            try:
                when = datetime.fromtimestamp(p.get("first_seen", 0) / 1000).strftime("%Y-%m-%d %H:%M")
            except (ValueError, OSError, OverflowError):
                pass
            q = urllib.parse.quote(email)
            acts = []
            if role == "user":
                acts.append(f'<form method="post" action="/access-requests/action?email={q}&do=promote">'
                            '<button class="mini mini-ok" type="submit">Make member</button></form>')
            elif role == "member":
                acts.append(f'<form method="post" action="/access-requests/action?email={q}&do=demote">'
                            '<button class="mini" type="submit">Remove member</button></form>')
            if role != "admin":
                acts.append(f'<form method="post" action="/access-requests/action?email={q}&do=block">'
                            '<button class="mini mini-no" type="submit">Block</button></form>')
            parts.append(
                f'<tr><td><span class="u-name">{name}</span><br>'
                f'<span class="u-mail">{email}</span></td>'
                f'<td><span class="pill pill-{role}">{role}</span></td>'
                f'<td class="u-when">{when}</td>'
                f'<td><div class="act">{"".join(acts)}</div></td></tr>')
        rows = "".join(parts)
    tpl = """<!DOCTYPE html><html lang="en"><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>People</title>
<link rel="stylesheet" href="/css/app.css">
<style>
.wrap{flex:1;overflow-y:auto;padding:14px 20px;}
.panel{border:1px solid var(--border);border-radius:12px;background:var(--bg);padding:14px 16px;max-width:900px;}
.panel__title{font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.07em;margin-bottom:10px;}
table{width:100%;border-collapse:collapse;font-size:12px;}
th,td{text-align:left;padding:9px 10px;border-bottom:1px solid var(--border);vertical-align:middle;}
th{color:var(--text-muted);font-weight:600;font-size:10px;text-transform:uppercase;letter-spacing:.06em;}
.u-name{font-weight:600;}
.u-mail{color:var(--text-muted);font-family:var(--mono);font-size:11px;}
.u-when{color:var(--text-muted);font-family:var(--mono);font-size:11px;white-space:nowrap;}
.pill{font-size:10px;font-weight:700;padding:2px 9px;border-radius:10px;text-transform:uppercase;letter-spacing:.04em;}
.pill-admin{background:#eef4ff;color:#0b5fd9;}
.pill-member{background:#eefaf2;color:#177a42;}
.pill-user{background:var(--surface);color:var(--text-muted);}
.act{display:flex;gap:6px;flex-wrap:wrap;}
.act form{display:inline;}
.mini{font-size:11px;font-weight:600;padding:5px 10px;border:1px solid var(--border);border-radius:7px;
background:var(--bg);cursor:pointer;color:var(--text);}
.mini:hover{border-color:var(--text);}
.mini-ok{color:#177a42;}
.mini-no{color:#a02020;}
.invite{margin-bottom:4px;display:flex;gap:8px;align-items:center;flex-wrap:wrap;}
.invite input{flex:1;min-width:240px;font:12px var(--mono);padding:7px 9px;border:1px solid var(--border);border-radius:7px;}
.invite button{font:600 11px sans-serif;text-transform:uppercase;letter-spacing:.05em;padding:7px 14px;
border:1px solid var(--text);border-radius:7px;background:var(--text);color:#fff;cursor:pointer;}
</style></head><body>
<header class="topbar">
  <div class="topbar__brand"><span class="topbar__mark"></span>
    <div><h1 class="topbar__title">People</h1>
      <span class="topbar__sub">Everyone who has signed in &middot; admin</span></div></div>
  <div class="topbar__meta"></div>
</header>
<main class="wrap">
  <section class="panel">
    <div class="panel__title">Invite a member</div>
    <div class="invite">
      <input id="invite-url" readonly placeholder="Click Generate to create an invite link">
      <button onclick="fetch('/invite/new').then(r=>r.json()).then(d=>{document.getElementById('invite-url').value=d.invite_url;})">Generate link</button>
      <button onclick="navigator.clipboard.writeText(document.getElementById('invite-url').value)">Copy</button>
    </div>
    <div class="panel__title" style="margin-top:14px">Everyone who has signed in (__N__)</div>
    <table>
      <thead><tr><th>Person</th><th>Role</th><th>First seen</th><th>Action</th></tr></thead>
      <tbody>__ROWS__</tbody>
    </table>
  </section>
</main>
<script type="module" src="/js/nav.js"></script>
</body></html>"""
    return tpl.replace("__N__", str(len(people))).replace("__ROWS__", rows)


def _admin_page(items: list[dict]) -> str:
    """Full app-layout page — same topbar/statusbar as the main web UI
    (loads the webapp's /css/app.css from the shared origin)."""
    order = {"pending": 0, "approved": 1, "denied": 2, "blocked": 3}
    items = sorted(items, key=lambda x: (order.get(x.get("status"), 9),
                                         -float(x.get("requested_at", 0) or 0)))
    pending = sum(1 for it in items if it.get("status") == "pending")

    if not items:
        rows = ('<tr><td colspan="4" style="text-align:center;color:var(--text-muted);'
                'padding:26px;">No access requests yet.</td></tr>')
    else:
        parts = []
        for it in items:
            email = it.get("email", "")
            name = it.get("name", "") or email.split("@")[0]
            st = it.get("status", "pending")
            when = ""
            try:
                when = datetime.fromtimestamp(float(it.get("requested_at", 0))).strftime("%Y-%m-%d %H:%M")
            except (ValueError, OSError, OverflowError):
                pass
            q = urllib.parse.quote(email)
            acts = []
            if st != "approved":
                acts.append(f'<form method="post" action="/access-requests/action?email={q}&do=approve">'
                            '<button class="mini mini-ok" type="submit">Approve</button></form>')
            if st != "denied":
                acts.append(f'<form method="post" action="/access-requests/action?email={q}&do=deny">'
                            '<button class="mini mini-no" type="submit">Deny</button></form>')
            if st != "blocked":
                acts.append(f'<form method="post" action="/access-requests/action?email={q}&do=block">'
                            '<button class="mini mini-no" type="submit">Block</button></form>')
            acts.append(f'<form method="post" action="/access-requests/action?email={q}&do=delete">'
                        '<button class="mini" type="submit">Delete</button></form>')
            parts.append(
                f'<tr><td><span class="u-name">{name}</span><br>'
                f'<span class="u-mail">{email}</span></td>'
                f'<td><span class="pill pill-{st}">{st}</span></td>'
                f'<td class="u-when">{when}</td>'
                f'<td><div class="act">{"".join(acts)}</div></td></tr>'
            )
        rows = "".join(parts)

    return f"""<!DOCTYPE html><html lang="en"><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>Access Requests</title>
<link rel="stylesheet" href="/css/app.css">
<style>
.wrap{{flex:1;overflow-y:auto;padding:14px 20px;}}
.panel{{border:1px solid var(--border);background:var(--bg);padding:12px 14px 14px;max-width:860px;}}
.panel__title{{font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.07em;margin-bottom:10px;}}
table{{width:100%;border-collapse:collapse;font-size:12px;}}
th,td{{text-align:left;padding:9px 10px;border-bottom:1px solid var(--border);vertical-align:middle;}}
th{{color:var(--text-muted);font-weight:600;font-size:10px;text-transform:uppercase;letter-spacing:.06em;}}
.u-name{{font-weight:600;}}
.u-mail{{color:var(--text-muted);font-family:var(--mono);font-size:11px;}}
.u-when{{color:var(--text-muted);font-family:var(--mono);font-size:11px;white-space:nowrap;}}
.pill{{font-size:10px;font-weight:700;padding:2px 9px;text-transform:uppercase;letter-spacing:.04em;}}
.pill-pending{{background:#fdf6e3;color:#a07d0b;border:1px solid var(--warn);}}
.pill-approved{{background:#eefaf2;color:#177a42;border:1px solid var(--ok);}}
.pill-denied{{background:#fdf0f0;color:#a02020;border:1px solid var(--err);}}
.pill-blocked{{background:#f4d7d7;color:#8a1414;border:1px solid #a02020;}}
.act{{display:flex;gap:6px;flex-wrap:wrap;}}
.act form{{display:inline;}}
.mini{{font-size:11px;font-weight:600;padding:5px 10px;border:1px solid var(--border-strong);
background:var(--bg);cursor:pointer;color:var(--text);}}
.mini:hover{{border-color:var(--text);}}
.mini-ok{{color:#177a42;border-color:var(--ok);}}
.mini-no{{color:#a02020;border-color:var(--err);}}
</style></head><body>

<header class="topbar">
  <div class="topbar__brand">
    <span class="topbar__mark"></span>
    <div>
      <h1 class="topbar__title">Access Requests</h1>
      <span class="topbar__sub">Sign-in requests &middot; admin</span>
    </div>
  </div>
  <div class="topbar__meta"></div>
</header>

<main class="wrap">
  <section class="panel">
    <div class="panel__title">Approve, deny or block sign-in requests</div>
    <table>
      <thead><tr><th>User</th><th>Status</th><th>Requested</th><th>Action</th></tr></thead>
      <tbody>{rows}</tbody>
    </table>
  </section>
</main>

<footer class="statusbar">
  <span>{len(items)} request{"" if len(items) == 1 else "s"} &middot; {pending} pending</span>
  <span class="statusbar__right">approved accounts sign in on their next attempt</span>
</footer>

<script type="module" src="/js/nav.js"></script>
</body></html>"""


# ---- routes -----------------------------------------------------------

@app.get("/auth/verify")
def verify(request: Request):
    if not ENABLED:
        return Response(status_code=204)
    if SKIP_LAN and _is_lan_client(request):
        return Response(status_code=204)
    if _session(request):
        return Response(status_code=204)
    uri = request.headers.get("x-forwarded-uri", "/")
    return RedirectResponse(f"/login?next={urllib.parse.quote(_safe_next(uri))}", status_code=303)


@app.get("/auth/me")
def me(request: Request):
    if not ENABLED:
        return {"enabled": False, "authenticated": False}
    sess = _session(request)
    if sess:
        role = _role(sess.get("email"))
        return {"enabled": True, "authenticated": True,
                "role": role, "admin": role == "admin", **sess}
    if SKIP_LAN and _is_lan_client(request):
        return {"enabled": True, "authenticated": False, "lan_bypass": True}
    return JSONResponse({"enabled": True, "authenticated": False}, status_code=401)


@app.get("/login")
def login(request: Request):
    if not ENABLED or _session(request):
        return RedirectResponse("/", status_code=303)
    return HTMLResponse(_login_page(next_url=request.query_params.get("next", "/")))


@app.get("/auth/google/login")
def google_login(request: Request):
    state = secrets.token_urlsafe(24)
    nxt = _safe_next(request.query_params.get("next", "/"))
    params = {
        "client_id": CFG.get("google_client_id", ""),
        "redirect_uri": _public_base(request) + "/auth/google/callback",
        "response_type": "code",
        "scope": "openid email profile",
        "state": state,
        "access_type": "online",
        "prompt": "select_account",
    }
    resp = RedirectResponse(GOOGLE_AUTH_URL + "?" + urllib.parse.urlencode(params), status_code=303)
    resp.set_cookie("ccvt_oauth", signer.dumps({"state": state, "next": nxt}),
                    max_age=600, httponly=True, samesite="lax")
    return resp


@app.get("/auth/google/callback")
def google_callback(request: Request):
    if request.query_params.get("error"):
        return HTMLResponse(_login_page(error="Google sign-in was cancelled."), status_code=401)
    try:
        oauth = signer.loads(request.cookies.get("ccvt_oauth", ""), max_age=600)
    except BadSignature:
        oauth = None
    if not oauth or request.query_params.get("state") != oauth.get("state"):
        return HTMLResponse(_login_page(error="Session expired — please try again."), status_code=400)
    code = request.query_params.get("code")
    if not code:
        return HTMLResponse(_login_page(error="No authorization code returned."), status_code=400)
    try:
        tok = _http_post_form(GOOGLE_TOKEN_URL, {
            "code": code,
            "client_id": CFG.get("google_client_id", ""),
            "client_secret": CFG.get("google_client_secret", ""),
            "redirect_uri": _public_base(request) + "/auth/google/callback",
            "grant_type": "authorization_code",
        })
        info = _http_get_json(GOOGLE_USERINFO_URL, tok.get("access_token", ""))
    except Exception:
        log.exception("token exchange failed")
        return HTMLResponse(_login_page(error="Sign-in failed — please try again."), status_code=502)

    email = info.get("email", "")
    if not email or not info.get("email_verified", True):
        return HTMLResponse(_login_page(error="Could not verify your Google account."), status_code=403)
    if _is_blocked(email):
        log.warning("blocked login: %s", email)
        return HTMLResponse(_denied_page(email, "blocked"), status_code=403)

    log.info("login ok: %s (role=%s)", email, _role(email))
    _record_login(email, info.get("name"), info.get("picture"))
    session = {
        "email": email,
        "name": info.get("name") or email.split("@")[0],
        "picture": info.get("picture") or "",
    }
    secure = request.headers.get("x-forwarded-proto", "http") == "https"
    resp = RedirectResponse(_safe_next(oauth.get("next", "/")), status_code=303)
    resp.set_cookie(COOKIE, signer.dumps(session), max_age=MAX_AGE,
                    httponly=True, samesite="lax", secure=secure)
    resp.delete_cookie("ccvt_oauth")
    return resp


@app.get("/auth/new-people")
def new_people(request: Request):
    sess = _session(request)
    if not sess or _role(sess.get("email")) != "admin":
        return JSONResponse({"error": "admin only"}, status_code=403)
    return {"new": _new_count()}


@app.get("/invite/new")
def invite_new(request: Request):
    """Admin-only: mint a member-invite link. Anyone who opens it and signs in
    with Google becomes a member. Reusable; expires in INVITE_DAYS."""
    sess = _session(request)
    if not sess or _role(sess.get("email")) != "admin":
        return JSONResponse({"error": "admin only"}, status_code=403)
    token = signer.dumps({"kind": "member-invite"})
    link = _public_base(request).rstrip("/") + "/invite/" + urllib.parse.quote(token)
    return {"invite_url": link, "expires_days": INVITE_DAYS}


@app.get("/invite/{token}")
def invite_accept(request: Request, token: str):
    """Open an invite link. Must be signed in first (Google); then the caller
    is promoted to member and sent into the app."""
    sess = _session(request)
    if not sess:
        nxt = urllib.parse.quote("/invite/" + token)
        return RedirectResponse(f"/login?next={nxt}", status_code=303)
    try:
        data = signer.loads(token, max_age=INVITE_DAYS * 86400)
        if data.get("kind") != "member-invite":
            raise BadSignature("wrong kind")
    except BadSignature:
        return HTMLResponse(_page("Invite invalid", _center(
            "This invite link is invalid or has expired. Ask the administrator "
            "for a new one.")), status_code=400)
    _add_member(sess.get("email"))
    log.info("invite accepted: %s -> member", sess.get("email"))
    return RedirectResponse("/", status_code=303)


@app.get("/logout")
def logout():
    resp = RedirectResponse("/login", status_code=303)
    resp.delete_cookie(COOKIE)
    resp.delete_cookie("ccvt_pending")
    return resp


@app.post("/auth/request-access")
def request_access(request: Request):
    try:
        pending = signer.loads(request.cookies.get("ccvt_pending", ""), max_age=600)
    except BadSignature:
        pending = None
    if not pending or not pending.get("email"):
        return RedirectResponse("/login", status_code=303)
    email = pending["email"]
    if _request_status(email) in ("blocked", "denied"):
        return HTMLResponse(_denied_page(email, _request_status(email)), status_code=403)
    _add_request(email, pending.get("name", ""), pending.get("picture", ""))
    log.info("access requested: %s", email)
    return HTMLResponse(_request_sent_page(email))


def _admin_session(request: Request) -> dict | None:
    sess = _session(request)
    if sess and _is_admin_email(sess.get("email")):
        return sess
    return None


@app.get("/access-requests")
def access_requests_page(request: Request):
    if not ENABLED:
        return RedirectResponse("/", status_code=303)
    if not _admin_session(request):
        if _session(request):
            inner = (
                '<div class="mark"></div><h1>Not authorized</h1>'
                '<div class="sub">Administrators only</div>'
                '<a class="ghost" href="/">Back to cameras</a>'
            )
            return HTMLResponse(_page("Forbidden", inner), status_code=403)
        return RedirectResponse("/login?next=/access-requests", status_code=303)
    people = sorted(({"email": k, **v} for k, v in _load_roster().items()),
                    key=lambda p: p.get("first_seen", 0), reverse=True)
    for p in people:
        p["role"] = _role(p["email"])
    html = _people_page(people)
    _mark_roster_seen()               # viewing clears the "new" badge
    return HTMLResponse(html)


@app.post("/access-requests/action")
def access_requests_action(request: Request):
    if not _admin_session(request):
        return JSONResponse({"error": "forbidden"}, status_code=403)
    email = (request.query_params.get("email", "") or "").strip().lower()
    do = request.query_params.get("do", "")
    if email and not _is_admin_email(email):     # never demote/block an admin
        if do == "promote":
            _add_member(email)
        elif do == "demote":
            _remove_member(email)
        elif do == "block":
            _remove_member(email)
            _block_email(email)
        log.info("admin action %s on %s", do, email)
    return RedirectResponse("/access-requests", status_code=303)


if __name__ == "__main__":
    log.info("auth service on :8082 (enabled=%s, skip_lan=%s)", ENABLED, SKIP_LAN)
    uvicorn.run(app, host="0.0.0.0", port=8082, log_level="warning")
