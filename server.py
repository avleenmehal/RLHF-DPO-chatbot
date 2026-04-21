"""FastAPI server — wraps Gradio app with JWT-based authentication."""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import gradio as gr
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from starlette.middleware.base import BaseHTTPMiddleware

from api.auth import AuthManager
from core.config import Config
from db.database import SessionLocal, init_db

# ── HTML templates ────────────────────────────────────────────────────────────

def _page(title: str, body: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>{title} — Medical Chatbot</title>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #f0f4f8;
      display: flex;
      align-items: center;
      justify-content: center;
      min-height: 100vh;
    }}
    .card {{
      background: white;
      border-radius: 12px;
      box-shadow: 0 4px 24px rgba(0,0,0,0.08);
      padding: 2.5rem 2rem;
      width: 100%;
      max-width: 400px;
    }}
    .logo {{
      text-align: center;
      font-size: 2rem;
      margin-bottom: 0.5rem;
    }}
    h1 {{
      text-align: center;
      font-size: 1.4rem;
      color: #1a202c;
      margin-bottom: 1.5rem;
    }}
    label {{
      display: block;
      font-size: 0.85rem;
      color: #4a5568;
      margin-bottom: 0.3rem;
      margin-top: 1rem;
    }}
    input {{
      width: 100%;
      padding: 0.6rem 0.8rem;
      border: 1px solid #cbd5e0;
      border-radius: 6px;
      font-size: 0.95rem;
      outline: none;
      transition: border 0.2s;
    }}
    input:focus {{ border-color: #4299e1; }}
    button {{
      margin-top: 1.5rem;
      width: 100%;
      padding: 0.7rem;
      background: #3182ce;
      color: white;
      border: none;
      border-radius: 6px;
      font-size: 1rem;
      cursor: pointer;
      transition: background 0.2s;
    }}
    button:hover {{ background: #2b6cb0; }}
    .error {{
      background: #fff5f5;
      border: 1px solid #feb2b2;
      color: #c53030;
      border-radius: 6px;
      padding: 0.6rem 0.8rem;
      font-size: 0.875rem;
      margin-top: 1rem;
    }}
    .footer {{
      text-align: center;
      margin-top: 1.2rem;
      font-size: 0.85rem;
      color: #718096;
    }}
    .footer a {{ color: #3182ce; text-decoration: none; }}
    .footer a:hover {{ text-decoration: underline; }}
  </style>
</head>
<body>
  <div class="card">
    <div class="logo">🏥</div>
    <h1>{title}</h1>
    {body}
  </div>
</body>
</html>"""


def _login_html(error: str = "") -> str:
    error_block = f'<div class="error">{error}</div>' if error else ""
    return _page("Sign In", f"""
    <form method="post" action="/login">
      <label>Email</label>
      <input type="email" name="email" required autofocus/>
      <label>Password</label>
      <input type="password" name="password" required/>
      {error_block}
      <button type="submit">Sign In</button>
    </form>
    <div class="footer">No account? <a href="/register">Register</a></div>
    """)


def _register_html(error: str = "") -> str:
    error_block = f'<div class="error">{error}</div>' if error else ""
    return _page("Create Account", f"""
    <form method="post" action="/register">
      <label>Email</label>
      <input type="email" name="email" required autofocus/>
      <label>Username</label>
      <input type="text" name="username" required/>
      <label>Password</label>
      <input type="password" name="password" required/>
      {error_block}
      <button type="submit">Create Account</button>
    </form>
    <div class="footer">Already have an account? <a href="/login">Sign In</a></div>
    """)


# ── App setup ─────────────────────────────────────────────────────────────────

app = FastAPI()
init_db()


# ── Health check — required by Cloud Run ─────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok"}


# ── Middleware: protect /app ──────────────────────────────────────────────────

_GRADIO_PASSTHROUGH = (
    "/app/gradio_api/",
    "/app/queue/",
    "/app/info",
    "/app/theme.css",
    "/app/assets/",
    "/app/favicon.ico",
)

class AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        if path.startswith("/app"):
            if not any(path.startswith(p) for p in _GRADIO_PASSTHROUGH):
                token = request.cookies.get("access_token")
                if not token or not AuthManager.decode_token(token):
                    return RedirectResponse(url="/login")
        return await call_next(request)


app.add_middleware(AuthMiddleware)


# ── Auth routes ───────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root():
    return RedirectResponse(url="/app")


@app.get("/login", response_class=HTMLResponse)
async def login_page():
    return _login_html()


@app.post("/login")
async def login(email: str = Form(...), password: str = Form(...)):
    db = SessionLocal()
    try:
        user = AuthManager.login(db, email, password)
    finally:
        db.close()

    if not user:
        return HTMLResponse(_login_html(error="Invalid email or password."))

    token = AuthManager.create_token(user.user_id)
    response = RedirectResponse(url="/app", status_code=303)
    response.set_cookie(
        key="access_token",
        value=token,
        httponly=True,
        max_age=Config.JWT_EXPIRE_HOURS * 3600,
        samesite="lax",
    )
    return response


@app.get("/register", response_class=HTMLResponse)
async def register_page():
    return _register_html()


@app.post("/register")
async def register(
    email: str = Form(...),
    username: str = Form(...),
    password: str = Form(...),
):
    db = SessionLocal()
    try:
        AuthManager.register(db, email, username, password)
    except ValueError as e:
        db.close()
        return HTMLResponse(_register_html(error=str(e)))
    finally:
        db.close()

    return RedirectResponse(url="/login", status_code=303)


@app.get("/logout")
async def logout():
    response = RedirectResponse(url="/login")
    response.delete_cookie("access_token")
    return response


# ── Mount Gradio ──────────────────────────────────────────────────────────────

from ui.app import demo  # noqa: E402  (import after env var is set)

gr.mount_gradio_app(app, demo, path="/app", root_path="/app")
