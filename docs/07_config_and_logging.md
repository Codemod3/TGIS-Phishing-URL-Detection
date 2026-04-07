# Configuration, Environment Variables & Logging

This file explains how the application gets its settings (database passwords, API keys,
port numbers) and how it records what's happening while it runs.

---

## 1. The Problem with Hardcoded Secrets

Imagine writing your database password directly in your Python code:

```python
engine = create_engine("postgresql://postgres:MyPassword123@localhost/phishing_db")
```

This is dangerous for two reasons:
1. Anyone who reads your code (or your GitHub repo) sees your password.
2. Your production server uses a different database than your laptop — you'd have to
   change the code every time you switch environments.

The solution is **environment variables** — values that live outside the code, in the
operating system or a `.env` file that is never committed to git.

---

## 2. The `.env` File

```
# .env  (never commit this file — it's in .gitignore)
DATABASE_URL=postgresql://postgres:secret@localhost:5432/phishing_detector
REDIS_HOST=localhost
REDIS_PORT=6379
SAFE_BROWSING_API_KEY=AIzaSy...
LOG_LEVEL=INFO
SECRET_KEY=some-random-string
```

This file is read at startup. It's listed in `.gitignore` so it's never accidentally
pushed to GitHub. The `.env.example` file (which IS committed) shows the required variable
names with placeholder values so other developers know what to set.

---

## 3. Pydantic Settings — `src/core/config.py`

```python
# src/core/config.py  (line 6-46)
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    PORT: int = Field(default=8000, env="PORT")
    DB_HOST: str = Field(default="localhost", env="DB_HOST")
    DB_PASSWORD: str = Field(default="password", env="DB_PASSWORD")
    REDIS_HOST: str = Field(default="localhost", env="REDIS_HOST")
    SAFE_BROWSING_API_KEY: str = Field(default="placeholder_key", env="SAFE_BROWSING_API_KEY")
    LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")

    model_config = SettingsConfigDict(
        env_file=".env",         # Look for a .env file
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"           # Ignore unknown variables in .env
    )
```

`BaseSettings` is a special Pydantic class that reads values from environment variables
and `.env` files automatically. The priority order is:

```
System environment variable (highest) → .env file → Field(default=...) (lowest)
```

So if you set `REDIS_HOST=redis-server` in your terminal, it overrides the `.env` file.
This is how Docker and cloud deployments work — they inject environment variables
without needing a `.env` file at all.

**Type safety:**
```python
PORT: int = Field(default=8000, env="PORT")
```
If someone sets `PORT=abc` (not a number), Pydantic raises a clear error immediately
on startup instead of failing mysteriously later when the port is used.

### Properties for Constructed URLs

```python
# src/core/config.py  (line 48-56)
@property
def database_url(self) -> str:
    return f"postgresql://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"

@property
def redis_url(self) -> str:
    return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"
```

A `@property` is a method you call like an attribute (no parentheses).
`settings.database_url` constructs the full connection string from the individual pieces.
This means you configure individual parts (host, port, name) instead of one opaque URL,
which is easier to manage in environments where each part comes from a different source.

### The Global Settings Instance

```python
# src/core/config.py  (line 59)
settings = Settings()
```

This single line creates the settings object and reads all environment variables.
Every other module imports `settings` from here:

```python
from src.core.config import settings
redis.Redis(host=settings.REDIS_HOST, port=settings.REDIS_PORT)
```

The settings are loaded **once** at import time and reused everywhere.

---

## 4. The Logging System — `src/core/logger.py`

Python has a built-in `logging` module, but this project uses **Loguru** — a library
that makes logging dramatically simpler and prettier.

### Why Loguru?

Standard Python logging requires ~20 lines of boilerplate to configure.
Loguru replaces that with a single function call, adds colored terminal output,
supports `log.success()` (which standard logging lacks), and handles file rotation.

### Logger Setup

```python
# src/core/logger.py  (line 27-71)
def setup_logging():
    logger.remove()  # Remove Loguru's default handler (just prints everything)

    log_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "           # Level padded to 8 chars: INFO    
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    )
```

The format string uses `<color>...</color>` tags for terminal coloring, and `{field}` for
log record fields. The output looks like:

```
2026-04-07 14:32:01 | INFO     | api.routes.predict:predict_url:22 | API Request: POST /api/v1/predict
```

### Console Output

```python
# src/core/logger.py  (line 47-52)
logger.add(
    sys.stdout,
    level=settings.LOG_LEVEL,   # Only show INFO and above (or DEBUG in dev)
    format=log_format,
    colorize=True,
)
```

`sys.stdout` sends logs to the terminal. `level=settings.LOG_LEVEL` means you can set
`LOG_LEVEL=DEBUG` in `.env` to see detailed debug messages during development,
or `LOG_LEVEL=WARNING` in production to reduce noise.

### File Output with Rotation

```python
# src/core/logger.py  (line 55-62)
logger.add(
    "logs/app.log",
    level=settings.LOG_LEVEL,
    format=log_format,
    rotation="10 MB",     # Start a new file when the current one hits 10 MB
    retention="10 days",  # Delete log files older than 10 days
    compression="zip",    # Compress old log files to save disk space
)
```

Without rotation, log files grow forever and eventually fill up the disk.
`rotation="10 MB"` creates `app.log`, then `app.2026-04-07_14-32-01.log.zip`, etc.
`retention="10 days"` automatically deletes files older than 10 days.
You can see these compressed archives in `logs/` (e.g. `app.2026-04-03_17-29-05_678473.log.zip`).

### Intercepting Third-Party Logs

```python
# src/core/logger.py  (line 7-25)
class InterceptHandler(logging.Handler):
    """Routes standard logging calls into Loguru."""
    def emit(self, record: logging.LogRecord) -> None:
        level = logger.level(record.levelname).name
        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())

logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)
```

Libraries like FastAPI, SQLAlchemy, and XGBoost use Python's standard `logging` module,
not Loguru. `InterceptHandler` acts as a bridge — it catches their log messages and
re-routes them through Loguru, so everything appears in the same format and the same files.

`depth=depth` tells Loguru to show the correct file/line number from the original calling
code, not from the `InterceptHandler` itself (otherwise all third-party logs would appear
to come from line 23 of `logger.py`).

### How Logs Are Used Throughout the Code

```python
# Anywhere in the codebase:
from src.core.logger import log

log.debug("Detailed trace only shown in dev")   # Not shown at LOG_LEVEL=INFO
log.info("Normal operational message")
log.warning("Something unexpected, but not a crash")
log.error("A serious problem occurred")
log.success("All good — Loguru-exclusive level")  # Shown in green
```

`log` is the configured Loguru instance exported from `logger.py`. Because it's a module-level
variable created once, importing `log` from anywhere gives you the same configured logger —
the same file handler, same format, same level. You never have to configure it again.
