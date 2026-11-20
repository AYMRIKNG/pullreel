"""Git diff analysis and UI signal extraction for PullReel.

Inspects a repository's diff against a base branch (optionally including uncommitted
worktree edits and untracked files), filters out noise (lockfiles, build artifacts,
binaries), infers framework routes, and extracts compact UI signals (components,
user-facing labels, test IDs, event handlers, and a bounded added-lines excerpt)
into a ``DiffSummary`` ready for the storyboard LLM.
"""

from __future__ import annotations

import json
import math
import re
import subprocess
import tomllib
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from pathlib import Path, PurePosixPath

from pullreel.models import ChangedFile, DiffSummary, FeatureCandidate, FileStatus

__all__ = [
    "GitError",
    "analyze_diff",
    "build_excerpt",
    "build_feature_candidates",
    "extract_from_added_lines",
    "extract_routes_from_lines",
    "extract_routes_from_path",
    "git",
    "is_ignored_path",
    "is_ui_file",
    "parse_added_lines_by_file",
    "parse_import_specifiers",
    "parse_name_status",
    "parse_numstat",
    "read_project_context",
    "resolve_import",
    "score_feature",
]

MAX_LABELS = 40
MAX_HANDLERS = 30
MAX_EXCERPT_LINES_PER_FILE = 40
MAX_STYLE_LINE_CHARS = 160
MAX_COMMITS = 30
MAX_COMMIT_SUBJECT_CHARS = 120
MAX_PROJECT_CHARS = 600
MAX_FEATURES = 8
MAX_FEATURE_LABELS = 12
MAX_FEATURE_COMMITS = 6

_IGNORED_DIR_NAMES: frozenset[str] = frozenset(
    {
        "node_modules",
        "dist",
        "build",
        ".next",
        "__snapshots__",
    }
)

_IGNORED_FILENAMES: frozenset[str] = frozenset(
    {
        "package-lock.json",
        "pnpm-lock.yaml",
        "yarn.lock",
        "uv.lock",
        "poetry.lock",
        "cargo.lock",
        "bun.lockb",
        "bun.lock",
        "composer.lock",
        "gemfile.lock",
        "pipfile.lock",
        "flake.lock",
    }
)

_IGNORED_EXTENSIONS: frozenset[str] = frozenset(
    {
        # Lockfiles, maps, snapshots
        ".lock",
        ".lockb",
        ".map",
        ".snap",
        # Images
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".webp",
        ".svg",
        ".ico",
        ".icns",
        ".bmp",
        ".tif",
        ".tiff",
        ".avif",
        ".heic",
        ".psd",
        # Fonts
        ".woff",
        ".woff2",
        ".ttf",
        ".otf",
        ".eot",
        # Media / documents / archives / compiled binaries
        ".pdf",
        ".zip",
        ".tar",
        ".gz",
        ".tgz",
        ".bz2",
        ".xz",
        ".7z",
        ".rar",
        ".mp4",
        ".mov",
        ".webm",
        ".avi",
        ".mkv",
        ".mp3",
        ".wav",
        ".ogg",
        ".flac",
        ".exe",
        ".dll",
        ".so",
        ".dylib",
        ".bin",
        ".wasm",
        ".pyc",
        ".pyo",
        ".class",
        ".jar",
        ".o",
        ".a",
        ".lib",
        ".sqlite",
        ".db",
    }
)

_UI_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".tsx",
        ".jsx",
        ".vue",
        ".svelte",
        ".astro",
        ".html",
        ".htm",
        ".css",
        ".scss",
        ".sass",
        ".less",
        ".mdx",
        ".erb",
        ".jinja",
        ".jinja2",
        ".j2",
        ".hbs",
        ".handlebars",
        ".twig",
        ".ejs",
        ".pug",
        ".liquid",
        ".razor",
        ".cshtml",
    }
)

_UI_SCRIPT_EXTENSIONS: frozenset[str] = frozenset({".ts", ".js"})
_UI_DIR_SEGMENTS: frozenset[str] = frozenset({"components", "app", "pages", "views", "ui"})

_ASSET_ROUTE_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".webp",
        ".svg",
        ".ico",
        ".bmp",
        ".avif",
        ".css",
        ".scss",
        ".sass",
        ".less",
        ".js",
        ".mjs",
        ".cjs",
        ".ts",
        ".tsx",
        ".jsx",
        ".map",
        ".json",
        ".xml",
        ".txt",
        ".pdf",
        ".woff",
        ".woff2",
        ".ttf",
        ".otf",
        ".eot",
        ".mp4",
        ".mp3",
        ".webm",
        ".wav",
        ".zip",
        ".wasm",
    }
)

_HTML_TAGS_UPPER: frozenset[str] = frozenset(
    {
        "A",
        "ABBR",
        "ADDRESS",
        "AREA",
        "ARTICLE",
        "ASIDE",
        "AUDIO",
        "B",
        "BASE",
        "BDI",
        "BDO",
        "BLOCKQUOTE",
        "BODY",
        "BR",
        "BUTTON",
        "CANVAS",
        "CAPTION",
        "CITE",
        "CODE",
        "COL",
        "COLGROUP",
        "DATA",
        "DATALIST",
        "DD",
        "DEL",
        "DETAILS",
        "DFN",
        "DIALOG",
        "DIV",
        "DL",
        "DT",
        "EM",
        "EMBED",
        "FIELDSET",
        "FIGCAPTION",
        "FIGURE",
        "FOOTER",
        "FORM",
        "FRAGMENT",
        "H1",
        "H2",
        "H3",
        "H4",
        "H5",
        "H6",
        "HEAD",
        "HEADER",
        "HGROUP",
        "HR",
        "HTML",
        "I",
        "IFRAME",
        "IMG",
        "INPUT",
        "INS",
        "KBD",
        "LABEL",
        "LEGEND",
        "LI",
        "LINK",
        "MAIN",
        "MAP",
        "MARK",
        "MENU",
        "META",
        "METER",
        "NAV",
        "NOSCRIPT",
        "OBJECT",
        "OL",
        "OPTGROUP",
        "OPTION",
        "OUTPUT",
        "P",
        "PARAM",
        "PICTURE",
        "PRE",
        "PROGRESS",
        "Q",
        "RP",
        "RT",
        "RUBY",
        "S",
        "SAMP",
        "SCRIPT",
        "SEARCH",
        "SECTION",
        "SELECT",
        "SLOT",
        "SMALL",
        "SOURCE",
        "SPAN",
        "STRONG",
        "STYLE",
        "SUB",
        "SUMMARY",
        "SUP",
        "SVG",
        "TABLE",
        "TBODY",
        "TD",
        "TEMPLATE",
        "TEXTAREA",
        "TFOOT",
        "TH",
        "THEAD",
        "TIME",
        "TITLE",
        "TR",
        "TRACK",
        "U",
        "UL",
        "VAR",
        "VIDEO",
        "WBR",
    }
)

_NON_COMPONENT_NAMES: frozenset[str] = frozenset(
    {
        "Array",
        "Boolean",
        "Date",
        "Error",
        "Fragment",
        "Function",
        "JSON",
        "Map",
        "Math",
        "Number",
        "Object",
        "Promise",
        "Proxy",
        "Record",
        "RegExp",
        "Set",
        "String",
        "Symbol",
        "URL",
        "URLSearchParams",
        "WeakMap",
        "WeakSet",
    }
)

_HANDLER_SKIP_VALUES: frozenset[str] = frozenset(
    {
        "async",
        "await",
        "false",
        "function",
        "new",
        "null",
        "return",
        "true",
        "undefined",
        "void",
    }
)

# Framework route path regexes
_RE_SVELTEKIT_ROUTE = re.compile(r"^(?:.*/)?src/routes(?:/(.*))?/\+page\.svelte$")
_RE_REMIX_ROUTE = re.compile(r"^(?:.*/)?app/routes/(.+)\.(?:tsx|jsx|ts|js)$")
_RE_NEXT_APP_ROUTE = re.compile(r"^(?:.*/)?app(?:/(.*))?/page\.(?:tsx|jsx|ts|js|mdx)$")
_RE_ASTRO_ROUTE = re.compile(r"^(?:.*/)?src/pages/(.+)\.astro$")
_RE_NUXT_ROUTE = re.compile(r"^(?:.*/)?pages/(.+)\.vue$")
_RE_NEXT_PAGES_ROUTE = re.compile(r"^(?:.*/)?pages/(.+)\.(?:tsx|jsx|ts|js|mdx)$")
_RE_ROUTE_GROUP = re.compile(r"^\([^)]+\)$")

# Route extraction from added lines
_RE_ROUTE_PROP = re.compile(r"""\bpath\s*:\s*["']([^"']+)["']""")
_RE_ROUTE_ATTR = re.compile(
    r"""(?<![A-Za-z0-9_-])(?:path|href|to)\s*=\s*(?:\{\s*["']([^"']+)["']\s*\}|["']([^"']+)["'])"""
)
_RE_ROUTE_NAVIGATE = re.compile(
    r"""\b(?:router\.(?:push|replace)|navigate)\s*\(\s*["']([^"']+)["']"""
)
_RE_ROUTE_DJANGO = re.compile(r"""(?<![.\w])path\s*\(\s*["']([^"']*)["']\s*,""")
_RE_ROUTE_DECORATOR = re.compile(
    r"""@\w+\.(?:route|get|post|put|patch|delete)\s*\(\s*["']([^"']+)["']"""
)

# Component extraction regexes
_RE_COMP_FUNC = re.compile(
    r"""\b(?:export\s+(?:default\s+)?)?(?:async\s+)?function\s+([A-Z][A-Za-z0-9_]*)\s*(?:<[^>]*>)?\s*\("""
)
_RE_COMP_CONST = re.compile(
    r"""\b(?:export\s+)?const\s+([A-Z][A-Za-z0-9_]*)(?:\s*:[^=]+)?\s*=\s*(?:(?:React\.)?(?:memo|forwardRef)\s*(?:<[^>]*>)?\s*\(|(?:async\s+)?\()"""
)
_RE_COMP_CLASS = re.compile(
    r"""\b(?:export\s+(?:default\s+)?)?class\s+([A-Z][A-Za-z0-9_]*)\s+extends\s+(?:React\.)?(?:Pure)?Component\b"""
)
_RE_COMP_JSX_TAG = re.compile(r"""(?<![A-Za-z0-9_$])<([A-Z][A-Za-z0-9_]+)(?=[\s/>])""")

# Label extraction regexes
_RE_LABEL_ATTR = re.compile(
    r"""(?<![A-Za-z0-9_:-])(?:aria-label|placeholder|title|alt|label)\s*=\s*(?:\{\s*(?:"([^"\\]*(?:\\.[^"\\]*)*)"|'([^'\\]*(?:\\.[^'\\]*)*)')\s*\}|"([^"]*)"|'([^']*)')"""
)
_RE_INLINE_LABEL_TAG = re.compile(
    r"""<(button|h[1-6]|label)\b(?:[^>"']|"[^"]*"|'[^']*'|=>)*>([^<>]+)</\1>""",
    re.IGNORECASE,
)
_RE_VISIBLE_TEXT = re.compile(r""">\s*([^<>{}=;\n]+?)\s*<""")
_RE_SCRIPT_OR_STYLE_BLOCK = re.compile(
    r"""<(script|style)\b[^>]*>.*?</\1>""",
    re.IGNORECASE | re.DOTALL,
)
_RE_CODE_NOISE_TEXT = re.compile(
    r"""(?:&&|\|\||=>|===|!==|==|!=|<=|>=|^\s*(?:import|export|return|const|let|var|function|class|from)\b)"""
)

# Test ID extraction regex
_RE_TEST_ID = re.compile(
    r"""(?<![A-Za-z0-9_-])(?:data-testid|data-test-id|data-test|data-cy|testID)\s*=\s*(?:\{\s*(?:"([^"]+)"|'([^']+)')\s*\}|"([^"]+)"|'([^']+)')"""
)

# Handler & state extraction regexes
_RE_JSX_HANDLER = re.compile(r"""\b(on[A-Z]\w*)\s*=\s*\{\s*([A-Za-z_$][\w$.]*)(?=\s*\})""")
_RE_USE_STATE = re.compile(
    r"""\bconst\s*\[\s*([A-Za-z_$]\w*)\s*,\s*[A-Za-z_$]\w*\s*\]\s*=\s*(?:React\.)?useState\b"""
)
_RE_VUE_HANDLER = re.compile(
    r"""(?:^|[\s<])(@[a-zA-Z][\w:.-]*)\s*=\s*["']\s*([A-Za-z_$][\w$.]*)(?:\s*\([^"']*\))?\s*["']"""
)
_RE_SVELTE_HANDLER = re.compile(
    r"""\b(on:[a-zA-Z][\w|.-]*)\s*=\s*\{\s*([A-Za-z_$][\w$.]*)\s*\}"""
)

# Excerpt filtering regexes
_RE_IMPORT_LINE = re.compile(r"""^\s*import(?:\s|\{|["'*])""")
_RE_PURE_STYLE_LINE = re.compile(r"""^\s*(?:className|class|style)\s*=""")


class GitError(RuntimeError):
    """Raised when a git command fails or a repository/ref cannot be resolved."""


def git(repo: Path, *args: str) -> str:
    """Run ``git -C <repo> <args...>`` and return stdout, raising ``GitError`` on failure."""
    try:
        proc = subprocess.run(
            ["git", "-C", str(repo), *args],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
    except OSError as exc:
        raise GitError(f"Failed to execute git ({' '.join(args)}): {exc}") from exc

    if proc.returncode != 0:
        detail = proc.stderr.strip() or proc.stdout.strip() or f"exit code {proc.returncode}"
        raise GitError(f"git {' '.join(args)} failed: {detail}")
    return proc.stdout


def _normalize_path(path: str) -> str:
    normalized = path.replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized


def is_ignored_path(path: str) -> bool:
    """Return True if ``path`` is diff noise (lockfiles, build output, snapshots, binaries)."""
    norm = _normalize_path(path)
    if not norm:
        return True
    lower = norm.lower()
    posix = PurePosixPath(lower)
    if any(part in _IGNORED_DIR_NAMES for part in posix.parts[:-1]):
        return True
    name = posix.name
    if name in _IGNORED_FILENAMES:
        return True
    if name.endswith((".min.js", ".min.css", ".map", ".lock", ".snap")):
        return True
    if posix.suffix in _IGNORED_EXTENSIONS:
        return True
    return False


def is_ui_file(path: str) -> bool:
    """Return True if ``path`` is a UI markup, template, component, or style file."""
    norm = _normalize_path(path)
    lower = norm.lower()
    if lower.endswith(".blade.php"):
        return True
    posix = PurePosixPath(lower)
    if posix.suffix in _UI_EXTENSIONS:
        return True
    if posix.suffix in _UI_SCRIPT_EXTENSIONS:
        parent_parts = posix.parts[:-1]
        if any(part in _UI_DIR_SEGMENTS for part in parent_parts):
            return True
        dir_str = "/" + "/".join(parent_parts) + "/"
        if "/src/routes/" in dir_str:
            return True
    return False


_FRAMEWORK_DEPS: dict[str, str] = {
    "next": "next",
    "nuxt": "nuxt",
    "nuxt3": "nuxt",
    "@sveltejs/kit": "sveltekit",
    "astro": "astro",
    "@remix-run/react": "remix",
    "@remix-run/node": "remix",
    "@remix-run/dev": "remix",
    "@react-router/dev": "remix",  # React Router v7 framework mode keeps the app/routes convention
}


def detect_frameworks(repo: Path) -> set[str] | None:
    """File-routing frameworks declared in the repo's package.json files.

    Returns None when the repo has no package.json at all (unknown stack: every convention stays allowed), and an
    empty set for a JS app without file-based routing (Vite + React Router, CRA...), where ``pages/Foo.tsx`` is just a
    component, not a URL.
    """
    try:
        listed = git(repo, "ls-files", "--", "package.json", "*/package.json", "*/*/package.json")
    except GitError:
        return None
    manifests = [p for p in listed.splitlines() if p and "node_modules" not in p]
    if not manifests:
        return None
    found: set[str] = set()
    for rel in manifests:
        try:
            data = json.loads((repo / rel).read_text(encoding="utf-8", errors="replace"))
        except (OSError, ValueError):
            continue
        if not isinstance(data, dict):
            continue
        for key in ("dependencies", "devDependencies", "peerDependencies"):
            deps = data.get(key)
            if isinstance(deps, dict):
                found.update(_FRAMEWORK_DEPS[d] for d in deps if d in _FRAMEWORK_DEPS)
    return found


# <Route path="/x" element={<Page />}>, <Route element={<Page />} path="/x">, { path: "/x", element: <Page /> }
_RE_ROUTE_PATH_FIRST = re.compile(
    r"""<Route\b[^>]*?\bpath=\{?\s*["'`]([^"'`]+)["'`]\s*\}?[^>]*?\b(?:element|component|Component)=\{\s*<?\s*([A-Z][A-Za-z0-9_]*)"""
)
_RE_ROUTE_ELEMENT_FIRST = re.compile(
    r"""<Route\b[^>]*?\b(?:element|component|Component)=\{\s*<?\s*([A-Z][A-Za-z0-9_]*)(?:(?!<Route\b).){0,200}?\bpath=\{?\s*["'`]([^"'`]+)["'`]""",
    re.S,
)
_RE_ROUTE_OBJECT = re.compile(
    r"""\bpath\s*:\s*["'`]([^"'`]+)["'`][^{}]{0,200}?\b(?:element|Component|component)\s*:\s*<?\s*([A-Z][A-Za-z0-9_]*)""",
    re.S,
)


def _clean_router_path(raw: str) -> str | None:
    p = raw.strip()
    if not p or p == "*" or "://" in p:
        return None
    p = p.removesuffix("/*").removesuffix("*") or "/"
    return p if p.startswith("/") else "/" + p


def map_router_components(sources: Iterable[str]) -> dict[str, str]:
    """Component name -> URL path from React Router / similar route tables (first declaration wins)."""
    mapping: dict[str, str] = {}
    for text in sources:
        pairs: list[tuple[str, str]] = []
        pairs += [(c, p) for p, c in _RE_ROUTE_PATH_FIRST.findall(text)]
        pairs += _RE_ROUTE_ELEMENT_FIRST.findall(text)
        pairs += [(c, p) for p, c in _RE_ROUTE_OBJECT.findall(text)]
        for comp, raw in pairs:
            path = _clean_router_path(raw)
            if path and comp not in ("Navigate", "Redirect", "Outlet") and comp not in mapping:
                mapping[comp] = path
    return mapping


def _router_sources(repo: Path) -> list[str]:
    try:
        listed = git(repo, "grep", "-l", "-E", r"<Route\b|createBrowserRouter|createRoutesFromElements|useRoutes\(",
                     "--", "*.tsx", "*.jsx", "*.ts", "*.js")  # fmt: skip
    except GitError:
        return []  # git grep exits 1 when nothing matches
    texts: list[str] = []
    for rel in listed.splitlines()[:20]:
        if not rel or is_ignored_path(rel):
            continue
        try:
            texts.append((repo / rel).read_text(encoding="utf-8", errors="replace"))
        except OSError:
            continue
    return texts


def extract_routes_from_path(path: str, frameworks: set[str] | None = None) -> str | None:
    """Infer a URL route path from a framework page/route file path, or None.

    ``frameworks`` (from :func:`detect_frameworks`) restricts which file-routing conventions apply; None allows all.
    """
    norm = _normalize_path(path)
    if not norm or is_ignored_path(norm):
        return None

    def allowed(name: str) -> bool:
        return frameworks is None or name in frameworks

    # SvelteKit: src/routes/**/+page.svelte
    m = _RE_SVELTEKIT_ROUTE.match(norm) if allowed("sveltekit") else None
    if m:
        raw_dir = m.group(1) or ""
        segs = [s for s in raw_dir.split("/") if s and not _RE_ROUTE_GROUP.match(s)]
        return "/" + "/".join(segs) if segs else "/"

    # Remix: app/routes/*.tsx (dots -> slashes, _index -> "/")
    m = _RE_REMIX_ROUTE.match(norm) if allowed("remix") else None
    if m and PurePosixPath(norm).stem != "page":
        rel = m.group(1)
        segs: list[str] = []
        for part in rel.split("/"):
            for seg in part.split("."):
                if not seg or seg in ("_index", "index"):
                    continue
                segs.append(seg)
        return "/" + "/".join(segs) if segs else "/"

    # Next.js App Router: app/**/page.(tsx|jsx|ts|js|mdx)
    m = _RE_NEXT_APP_ROUTE.match(norm) if allowed("next") else None
    if m:
        raw_dir = m.group(1) or ""
        segs = [s for s in raw_dir.split("/") if s and not _RE_ROUTE_GROUP.match(s)]
        return "/" + "/".join(segs) if segs else "/"

    # Astro: src/pages/**.astro
    m = _RE_ASTRO_ROUTE.match(norm) if allowed("astro") else None
    if m:
        segs = [s for s in m.group(1).split("/") if s]
        if segs and segs[-1] == "index":
            segs.pop()
        return "/" + "/".join(segs) if segs else "/"

    # Nuxt: pages/**.vue
    m = _RE_NUXT_ROUTE.match(norm) if allowed("nuxt") else None
    if m:
        segs = [s for s in m.group(1).split("/") if s]
        if segs and segs[-1] == "index":
            segs.pop()
        return "/" + "/".join(segs) if segs else "/"

    # Next.js Pages Router: pages/**.(tsx|jsx|ts|js|mdx)
    m = _RE_NEXT_PAGES_ROUTE.match(norm) if allowed("next") else None
    if m:
        segs = [s for s in m.group(1).split("/") if s]
        if not segs or segs[0] == "api" or any(s.startswith("_") for s in segs):
            return None
        if segs[-1] == "index":
            segs.pop()
        return "/" + "/".join(segs) if segs else "/"

    return None


def _normalize_route(raw: str, *, prepend_slash: bool = False) -> str | None:
    s = raw.strip()
    lower = s.lower()
    if lower.startswith(("http://", "https://", "mailto:", "tel:", "javascript:", "data:", "//")):
        return None
    if s.startswith("#"):
        return None

    if prepend_slash:
        if not s:
            s = "/"
        elif not s.startswith("/"):
            s = "/" + s

    if not s.startswith("/"):
        return None

    # Strip anchor fragment if attached to a path (e.g., /settings#billing -> /settings)
    if "#" in s:
        s = s.split("#", 1)[0]
    if not s:
        return None

    if any(ch.isspace() for ch in s):
        return None
    if any(tok in s for tok in ("${", "{{", "<", ">", "*", "\\", "`")):
        return None

    path_only = s.split("?", 1)[0]
    if PurePosixPath(path_only).suffix.lower() in _ASSET_ROUTE_EXTENSIONS:
        return None

    s = re.sub(r"/{2,}", "/", s)
    return s


def _dedupe_preserve(items: Iterable[str], *, case_insensitive: bool = False, limit: int | None = None) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        key = item.casefold() if case_insensitive else item
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
        if limit is not None and len(out) >= limit:
            break
    return out


def extract_routes_from_lines(lines: Sequence[str]) -> list[str]:
    """Extract normalized URL routes from added source lines."""
    found: list[str] = []
    for line in lines:
        for match in _RE_ROUTE_PROP.finditer(line):
            route = _normalize_route(match.group(1))
            if route is not None:
                found.append(route)
        for match in _RE_ROUTE_ATTR.finditer(line):
            raw = match.group(1) if match.group(1) is not None else match.group(2)
            route = _normalize_route(raw)
            if route is not None:
                found.append(route)
        for match in _RE_ROUTE_NAVIGATE.finditer(line):
            route = _normalize_route(match.group(1))
            if route is not None:
                found.append(route)
        for match in _RE_ROUTE_DJANGO.finditer(line):
            route = _normalize_route(match.group(1), prepend_slash=True)
            if route is not None:
                found.append(route)
        for match in _RE_ROUTE_DECORATOR.finditer(line):
            route = _normalize_route(match.group(1))
            if route is not None:
                found.append(route)
    return _dedupe_preserve(found)


def _clean_label_text(raw: str, *, strict_visible: bool) -> str | None:
    text = " ".join(raw.split())
    if not text:
        return None
    if not any(ch.isalpha() for ch in text):
        return None
    if strict_visible:
        if not 2 <= len(text) <= 60:
            return None
        if any(ch in text for ch in ("{", "}", "=", ";")):
            return None
        if _RE_CODE_NOISE_TEXT.search(text):
            return None
    else:
        if not 2 <= len(text) <= 120:
            return None
        if "${" in text or "{{" in text:
            return None
    return text


def extract_from_added_lines(
    path: str,
    lines: Sequence[str],
    *,
    is_added_file: bool = False,
) -> dict[str, list[str]]:
    """Extract UI signals and routes from added lines (leading '+' already stripped).

    Routes are extracted from any file; components, labels, test IDs, and handlers
    are extracted for UI / route files.
    """
    routes = extract_routes_from_lines(lines)

    norm = _normalize_path(path)
    ui_relevant = is_ui_file(norm) or extract_routes_from_path(norm) is not None
    if not ui_relevant:
        return {
            "routes": routes,
            "components": [],
            "labels": [],
            "test_ids": [],
            "handlers": [],
        }

    components: list[str] = []
    labels: list[str] = []
    test_ids: list[str] = []
    handlers: list[str] = []

    posix = PurePosixPath(norm)
    suffix_lower = posix.suffix.lower()
    if is_added_file and suffix_lower in (".vue", ".svelte"):
        stem = posix.stem
        if stem and not stem.startswith("+"):
            components.append(stem)

    for line in lines:
        # Components
        for pattern in (_RE_COMP_FUNC, _RE_COMP_CONST, _RE_COMP_CLASS):
            for m in pattern.finditer(line):
                name = m.group(1)
                if name not in _NON_COMPONENT_NAMES:
                    components.append(name)
        for m in _RE_COMP_JSX_TAG.finditer(line):
            name = m.group(1)
            if name not in _NON_COMPONENT_NAMES and name.upper() not in _HTML_TAGS_UPPER:
                components.append(name)

        # Attribute labels
        for m in _RE_LABEL_ATTR.finditer(line):
            raw = next((g for g in m.groups() if g is not None), "")
            cleaned = _clean_label_text(raw, strict_visible=False)
            if cleaned is not None:
                labels.append(cleaned)

        # Single-line <button>, <h1-6>, <label> text
        for m in _RE_INLINE_LABEL_TAG.finditer(line):
            cleaned = _clean_label_text(m.group(2), strict_visible=True)
            if cleaned is not None:
                labels.append(cleaned)

        # Test IDs
        for m in _RE_TEST_ID.finditer(line):
            raw = next((g for g in m.groups() if g is not None), "").strip()
            if raw:
                test_ids.append(raw)

        # Handlers & state
        for m in _RE_JSX_HANDLER.finditer(line):
            event_name, target = m.group(1), m.group(2)
            if target not in _HANDLER_SKIP_VALUES:
                handlers.append(f"{event_name}={target}")
        for m in _RE_USE_STATE.finditer(line):
            handlers.append(f"useState:{m.group(1)}")
        for m in _RE_VUE_HANDLER.finditer(line):
            event_name, target = m.group(1), m.group(2)
            if target not in _HANDLER_SKIP_VALUES:
                handlers.append(f"{event_name}={target}")
        for m in _RE_SVELTE_HANDLER.finditer(line):
            event_name, target = m.group(1), m.group(2)
            if target not in _HANDLER_SKIP_VALUES:
                handlers.append(f"{event_name}={target}")

    # Visible JSX/HTML text between tags (works for both single-line and multi-line tags)
    joined = "\n".join(lines)
    markup_only = _RE_SCRIPT_OR_STYLE_BLOCK.sub("", joined)
    for m in _RE_VISIBLE_TEXT.finditer(markup_only):
        cleaned = _clean_label_text(m.group(1), strict_visible=True)
        if cleaned is not None:
            labels.append(cleaned)

    return {
        "routes": routes,
        "components": _dedupe_preserve(components),
        "labels": _dedupe_preserve(labels, case_insensitive=True, limit=MAX_LABELS),
        "test_ids": _dedupe_preserve(test_ids),
        "handlers": _dedupe_preserve(handlers, limit=MAX_HANDLERS),
    }


def parse_name_status(output: str) -> dict[str, FileStatus]:
    """Parse ``git diff --name-status -M -z`` output into ``{path: FileStatus}``."""
    statuses: dict[str, FileStatus] = {}
    tokens = output.split("\0")
    idx = 0
    n = len(tokens)
    while idx < n:
        raw_status = tokens[idx]
        idx += 1
        if not raw_status:
            continue
        code = raw_status[0]
        if code in ("R", "C"):
            if idx + 1 >= n:
                break
            _old_path = tokens[idx]
            new_path = _normalize_path(tokens[idx + 1])
            idx += 2
            statuses[new_path] = "renamed" if code == "R" else "added"
        else:
            if idx >= n:
                break
            path = _normalize_path(tokens[idx])
            idx += 1
            if code == "A":
                statuses[path] = "added"
            elif code == "D":
                statuses[path] = "deleted"
            else:
                statuses[path] = "modified"
    return statuses


def parse_numstat(output: str) -> dict[str, tuple[int, int]]:
    """Parse ``git diff --numstat -M -z`` output into ``{path: (added, removed)}``."""
    stats: dict[str, tuple[int, int]] = {}
    tokens = output.split("\0")
    idx = 0
    n = len(tokens)
    while idx < n:
        entry = tokens[idx]
        idx += 1
        if not entry:
            continue
        parts = entry.split("\t", 2)
        if len(parts) < 3:
            continue
        added_str, removed_str, path_part = parts
        added = 0 if added_str == "-" else int(added_str)
        removed = 0 if removed_str == "-" else int(removed_str)
        if path_part == "":
            # Rename entry: followed by old_path\0new_path\0
            if idx + 1 >= n:
                break
            _old_path = tokens[idx]
            path = _normalize_path(tokens[idx + 1])
            idx += 2
        else:
            path = _normalize_path(path_part)
        stats[path] = (added, removed)
    return stats


def parse_added_lines_by_file(diff_text: str) -> dict[str, list[str]]:
    """Parse ``git diff -U3`` output into ``{path: [added_line_without_plus, ...]}``."""
    added_by_file: dict[str, list[str]] = {}
    current_path: str | None = None
    in_hunk = False

    for line in diff_text.splitlines():
        if line.startswith("diff --git "):
            current_path = None
            in_hunk = False
            continue
        if not in_hunk:
            if line.startswith("+++ "):
                target = line[4:].rstrip("\t")
                if target == "/dev/null":
                    current_path = None
                elif target.startswith("b/"):
                    current_path = _normalize_path(target[2:])
                else:
                    current_path = _normalize_path(target)
                continue
            if line.startswith("@@ "):
                in_hunk = True
                continue
        else:
            if line.startswith("@@ "):
                continue
            if line.startswith("+"):
                if current_path is not None:
                    added_by_file.setdefault(current_path, []).append(line[1:])
            elif line.startswith(("-", " ", "\\")):
                continue
            else:
                in_hunk = False

    return added_by_file


def _clean_excerpt_lines(lines: Sequence[str], *, max_lines: int = MAX_EXCERPT_LINES_PER_FILE) -> list[str]:
    cleaned: list[str] = []
    prev_blank = True
    for raw in lines:
        if _RE_IMPORT_LINE.match(raw):
            continue
        if len(raw) > MAX_STYLE_LINE_CHARS and (
            _RE_PURE_STYLE_LINE.match(raw)
            or (
                any(tok in raw for tok in ("className=", "class=", "style="))
                and not _RE_VISIBLE_TEXT.search(raw)
            )
        ):
            continue
        if not raw.strip():
            if not prev_blank:
                cleaned.append("")
                prev_blank = True
            continue
        cleaned.append(raw)
        prev_blank = False

    while cleaned and cleaned[-1] == "":
        cleaned.pop()

    return cleaned[:max_lines]


def build_excerpt(
    files: Sequence[ChangedFile],
    added_lines_by_file: dict[str, list[str]],
    *,
    max_excerpt_chars: int = 3500,
) -> str:
    """Build a compact added-lines excerpt prioritizing route files, then added lines desc."""
    if max_excerpt_chars <= 0:
        return ""

    ui_candidates = [
        f for f in files if (f.is_ui or f.is_route) and added_lines_by_file.get(f.path)
    ]
    ui_candidates.sort(key=lambda f: (not f.is_route, -f.added, f.path))

    blocks: list[str] = []
    used = 0
    for changed in ui_candidates:
        cleaned = _clean_excerpt_lines(added_lines_by_file[changed.path])
        if not cleaned:
            continue
        block = f"--- {changed.path}\n" + "\n".join(cleaned)
        sep_len = 2 if blocks else 0
        if used + sep_len + len(block) <= max_excerpt_chars:
            blocks.append(block)
            used += sep_len + len(block)
        else:
            remaining = max_excerpt_chars - used - sep_len
            header = f"--- {changed.path}\n"
            if remaining > len(header) + 20:
                blocks.append(block[:remaining].rstrip())
            break

    return "\n\n".join(blocks)[:max_excerpt_chars]


def _ref_exists(repo: Path, ref: str) -> bool:
    try:
        git(repo, "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}")
    except GitError:
        return False
    return True


def _resolve_base_ref(repo: Path, base: str) -> str:
    candidates: list[str] = []
    for cand in (base, f"origin/{base}", "master", "origin/master"):
        if cand not in candidates:
            candidates.append(cand)
    for cand in candidates:
        if _ref_exists(repo, cand):
            return cand
    tried = ", ".join(repr(c) for c in candidates)
    raise GitError(
        f"Could not resolve base ref {base!r} (tried {tried}). "
        "Pass an explicit base branch/commit or fetch from origin."
    )


def _read_untracked_lines(repo: Path, rel_path: str) -> list[str] | None:
    """Read lines from an untracked file, returning None for binary/unreadable files."""
    full_path = repo / rel_path
    try:
        if not full_path.is_file():
            return None
        raw = full_path.read_bytes()
    except OSError:
        return None
    if b"\x00" in raw[:8192]:
        return None
    return raw.decode("utf-8", errors="replace").splitlines()


_STYLE_EXTENSIONS: frozenset[str] = frozenset({".css", ".scss", ".sass", ".less"})
_IMPORT_EXTENSIONS: tuple[str, ...] = (".tsx", ".ts", ".jsx", ".js", ".vue", ".svelte")
_ROUTE_SOURCE_EXTENSIONS: frozenset[str] = frozenset(
    {".tsx", ".ts", ".jsx", ".js", ".vue", ".svelte", ".astro", ".mdx"}
)
_GENERIC_ROUTE_STEMS: frozenset[str] = frozenset({"page", "+page", "index", "_index"})
_README_CANDIDATES: tuple[str, ...] = (
    "readme.md",
    "readme",
    "readme.markdown",
    "readme.txt",
    "readme.rst",
)

_RE_NOISE_COMMIT = re.compile(
    r"^(?:wip|fixup!|squash!|merge |bump |chore\(deps\)|format|lint)",
    re.IGNORECASE,
)
_RE_HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
_RE_HTML_TAG = re.compile(r"<[^>]+>")
_RE_MD_HEADING = re.compile(r"^\s*#+\s*")
_RE_MD_LINK = re.compile(r"\[([^\]]+)\]\([^)]+\)")

_RE_IMPORT_FROM = re.compile(
    r"""(?m)^\s*import\s+(?:type\s+)?(?:[\w$*,\s{}]+?)\bfrom\s*["']([^"']+)["']"""
)
_RE_EXPORT_FROM = re.compile(
    r"""(?m)^\s*export\s+(?:type\s+)?(?:\*(?:\s+as\s+[\w$]+)?|\{[^}]*\})\s*from\s*["']([^"']+)["']"""
)
_RE_IMPORT_SIDE_EFFECT = re.compile(r"""(?m)^\s*import\s*["']([^"']+)["']""")
_RE_IMPORT_DYNAMIC = re.compile(
    r"""\bimport\s*\(\s*(?:/\*.*?\*/\s*)?["']([^"']+)["']\s*\)"""
)


def _clean_commit_subject(raw: str) -> str | None:
    subject = raw.strip()
    if not subject or _RE_NOISE_COMMIT.match(subject):
        return None
    return subject[:MAX_COMMIT_SUBJECT_CHARS].rstrip()


def _parse_commit_log(output: str) -> tuple[list[str], list[tuple[str, frozenset[str]]]]:
    """Parse ``git log --no-merges --format=%x1e%s --name-only`` output."""
    if not output:
        return [], []
    subjects: list[str] = []
    entries: list[tuple[str, frozenset[str]]] = []
    for chunk in output.split("\x1e")[1:]:
        lines = chunk.splitlines()
        if not lines:
            continue
        subject = _clean_commit_subject(lines[0])
        if subject is None:
            continue
        touched = frozenset(_normalize_path(ln) for ln in lines[1:] if ln.strip())
        subjects.append(subject)
        entries.append((subject, touched))
    return _dedupe_preserve(subjects, limit=MAX_COMMITS), entries


def _clean_readme_text(raw: str, *, skip_title: str = "") -> str:
    kept_lines: list[str] = []
    in_fence = False
    for line in raw.splitlines():
        stripped = line.strip()
        if stripped.startswith(("```", "~~~")):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        lower = line.lower()
        if "![" in line or "<img" in lower or "shields.io" in lower:
            continue
        kept_lines.append(line)

    text = "\n".join(kept_lines)
    text = _RE_HTML_COMMENT.sub(" ", text)
    text = _RE_HTML_TAG.sub(" ", text)

    cleaned_lines: list[str] = []
    total_chars = 0
    for line in text.splitlines():
        line = _RE_MD_HEADING.sub("", line)
        line = _RE_MD_LINK.sub(r"\1", line)
        collapsed = " ".join(line.split())
        if not collapsed or not any(ch.isalnum() for ch in collapsed):
            continue
        cleaned_lines.append(collapsed)
        total_chars += len(collapsed) + 1
        if total_chars > MAX_PROJECT_CHARS * 2:
            break

    if (
        skip_title
        and len(cleaned_lines) > 1
        and cleaned_lines[0].casefold() == skip_title.casefold()
    ):
        cleaned_lines.pop(0)

    return " ".join(cleaned_lines)


def read_project_context(repo: Path) -> str:
    """Build a compact description of the project (max 600 chars) from manifest + README.

    Never raises; returns ``""`` on any error.
    """
    try:
        repo_path = Path(repo)
        if not repo_path.is_dir():
            return ""

        pkg_name = ""
        pkg_desc = ""

        pkg_json = repo_path / "package.json"
        if pkg_json.is_file():
            try:
                data = json.loads(pkg_json.read_text(encoding="utf-8", errors="replace"))
            except (OSError, ValueError):
                data = None
            if isinstance(data, dict):
                raw_name = data.get("name")
                raw_desc = data.get("description")
                if isinstance(raw_name, str):
                    pkg_name = " ".join(raw_name.split())
                if isinstance(raw_desc, str):
                    pkg_desc = " ".join(raw_desc.split())

        if not pkg_name and not pkg_desc:
            pyproject = repo_path / "pyproject.toml"
            if pyproject.is_file():
                try:
                    toml_data = tomllib.loads(pyproject.read_text(encoding="utf-8", errors="replace"))
                except (OSError, tomllib.TOMLDecodeError, ValueError):
                    toml_data = None
                if isinstance(toml_data, dict):
                    proj = toml_data.get("project")
                    if isinstance(proj, dict):
                        raw_name = proj.get("name")
                        raw_desc = proj.get("description")
                        if isinstance(raw_name, str):
                            pkg_name = " ".join(raw_name.split())
                        if isinstance(raw_desc, str):
                            pkg_desc = " ".join(raw_desc.split())

        readme_text = ""
        try:
            root_files = {
                entry.name.lower(): entry
                for entry in repo_path.iterdir()
                if entry.is_file()
            }
        except OSError:
            root_files = {}

        readme_file: Path | None = None
        for cand in _README_CANDIDATES:
            if cand in root_files:
                readme_file = root_files[cand]
                break

        if readme_file is not None:
            try:
                with readme_file.open("r", encoding="utf-8", errors="replace") as fh:
                    raw_readme = fh.read(16_384)
                readme_text = _clean_readme_text(raw_readme, skip_title=pkg_name)
            except OSError:
                readme_text = ""

        parts = [p for p in (pkg_name, pkg_desc, readme_text) if p]
        if not parts:
            return ""

        result = " ".join(" - ".join(parts).split())
        if len(result) > MAX_PROJECT_CHARS:
            result = result[: MAX_PROJECT_CHARS - 3].rstrip() + "..."
        return result
    except Exception:
        return ""


def parse_import_specifiers(text: str) -> list[str]:
    """Extract JS/TS/TSX import & re-export specifiers from source text in order."""
    found: list[tuple[int, str]] = []
    for pattern in (_RE_IMPORT_FROM, _RE_EXPORT_FROM, _RE_IMPORT_SIDE_EFFECT, _RE_IMPORT_DYNAMIC):
        for match in pattern.finditer(text):
            spec = match.group(1).strip()
            if spec:
                found.append((match.start(), spec))
    found.sort(key=lambda item: item[0])
    return _dedupe_preserve(spec for _, spec in found)


def _join_posix_relative(base_parts: Sequence[str], rel_spec: str) -> str | None:
    parts = list(base_parts)
    for seg in rel_spec.split("/"):
        if not seg or seg == ".":
            continue
        if seg == "..":
            if not parts:
                return None
            parts.pop()
        else:
            parts.append(seg)
    return "/".join(parts) if parts else None


def resolve_import(spec: str, from_file: str, known_files: Iterable[str]) -> str | None:
    """Resolve a relative or ``@/`` import specifier against ``known_files``."""
    cleaned = spec.strip().split("?", 1)[0].split("#", 1)[0]
    if not cleaned:
        return None

    if cleaned.startswith("@/"):
        base = _join_posix_relative(["src"], cleaned[2:])
    elif cleaned.startswith(("./", "../")) or cleaned in (".", ".."):
        parent = PurePosixPath(_normalize_path(from_file)).parent
        base_parts = [] if str(parent) in ("", ".") else list(parent.parts)
        base = _join_posix_relative(base_parts, cleaned)
    else:
        return None

    if not base:
        return None

    known_set = (
        known_files
        if isinstance(known_files, (set, frozenset))
        else {_normalize_path(p) for p in known_files}
    )

    if base in known_set:
        return base
    for ext in _IMPORT_EXTENSIONS:
        cand = f"{base}{ext}"
        if cand in known_set:
            return cand
    for ext in _IMPORT_EXTENSIONS:
        cand = f"{base}/index{ext}"
        if cand in known_set:
            return cand
    if base.endswith(".js"):
        stem = base[:-3]
        for ext in (".ts", ".tsx"):
            cand = f"{stem}{ext}"
            if cand in known_set:
                return cand
    elif base.endswith(".jsx"):
        cand = f"{base[:-4]}.tsx"
        if cand in known_set:
            return cand
    return None


def score_feature(
    added: int,
    new_files: int,
    label_count: int,
    commit_count: int,
    route: str | None,
) -> float:
    """Compute a feature candidate's demo-worthiness score, rounded to 2 decimals."""
    is_static = route is not None and not any(ch in route for ch in (":", "[", "*"))
    raw = (
        math.log2(1 + max(0, added))
        + 1.5 * max(0, new_files)
        + 0.4 * min(max(0, label_count), 10)
        + 1.0 * min(max(0, commit_count), 4)
        + (1.5 if route is not None else 0.0)
        + (1.0 if is_static else 0.0)
    )
    return round(raw, 2)


def _extract_primary_component(text: str) -> str | None:
    for pattern in (_RE_COMP_FUNC, _RE_COMP_CONST, _RE_COMP_CLASS):
        match = pattern.search(text)
        if match:
            name = match.group(1)
            if name not in _NON_COMPONENT_NAMES:
                return name
    return None


def _route_file_title(
    path: str,
    *,
    from_router: bool,
    added_lines_by_file: Mapping[str, Sequence[str]],
    read_file: Callable[[str], str | None] | None,
) -> str:
    stem = PurePosixPath(path).stem
    if from_router or stem.lower() not in _GENERIC_ROUTE_STEMS:
        return stem
    lines = added_lines_by_file.get(path)
    if lines:
        comp = _extract_primary_component("\n".join(lines))
        if comp:
            return comp
    if read_file is not None:
        text = read_file(path)
        if text:
            comp = _extract_primary_component(text)
            if comp:
                return comp
    return stem


def _make_repo_reader(repo: Path) -> Callable[[str], str | None]:
    cache: dict[str, str | None] = {}

    def _read(rel_path: str) -> str | None:
        if rel_path not in cache:
            full = repo / rel_path
            try:
                cache[rel_path] = (
                    full.read_text(encoding="utf-8", errors="replace") if full.is_file() else None
                )
            except OSError:
                cache[rel_path] = None
        return cache[rel_path]

    return _read


def build_feature_candidates(
    changed_files: Sequence[ChangedFile],
    added_lines_by_file: Mapping[str, Sequence[str]],
    *,
    frameworks: set[str] | None = None,
    router_map: Mapping[str, str] | None = None,
    known_files: Collection[str] | None = None,
    read_file: Callable[[str], str | None] | None = None,
    commit_entries: Sequence[tuple[str, Collection[str]]] = (),
    max_features: int = MAX_FEATURES,
) -> list[FeatureCandidate]:
    """Group changed UI files into scored, route-keyed ``FeatureCandidate`` objects."""
    ui_changed = [
        f
        for f in changed_files
        if f.status != "deleted" and not is_ignored_path(f.path) and (f.is_ui or f.is_route)
    ]
    if not ui_changed:
        return []

    known_set: set[str] = (
        {_normalize_path(p) for p in known_files if p and not is_ignored_path(p)}
        if known_files is not None
        else set()
    )
    known_set.update(_normalize_path(f.path) for f in changed_files if f.status != "deleted")

    # (a) path route, else (b) SPA router map
    assigned: dict[str, tuple[str, str]] = {}
    for f in ui_changed:
        route = extract_routes_from_path(f.path, frameworks)
        if route is not None:
            title = _route_file_title(
                f.path,
                from_router=False,
                added_lines_by_file=added_lines_by_file,
                read_file=read_file,
            )
            assigned[f.path] = (route, title)
            continue
        if router_map:
            posix = PurePosixPath(f.path)
            if posix.suffix.lower() not in _STYLE_EXTENSIONS:
                route = router_map.get(posix.stem)
                if route is not None:
                    assigned[f.path] = (route, posix.stem)

    # (c) 2-level reverse import graph limited to known route files from (a)/(b)
    unassigned_paths = {f.path for f in ui_changed if f.path not in assigned}
    if unassigned_paths and read_file is not None:
        route_files: list[tuple[str, str, str]] = []
        seen_route_paths: set[str] = set()

        for f in sorted(ui_changed, key=lambda cf: (-cf.added, cf.path)):
            if f.path in assigned:
                route, title = assigned[f.path]
                route_files.append((f.path, route, title))
                seen_route_paths.add(f.path)

        unchanged_route_files: list[tuple[str, str, str]] = []
        for path in sorted(known_set):
            if path in seen_route_paths:
                continue
            posix = PurePosixPath(path)
            if posix.suffix.lower() not in _ROUTE_SOURCE_EXTENSIONS:
                continue
            route = extract_routes_from_path(path, frameworks)
            if route is not None:
                title = _route_file_title(
                    path,
                    from_router=False,
                    added_lines_by_file=added_lines_by_file,
                    read_file=read_file,
                )
                unchanged_route_files.append((path, route, title))
                seen_route_paths.add(path)
                continue
            if router_map:
                route = router_map.get(posix.stem)
                if route is not None:
                    unchanged_route_files.append((path, route, posix.stem))
                    seen_route_paths.add(path)

        route_files.extend(unchanged_route_files)
        all_route_file_paths = {r_path for r_path, _, _ in route_files}

        resolved_imports_cache: dict[str, list[str]] = {}

        def _get_imports(file_path: str) -> list[str]:
            if file_path not in resolved_imports_cache:
                text = read_file(file_path)
                if not text:
                    resolved_imports_cache[file_path] = []
                else:
                    targets: list[str] = []
                    for spec in parse_import_specifiers(text):
                        resolved = resolve_import(spec, file_path, known_set)
                        if resolved is not None and resolved != file_path:
                            targets.append(resolved)
                    resolved_imports_cache[file_path] = _dedupe_preserve(targets)
            return resolved_imports_cache[file_path]

        # Depth 1
        level1_by_route: list[tuple[str, str, list[str]]] = []
        for r_path, route, title in route_files:
            imports_1 = _get_imports(r_path)
            level1_by_route.append((route, title, imports_1))
            for target in imports_1:
                if target in unassigned_paths and target not in assigned:
                    assigned[target] = (route, title)

        # Depth 2 (only if unassigned files remain)
        if unassigned_paths - assigned.keys():
            for route, title, imports_1 in level1_by_route:
                for mid in imports_1:
                    if mid in all_route_file_paths:
                        continue
                    if PurePosixPath(mid).suffix.lower() not in _IMPORT_EXTENSIONS:
                        continue
                    for target2 in _get_imports(mid):
                        if target2 in unassigned_paths and target2 not in assigned:
                            assigned[target2] = (route, title)

    # Group files: merge by route when assigned, else (d) route=None keyed by file stem
    groups: dict[tuple[str, str], list[ChangedFile]] = {}
    group_meta: dict[tuple[str, str], tuple[str, str | None]] = {}
    for f in sorted(ui_changed, key=lambda cf: (-cf.added, cf.path)):
        if f.path in assigned:
            route, title = assigned[f.path]
            key = ("route", route)
            if key not in group_meta:
                group_meta[key] = (title, route)
        else:
            stem = PurePosixPath(f.path).stem
            key = ("stem", stem)
            if key not in group_meta:
                group_meta[key] = (stem, None)
        groups.setdefault(key, []).append(f)

    def _is_style_only(group_files: Sequence[ChangedFile]) -> bool:
        return all(PurePosixPath(cf.path).suffix.lower() in _STYLE_EXTENSIONS for cf in group_files)

    has_non_style = any(not _is_style_only(g) for g in groups.values())

    candidates: list[FeatureCandidate] = []
    for key, group_files in groups.items():
        if has_non_style and _is_style_only(group_files):
            continue
        title, route = group_meta[key]
        sorted_files = sorted(group_files, key=lambda cf: (-cf.added, cf.path))
        file_paths = [cf.path for cf in sorted_files]
        added = sum(cf.added for cf in sorted_files)
        new_files = sum(1 for cf in sorted_files if cf.status == "added")

        raw_labels: list[str] = []
        for cf in sorted_files:
            lines = added_lines_by_file.get(cf.path, ())
            if not lines:
                continue
            extracted = extract_from_added_lines(
                cf.path,
                lines,
                is_added_file=(cf.status == "added"),
            )
            raw_labels.extend(extracted["labels"])
        group_labels = _dedupe_preserve(
            raw_labels,
            case_insensitive=True,
            limit=MAX_FEATURE_LABELS,
        )

        group_path_set = set(file_paths)
        group_commits = _dedupe_preserve(
            (subj for subj, touched in commit_entries if group_path_set.intersection(touched)),
            limit=MAX_FEATURE_COMMITS,
        )

        score = score_feature(
            added=added,
            new_files=new_files,
            label_count=len(group_labels),
            commit_count=len(group_commits),
            route=route,
        )
        candidates.append(
            FeatureCandidate(
                title=title,
                route=route,
                files=file_paths,
                added=added,
                new_files=new_files,
                labels=group_labels,
                commits=group_commits,
                score=score,
            )
        )

    candidates.sort(key=lambda c: (-c.score, -c.added, c.route or "", c.title))
    return candidates[:max_features]


def analyze_diff(
    repo: Path = Path("."),
    base: str = "main",
    *,
    include_worktree: bool = True,
    max_excerpt_chars: int = 3500,
) -> DiffSummary:
    """Analyze git changes in ``repo`` relative to ``base`` and return a ``DiffSummary``."""
    repo_path = Path(repo)
    resolved_base = _resolve_base_ref(repo_path, base)
    merge_base = git(repo_path, "merge-base", resolved_base, "HEAD").strip()

    head_branch = git(repo_path, "rev-parse", "--abbrev-ref", "HEAD").strip()
    is_dirty = bool(git(repo_path, "status", "--porcelain").strip()) if include_worktree else False
    head_ref = f"{head_branch} +worktree" if (include_worktree and is_dirty) else head_branch

    diff_target = [merge_base] if include_worktree else [f"{merge_base}..HEAD"]
    common_git = ["-c", "core.quotepath=false"]

    name_status_out = git(
        repo_path, *common_git, "diff", "--no-ext-diff", "--name-status", "-M", "-z", *diff_target
    )
    numstat_out = git(
        repo_path, *common_git, "diff", "--no-ext-diff", "--numstat", "-M", "-z", *diff_target
    )
    patch_out = git(
        repo_path, *common_git, "diff", "--no-ext-diff", "-U3", "-M", *diff_target
    )

    try:
        commit_log_out = git(
            repo_path,
            *common_git,
            "log",
            "--no-merges",
            "--format=%x1e%s",
            "--name-only",
            f"{merge_base}..HEAD",
        )
    except GitError:
        commit_log_out = ""
    commits, commit_entries = _parse_commit_log(commit_log_out)

    statuses = parse_name_status(name_status_out)
    numstats = parse_numstat(numstat_out)
    added_lines_by_file = parse_added_lines_by_file(patch_out)

    # Collect untracked files when include_worktree is enabled
    untracked_paths: list[str] = []
    if include_worktree:
        untracked_out = git(
            repo_path, *common_git, "ls-files", "--others", "--exclude-standard", "-z"
        )
        untracked_paths = [
            _normalize_path(p) for p in untracked_out.split("\0") if p
        ]

    frameworks = detect_frameworks(repo_path)
    changed_files: list[ChangedFile] = []
    seen_paths: set[str] = set()

    for path, status in statuses.items():
        if is_ignored_path(path):
            continue
        seen_paths.add(path)
        added, removed = numstats.get(path, (0, 0))
        route = extract_routes_from_path(path, frameworks)
        changed_files.append(
            ChangedFile(
                path=path,
                status=status,
                is_ui=is_ui_file(path),
                is_route=route is not None,
                added=added,
                removed=removed,
            )
        )

    for path in untracked_paths:
        if path in seen_paths or is_ignored_path(path):
            continue
        seen_paths.add(path)
        lines = _read_untracked_lines(repo_path, path)
        if lines is not None:
            added_lines_by_file[path] = lines
            added_count = len(lines)
        else:
            added_count = 0
        route = extract_routes_from_path(path, frameworks)
        changed_files.append(
            ChangedFile(
                path=path,
                status="added",
                is_ui=is_ui_file(path),
                is_route=route is not None,
                added=added_count,
                removed=0,
            )
        )

    # File-path routes from added/modified/renamed UI files come first
    file_route_candidates: list[tuple[ChangedFile, str]] = []
    for f in changed_files:
        if f.status == "deleted" or not (f.is_ui or f.is_route):
            continue
        route = extract_routes_from_path(f.path, frameworks)
        if route is not None:
            file_route_candidates.append((f, route))

    file_route_candidates.sort(
        key=lambda item: (item[0].status != "added", item[1] == "/", -item[0].added, item[0].path)
    )
    path_routes = [route for _, route in file_route_candidates]

    # Extract signals from added lines (prioritize UI route files, then UI files, then others)
    ordered_for_extraction = sorted(
        changed_files,
        key=lambda f: (not f.is_route, not f.is_ui, f.status != "added", -f.added, f.path),
    )

    line_routes: list[str] = []
    all_components: list[str] = []
    all_labels: list[str] = []
    all_test_ids: list[str] = []
    all_handlers: list[str] = []

    for f in ordered_for_extraction:
        lines = added_lines_by_file.get(f.path, [])
        if not lines and not (f.status == "added" and PurePosixPath(f.path).suffix.lower() in (".vue", ".svelte")):
            continue
        extracted = extract_from_added_lines(
            f.path,
            lines,
            is_added_file=(f.status == "added"),
        )
        line_routes.extend(extracted["routes"])
        all_components.extend(extracted["components"])
        all_labels.extend(extracted["labels"])
        all_test_ids.extend(extracted["test_ids"])
        all_handlers.extend(extracted["handlers"])

    # SPA routers (React Router...): a changed page component maps to the path its <Route> declares.
    router_map = map_router_components(_router_sources(repo_path))
    component_routes: list[str] = []
    if router_map and not path_routes:
        for f in sorted(changed_files, key=lambda f: (f.status == "deleted", -f.added, f.path)):
            if f.status == "deleted" or not f.is_ui:
                continue
            route = router_map.get(PurePosixPath(f.path).stem)
            if route is not None:
                f.is_route = True
                component_routes.append(route)

    try:
        tracked_out = git(repo_path, *common_git, "ls-files", "-z")
        known_files = {_normalize_path(p) for p in tracked_out.split("\0") if p}
    except GitError:
        known_files = set()
    known_files.update(untracked_paths)

    features = build_feature_candidates(
        changed_files,
        added_lines_by_file,
        frameworks=frameworks,
        router_map=router_map,
        known_files=known_files,
        read_file=_make_repo_reader(repo_path),
        commit_entries=commit_entries,
    )
    project = read_project_context(repo_path)

    routes = _dedupe_preserve([*path_routes, *component_routes, *line_routes])
    components = _dedupe_preserve(all_components)
    labels = _dedupe_preserve(all_labels, case_insensitive=True, limit=MAX_LABELS)
    test_ids = _dedupe_preserve(all_test_ids)
    handlers = _dedupe_preserve(all_handlers, limit=MAX_HANDLERS)

    excerpt = build_excerpt(
        changed_files,
        added_lines_by_file,
        max_excerpt_chars=max_excerpt_chars,
    )

    return DiffSummary(
        base_ref=resolved_base,
        head_ref=head_ref,
        project=project,
        commits=commits,
        features=features,
        files=changed_files,
        components=components,
        labels=labels,
        test_ids=test_ids,
        routes=routes,
        handlers=handlers,
        excerpt=excerpt,
    )
