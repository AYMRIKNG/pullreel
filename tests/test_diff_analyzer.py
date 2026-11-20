"""Unit and integration tests for pullreel.diff_analyzer."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from pullreel.diff_analyzer import (
    GitError,
    analyze_diff,
    build_excerpt,
    extract_from_added_lines,
    extract_routes_from_lines,
    extract_routes_from_path,
    is_ignored_path,
    is_ui_file,
    parse_name_status,
    parse_numstat,
)
from pullreel.models import ChangedFile


@pytest.mark.parametrize(
    ("file_path", "expected"),
    [
        # Next.js App Router
        ("app/page.tsx", "/"),
        ("src/app/settings/billing/page.tsx", "/settings/billing"),
        ("app/(marketing)/pricing/page.mdx", "/pricing"),
        ("app/(dashboard)/users/[id]/page.jsx", "/users/[id]"),
        # Next.js Pages Router
        ("pages/index.tsx", "/"),
        ("pages/users/[id].tsx", "/users/[id]"),
        ("pages/_app.tsx", None),
        ("pages/_document.tsx", None),
        ("pages/api/webhooks/stripe.ts", None),
        # SvelteKit
        ("src/routes/+page.svelte", "/"),
        ("src/routes/(app)/projects/[slug]/+page.svelte", "/projects/[slug]"),
        # Nuxt
        ("pages/index.vue", "/"),
        ("pages/settings/billing.vue", "/settings/billing"),
        # Remix
        ("app/routes/_index.tsx", "/"),
        ("app/routes/settings.billing.tsx", "/settings/billing"),
        ("app/routes/users.$id.tsx", "/users/$id"),
        # Astro
        ("src/pages/index.astro", "/"),
        ("src/pages/blog/[slug].astro", "/blog/[slug]"),
        # Non-route files
        ("components/PlanBadge.tsx", None),
        ("src/utils/format.ts", None),
    ],
)
def test_extract_routes_from_path(file_path: str, expected: str | None) -> None:
    assert extract_routes_from_path(file_path) == expected


def test_is_ignored_and_is_ui() -> None:
    assert is_ignored_path("package-lock.json")
    assert is_ignored_path("pnpm-lock.yaml")
    assert is_ignored_path("uv.lock")
    assert is_ignored_path("Cargo.lock")
    assert is_ignored_path("custom.lock")
    assert is_ignored_path("node_modules/react/index.js")
    assert is_ignored_path("dist/bundle.js")
    assert is_ignored_path("build/index.html")
    assert is_ignored_path(".next/static/chunks/app.js")
    assert is_ignored_path("public/vendor.min.js")
    assert is_ignored_path("src/app.js.map")
    assert is_ignored_path("src/__snapshots__/Button.test.tsx.snap")
    assert is_ignored_path("assets/logo.png")
    assert is_ignored_path("fonts/inter.woff2")
    assert not is_ignored_path("app/settings/billing/page.tsx")

    assert is_ui_file("app/settings/billing/page.tsx")
    assert is_ui_file("views/home.blade.php")
    assert is_ui_file("templates/email.jinja2")
    assert is_ui_file("components/helpers.ts")
    assert is_ui_file("src/routes/settings/+page.ts")
    assert not is_ui_file("src/server/db.ts")
    assert not is_ui_file("src/pullreel/diff_analyzer.py")


def test_extract_from_added_lines_jsx() -> None:
    jsx_lines = [
        'import React, { useState, forwardRef } from "react";',
        "export default function BillingPage() {",
        "  const [plan, setPlan] = useState('pro');",
        "  const handleUpgrade = () => router.push('/checkout/confirm');",
        "  return (",
        "    <section>",
        "      <h1>Billing & Plans</h1>",
        '      <PlanBadge label="Pro Tier" />',
        "      <input",
        '        placeholder="Card number"',
        '        aria-label={"Credit card number"}',
        '        data-test-id="card-input"',
        "      />",
        '      <button',
        '        title="Switch to annual"',
        '        data-testid="upgrade-btn"',
        "        onClick={handleUpgrade}",
        "      >",
        "        Upgrade plan",
        "      </button>",
        '      <a href="/invoices">View invoices</a>',
        '      <a href="https://example.com/docs">External docs</a>',
        '      <a href="#top">Back to top</a>',
        '      <img src="/logo.png" alt="Company logo" />',
        "      <span>{plan}</span>",
        "    </section>",
        "  );",
        "}",
        "const CardFooter = React.memo(() => <footer>Secure checkout</footer>);",
        "const CustomInput = forwardRef((props, ref) => <input ref={ref} {...props} />);",
        "class LegacyBanner extends React.Component { render() { return <div>Legacy</div>; } }",
    ]

    extracted = extract_from_added_lines("app/settings/billing/page.tsx", jsx_lines, is_added_file=True)

    assert "BillingPage" in extracted["components"]
    assert "PlanBadge" in extracted["components"]
    assert "CardFooter" in extracted["components"]
    assert "CustomInput" in extracted["components"]
    assert "LegacyBanner" in extracted["components"]

    assert "Billing & Plans" in extracted["labels"]
    assert "Pro Tier" in extracted["labels"]
    assert "Card number" in extracted["labels"]
    assert "Credit card number" in extracted["labels"]
    assert "Switch to annual" in extracted["labels"]
    assert "Upgrade plan" in extracted["labels"]
    assert "View invoices" in extracted["labels"]
    assert "Company logo" in extracted["labels"]
    assert "{plan}" not in extracted["labels"]

    assert "card-input" in extracted["test_ids"]
    assert "upgrade-btn" in extracted["test_ids"]

    assert "useState:plan" in extracted["handlers"]
    assert "onClick=handleUpgrade" in extracted["handlers"]

    assert "/checkout/confirm" in extracted["routes"]
    assert "/invoices" in extracted["routes"]
    assert "https://example.com/docs" not in extracted["routes"]
    assert "#top" not in extracted["routes"]
    assert "/logo.png" not in extracted["routes"]


def test_extract_from_added_lines_vue_and_svelte() -> None:
    vue_lines = [
        "<script setup lang='ts'>",
        "import { ref } from 'vue'",
        "const email = ref('')",
        "function submitInvite() {}",
        "</script>",
        "<template>",
        "  <InviteModal>",
        "    <h2>Invite teammates</h2>",
        '    <input placeholder="Work email" data-cy="invite-email" />',
        '    <button @click="submitInvite" data-test="send-invite">Send invite</button>',
        '    <RouterLink to="/team/members">Manage team</RouterLink>',
        "  </InviteModal>",
        "</template>",
    ]

    vue_extracted = extract_from_added_lines("components/TeamInviteCard.vue", vue_lines, is_added_file=True)
    assert "TeamInviteCard" in vue_extracted["components"]
    assert "InviteModal" in vue_extracted["components"]
    assert "Invite teammates" in vue_extracted["labels"]
    assert "Work email" in vue_extracted["labels"]
    assert "Send invite" in vue_extracted["labels"]
    assert "Manage team" in vue_extracted["labels"]
    assert "invite-email" in vue_extracted["test_ids"]
    assert "send-invite" in vue_extracted["test_ids"]
    assert "@click=submitInvite" in vue_extracted["handlers"]
    assert "/team/members" in vue_extracted["routes"]

    svelte_lines = [
        '<button on:click={handleExport} data-testid="export-csv">Export CSV</button>',
    ]
    svelte_extracted = extract_from_added_lines("components/ExportBar.svelte", svelte_lines, is_added_file=True)
    assert "ExportBar" in svelte_extracted["components"]
    assert "Export CSV" in svelte_extracted["labels"]
    assert "export-csv" in svelte_extracted["test_ids"]
    assert "on:click=handleExport" in svelte_extracted["handlers"]


def test_extract_routes_from_backend_and_router_lines() -> None:
    lines = [
        '{ path: "/dashboard/analytics", element: <Analytics /> }',
        '<Route path="/settings/security" element={<Security />} />',
        'navigate("/onboarding/welcome")',
        'path("reports/monthly/", views.monthly_report, name="monthly")',
        '@app.get("/health")',
        '@router.get("/api/plans")',
        '@app.route("/billing/portal")',
        'href="/static/app.css"',
        'href="/images/hero.png"',
        'href="mailto:support@example.com"',
        'href="#pricing"',
    ]
    routes = extract_routes_from_lines(lines)
    assert routes == [
        "/dashboard/analytics",
        "/settings/security",
        "/onboarding/welcome",
        "/reports/monthly/",
        "/health",
        "/api/plans",
        "/billing/portal",
    ]


def test_parse_numstat_and_name_status_with_binary_and_renames() -> None:
    name_status_out = (
        "A\x00app/new.tsx\x00"
        "M\x00app/page.tsx\x00"
        "D\x00app/old.tsx\x00"
        "R100\x00app/a.tsx\x00app/b.tsx\x00"
    )
    statuses = parse_name_status(name_status_out)
    assert statuses == {
        "app/new.tsx": "added",
        "app/page.tsx": "modified",
        "app/old.tsx": "deleted",
        "app/b.tsx": "renamed",
    }

    numstat_out = (
        "12\t0\tapp/new.tsx\x00"
        "-\t-\tassets/binfile\x00"
        "3\t1\t\x00app/a.tsx\x00app/b.tsx\x00"
    )
    stats = parse_numstat(numstat_out)
    assert stats == {
        "app/new.tsx": (12, 0),
        "assets/binfile": (0, 0),
        "app/b.tsx": (3, 1),
    }


def test_build_excerpt_filters_imports_long_styles_and_caps_size() -> None:
    files = [
        ChangedFile(path="components/Card.tsx", status="added", is_ui=True, is_route=False, added=10),
        ChangedFile(path="app/billing/page.tsx", status="added", is_ui=True, is_route=True, added=5),
    ]
    long_tailwind = '      className="' + ("flex items-center justify-between px-4 py-2 " * 6) + '"'
    added_lines = {
        "app/billing/page.tsx": [
            'import React from "react";',
            "",
            "",
            "export default function Billing() {",
            long_tailwind,
            "  return <button>Upgrade plan</button>;",
            "}",
        ],
        "components/Card.tsx": [
            "export function Card() {",
            "  return <div>Plan details</div>;",
            "}",
        ],
    }

    excerpt = build_excerpt(files, added_lines, max_excerpt_chars=3500)
    assert excerpt.startswith("--- app/billing/page.tsx\n")
    assert "import React" not in excerpt
    assert long_tailwind not in excerpt
    assert "Upgrade plan" in excerpt
    assert "--- components/Card.tsx" in excerpt


def _run_git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True)


@pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")
def test_analyze_diff_integration(tmp_path: Path) -> None:
    _run_git(tmp_path, "init", "-b", "main")
    _run_git(tmp_path, "config", "user.email", "test@example.com")
    _run_git(tmp_path, "config", "user.name", "Test User")

    # Initial commit on main
    (tmp_path / "app").mkdir(parents=True)
    (tmp_path / "app" / "page.tsx").write_text(
        "export default function HomePage() {\n  return <main><h1>Welcome</h1></main>;\n}\n",
        encoding="utf-8",
    )
    (tmp_path / "package-lock.json").write_text('{"name": "demo", "lockfileVersion": 3}\n', encoding="utf-8")
    _run_git(tmp_path, "add", ".")
    _run_git(tmp_path, "commit", "-m", "initial commit")

    # Feature branch with committed + uncommitted + untracked changes
    _run_git(tmp_path, "checkout", "-b", "feature")

    billing_dir = tmp_path / "app" / "settings" / "billing"
    billing_dir.mkdir(parents=True)
    (billing_dir / "page.tsx").write_text(
        "\n".join(
            [
                'import { PlanBadge } from "../../../components/PlanBadge";',
                "",
                "export default function BillingPage() {",
                "  const handleUpgrade = () => {};",
                "  return (",
                "    <section>",
                "      <h1>Billing</h1>",
                "      <PlanBadge />",
                '      <input placeholder="Card number" />',
                '      <button data-testid="upgrade-btn" onClick={handleUpgrade}>Upgrade plan</button>',
                "    </section>",
                "  );",
                "}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    (tmp_path / "app" / "page.tsx").write_text(
        "\n".join(
            [
                "export default function HomePage() {",
                "  return (",
                "    <main>",
                "      <h1>Welcome</h1>",
                '      <a href="/settings/billing">Go to billing</a>',
                "    </main>",
                "  );",
                "}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    (tmp_path / "package-lock.json").write_text(
        '{"name": "demo", "lockfileVersion": 3, "packages": {"a": {}}}\n',
        encoding="utf-8",
    )

    _run_git(tmp_path, "add", "app/settings/billing/page.tsx", "app/page.tsx", "package-lock.json")
    _run_git(tmp_path, "commit", "-m", "add billing page")

    # Untracked component + binary file (to verify resilience)
    (tmp_path / "components").mkdir(parents=True)
    (tmp_path / "components" / "PlanBadge.tsx").write_text(
        "export function PlanBadge() {\n  return <span>Pro plan</span>;\n}\n",
        encoding="utf-8",
    )
    (tmp_path / "raw_blob").write_bytes(b"\x00\xff\xfe\xfdnot-utf8")

    summary = analyze_diff(tmp_path, "main")

    assert summary.base_ref == "main"
    assert summary.head_ref == "feature +worktree"
    assert summary.routes[0] == "/settings/billing"
    assert "/" in summary.routes

    assert "Upgrade plan" in summary.labels
    assert "Card number" in summary.labels
    assert "upgrade-btn" in summary.test_ids
    assert "onClick=handleUpgrade" in summary.handlers
    assert "PlanBadge" in summary.components

    file_paths = {f.path for f in summary.files}
    assert "package-lock.json" not in file_paths
    assert "app/settings/billing/page.tsx" in file_paths
    assert "components/PlanBadge.tsx" in file_paths

    prompt = summary.to_prompt()
    assert len(prompt) <= 6000
    assert "/settings/billing" in prompt

    # Verify include_worktree=False excludes untracked files and +worktree suffix
    committed_only = analyze_diff(tmp_path, "main", include_worktree=False)
    assert committed_only.head_ref == "feature"
    assert "components/PlanBadge.tsx" not in {f.path for f in committed_only.files}

    # Verify missing base raises GitError with helpful message
    with pytest.raises(GitError, match="Could not resolve base ref"):
        analyze_diff(tmp_path, "nonexistent-branch")


@pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")
def test_analyze_diff_master_fallback_and_rename(tmp_path: Path) -> None:
    _run_git(tmp_path, "init", "-b", "master")
    _run_git(tmp_path, "config", "user.email", "test@example.com")
    _run_git(tmp_path, "config", "user.name", "Test User")

    (tmp_path / "components").mkdir(parents=True)
    (tmp_path / "components" / "OldBtn.tsx").write_text(
        "export function OldBtn() {\n  return <button>Save</button>;\n}\n",
        encoding="utf-8",
    )
    _run_git(tmp_path, "add", ".")
    _run_git(tmp_path, "commit", "-m", "init on master")

    _run_git(tmp_path, "checkout", "-b", "rename-branch")
    _run_git(tmp_path, "mv", "components/OldBtn.tsx", "components/NewBtn.tsx")
    _run_git(tmp_path, "commit", "-m", "rename component")

    # Requesting default "main" should fall back to "master"
    summary = analyze_diff(tmp_path, "main")
    assert summary.base_ref == "master"
    assert summary.head_ref == "rename-branch"
    assert len(summary.files) == 1
    assert summary.files[0].path == "components/NewBtn.tsx"
    assert summary.files[0].status == "renamed"
    assert summary.files[0].is_ui is True


# --- framework detection & SPA router mapping (regression: Vite + React Router app, src/pages/*.tsx) -------------
def test_vite_react_router_pages_are_not_file_routes() -> None:
    from pullreel.diff_analyzer import extract_routes_from_path

    assert extract_routes_from_path("src/pages/AdversairesPage.tsx", set()) is None
    assert extract_routes_from_path("src/pages/AdversairesPage.tsx", None) == "/AdversairesPage"  # unknown stack
    assert extract_routes_from_path("pages/settings.tsx", {"next"}) == "/settings"
    assert extract_routes_from_path("pages/settings.vue", {"next"}) is None


def test_map_router_components_handles_jsx_and_objects() -> None:
    from pullreel.diff_analyzer import map_router_components

    jsx = """
      <Route path="/" element={<Navigate to="/welcome" replace />} />
      <Route path="/adversaires" element={<AdversairesPage />} />
      <Route path="/statequipe/*" element={<StatEquipePage onNavigate={() => {}} />} />
      <Route element={<SettingsPage />} path="settings" />
      <Route path="*" element={<NotFound />} />
    """
    objects = """createBrowserRouter([{ path: "/billing", element: <BillingPage /> },
                                     { path: "/team", Component: TeamPage }])"""
    mapping = map_router_components([jsx, objects])
    assert mapping == {
        "AdversairesPage": "/adversaires",
        "StatEquipePage": "/statequipe",
        "SettingsPage": "/settings",
        "BillingPage": "/billing",
        "TeamPage": "/team",
    }


@pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")
def test_analyze_diff_maps_changed_page_to_react_router_path(tmp_path: Path) -> None:
    import json as _json
    import subprocess as _sp

    def run(*args: str) -> None:
        _sp.run(["git", "-C", str(tmp_path), *args], check=True, capture_output=True)

    run("init", "-b", "main")
    run("config", "user.email", "test@example.com")
    run("config", "user.name", "Test")
    (tmp_path / "package.json").write_text(_json.dumps({"dependencies": {"react-router-dom": "^6", "vite": "^5"}}))
    (tmp_path / "src" / "pages").mkdir(parents=True)
    (tmp_path / "src" / "App.tsx").write_text(
        '<Routes>\n  <Route path="/adversaires" element={<AdversairesPage />} />\n</Routes>\n'
    )
    page = tmp_path / "src" / "pages" / "AdversairesPage.tsx"
    page.write_text("export function AdversairesPage() {\n  return <h1>Adversaires</h1>;\n}\n")
    run("add", "-A")
    run("commit", "-m", "init")
    run("checkout", "-b", "feature")
    page.write_text(page.read_text() + 'export const Filter = () => <button>Leur jeu</button>;\n')

    summary = analyze_diff(tmp_path, "main")
    assert summary.routes[:1] == ["/adversaires"]
    assert "/AdversairesPage" not in summary.routes


# --- commits, project context, import resolution & feature candidates -------------------------------------------
def test_parse_import_specifiers_tsx_header() -> None:
    from pullreel.diff_analyzer import parse_import_specifiers

    tsx = """
    import React, { useState, lazy } from "react";
    import CarteZones from "../components/CarteZones";
    import {
      ZoneLegend,
      FilterBar as Filters,
    } from '@/components/zones';
    import type { MatchStats } from "@/types/stats";
    import "./AdversairesPage.css";
    const DetailModal = lazy(() => import('./DetailModal'));
    export { helper } from "./helper";
    """
    specs = parse_import_specifiers(tsx)
    assert specs == [
        "react",
        "../components/CarteZones",
        "@/components/zones",
        "@/types/stats",
        "./AdversairesPage.css",
        "./DetailModal",
        "./helper",
    ]


def test_resolve_import_variants() -> None:
    from pullreel.diff_analyzer import resolve_import

    known = {
        "src/pages/AdversairesPage.tsx",
        "src/components/CarteZones.tsx",
        "src/components/zones/index.tsx",
        "src/styles/theme.css",
        "src/utils/format.ts",
    }
    from_file = "src/pages/AdversairesPage.tsx"

    # Relative with explicit extension
    assert resolve_import("../components/CarteZones.tsx", from_file, known) == "src/components/CarteZones.tsx"
    assert resolve_import("../styles/theme.css", from_file, known) == "src/styles/theme.css"
    # Relative without extension
    assert resolve_import("../components/CarteZones", from_file, known) == "src/components/CarteZones.tsx"
    # Directory index file
    assert resolve_import("../components/zones", from_file, known) == "src/components/zones/index.tsx"
    # @/ alias -> src/
    assert resolve_import("@/utils/format", from_file, known) == "src/utils/format.ts"
    assert resolve_import("@/components/zones", from_file, known) == "src/components/zones/index.tsx"
    # External package and missing file -> None
    assert resolve_import("react", from_file, known) is None
    assert resolve_import("../components/Missing", from_file, known) is None


def test_read_project_context_package_json_and_readme(tmp_path: Path) -> None:
    import json as _json

    from pullreel.diff_analyzer import read_project_context

    (tmp_path / "package.json").write_text(
        _json.dumps(
            {
                "name": "handball-tracker",
                "description": "Live scouting and shot zone analytics for handball coaches",
            }
        ),
        encoding="utf-8",
    )
    long_tail = "Detailed tactical breakdowns for every match. " * 25
    (tmp_path / "README.md").write_text(
        "\n".join(
            [
                "# Handball Tracker",
                "[![CI](https://img.shields.io/badge/ci-passing-brightgreen)](https://example.com)",
                '<p align="center"><img src="banner.png" alt="banner" /></p>',
                "",
                "<div>Explore <strong>shot zones</strong> and opponent tendencies.</div>",
                "",
                "```bash",
                "npm run dev",
                "```",
                "",
                long_tail,
            ]
        ),
        encoding="utf-8",
    )

    ctx = read_project_context(tmp_path)
    assert "handball-tracker" in ctx
    assert "Live scouting and shot zone analytics" in ctx
    assert "Explore shot zones and opponent tendencies." in ctx
    assert "shields.io" not in ctx
    assert "![" not in ctx
    assert "<div" not in ctx
    assert "<strong" not in ctx
    assert "<img" not in ctx
    assert "npm run dev" not in ctx
    assert "#" not in ctx
    assert len(ctx) <= 600


def test_read_project_context_pyproject_fallback_and_never_raises(tmp_path: Path) -> None:
    from pullreel.diff_analyzer import read_project_context

    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "pullreel"\ndescription = "Cinematic PR demos"\n',
        encoding="utf-8",
    )
    (tmp_path / "readme").write_text("Record browser demos from git diffs.\n", encoding="utf-8")

    ctx = read_project_context(tmp_path)
    assert ctx == "pullreel - Cinematic PR demos - Record browser demos from git diffs."
    assert read_project_context(tmp_path / "does-not-exist") == ""


def test_score_feature_and_depth2_grouping() -> None:
    from pullreel.diff_analyzer import build_feature_candidates, score_feature

    static_score = score_feature(added=15, new_files=1, label_count=3, commit_count=1, route="/adversaires")
    dynamic_score = score_feature(added=15, new_files=1, label_count=3, commit_count=1, route="/adversaires/:id")
    no_route_score = score_feature(added=15, new_files=1, label_count=3, commit_count=1, route=None)
    assert static_score == round(4.0 + 1.5 + 1.2 + 1.0 + 1.5 + 1.0, 2)
    assert dynamic_score == round(static_score - 1.0, 2)
    assert no_route_score == round(static_score - 2.5, 2)

    files_on_disk = {
        "src/pages/AdversairesPage.tsx": 'import { CarteZones } from "../components/CarteZones";\n',
        "src/components/CarteZones.tsx": 'import { ZoneLegend } from "./ZoneLegend";\n',
        "src/components/ZoneLegend.tsx": "export function ZoneLegend() { return <span>Legende</span>; }\n",
    }
    changed = [
        ChangedFile(path="src/components/ZoneLegend.tsx", status="added", is_ui=True, added=12),
        ChangedFile(path="src/styles.css", status="modified", is_ui=True, added=5),
    ]
    candidates = build_feature_candidates(
        changed,
        {"src/components/ZoneLegend.tsx": ["<span>Legende</span>"]},
        frameworks=set(),
        router_map={"AdversairesPage": "/adversaires"},
        known_files=set(files_on_disk) | {"src/styles.css"},
        read_file=files_on_disk.get,
        commit_entries=[("Add legend", {"src/components/ZoneLegend.tsx"})],
    )
    assert len(candidates) == 1
    assert candidates[0].route == "/adversaires"
    assert candidates[0].title == "AdversairesPage"
    assert candidates[0].files == ["src/components/ZoneLegend.tsx"]
    assert candidates[0].labels == ["Legende"]


@pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")
def test_analyze_diff_commits_project_and_features_integration(tmp_path: Path) -> None:
    import json as _json

    _run_git(tmp_path, "init", "-b", "main")
    _run_git(tmp_path, "config", "user.email", "test@example.com")
    _run_git(tmp_path, "config", "user.name", "Test User")

    pkg_desc = "Tactical handball match analysis dashboard"
    (tmp_path / "package.json").write_text(
        _json.dumps(
            {
                "name": "handball-analytics",
                "description": pkg_desc,
                "dependencies": {"react-router-dom": "^6.22.0", "vite": "^5.0.0"},
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "README.md").write_text(
        "# Handball Analytics\n\nVisualize opponent zones and player metrics.\n",
        encoding="utf-8",
    )

    (tmp_path / "src" / "pages").mkdir(parents=True)
    (tmp_path / "src" / "components").mkdir(parents=True)
    (tmp_path / "src" / "App.tsx").write_text(
        "\n".join(
            [
                "import { Routes, Route } from 'react-router-dom';",
                "import { AdversairesPage } from './pages/AdversairesPage';",
                "import { JoueursPage } from './pages/JoueursPage';",
                "export function App() {",
                "  return (",
                "    <Routes>",
                '      <Route path="/adversaires" element={<AdversairesPage />} />',
                '      <Route path="/joueurs" element={<JoueursPage />} />',
                "    </Routes>",
                "  );",
                "}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / "src" / "pages" / "AdversairesPage.tsx").write_text(
        "\n".join(
            [
                'import { CarteZones } from "../components/CarteZones";',
                "export function AdversairesPage() {",
                "  return <main><h1>Adversaires</h1></main>;",
                "}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / "src" / "pages" / "JoueursPage.tsx").write_text(
        "export function JoueursPage() {\n  return <main><h1>Joueurs</h1></main>;\n}\n",
        encoding="utf-8",
    )
    (tmp_path / "src" / "styles.css").write_text("body { margin: 0; }\n", encoding="utf-8")

    _run_git(tmp_path, "add", "-A")
    _run_git(tmp_path, "commit", "-m", "initial commit")

    _run_git(tmp_path, "checkout", "-b", "feature")

    # Commit 1: add CarteZones.tsx and update AdversairesPage.tsx
    (tmp_path / "src" / "components" / "CarteZones.tsx").write_text(
        "\n".join(
            [
                "export function CarteZones() {",
                "  return (",
                "    <section>",
                "      <h2>Zones chaudes</h2>",
                "      <button>Leur jeu</button>",
                "    </section>",
                "  );",
                "}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / "src" / "pages" / "AdversairesPage.tsx").write_text(
        "\n".join(
            [
                'import { CarteZones } from "../components/CarteZones";',
                "export function AdversairesPage() {",
                "  return (",
                "    <main>",
                "      <h1>Adversaires</h1>",
                "      <CarteZones />",
                "    </main>",
                "  );",
                "}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    _run_git(tmp_path, "add", "src/components/CarteZones.tsx", "src/pages/AdversairesPage.tsx")
    _run_git(tmp_path, "commit", "-m", "Page Adversaires avec zones chaudes")

    # Commit 2: 1-line edit on JoueursPage.tsx
    (tmp_path / "src" / "pages" / "JoueursPage.tsx").write_text(
        "export function JoueursPage() {\n  return <main><h1>Joueurs actifs</h1></main>;\n}\n",
        encoding="utf-8",
    )
    _run_git(tmp_path, "add", "src/pages/JoueursPage.tsx")
    _run_git(tmp_path, "commit", "-m", "Fix typo")

    # Uncommitted change to src/styles.css
    (tmp_path / "src" / "styles.css").write_text("body { margin: 0; padding: 0; }\n", encoding="utf-8")

    summary = analyze_diff(tmp_path, "main")

    assert summary.commits == ["Fix typo", "Page Adversaires avec zones chaudes"]
    assert pkg_desc in summary.project

    assert summary.features
    top = summary.features[0]
    assert top.route == "/adversaires"
    assert top.title == "AdversairesPage"
    assert "src/components/CarteZones.tsx" in top.files
    assert "Zones chaudes" in top.labels
    assert "Leur jeu" in top.labels
    assert "Page Adversaires avec zones chaudes" in top.commits

    joueurs = [f for f in summary.features if f.route == "/joueurs"]
    assert len(joueurs) == 1
    assert joueurs[0].score < top.score

    assert not any(
        all(p.endswith((".css", ".scss", ".sass", ".less")) for p in f.files)
        for f in summary.features
    )

    prompt = summary.to_prompt()
    assert "Feature candidates" in prompt
    assert len(prompt) <= 6000
