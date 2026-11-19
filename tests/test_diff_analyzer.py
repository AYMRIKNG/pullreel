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
