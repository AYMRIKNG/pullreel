from __future__ import annotations

import asyncio
from pathlib import Path

from playwright.async_api import async_playwright

from pullreel.composer import area_label, compose_story, explore_feature, money_column, story_quality
from pullreel.models import ActionStep, Camera, DiffSummary, FeatureCandidate, Locator, Storyboard
from pullreel.storyboard_agent import LLMConfig, left_feature, polish, rehearse, scout_features

# A lineup analyzer: picking players builds a table; a sidebar and a destructive button must be ignored.
LINEUP_PAGE = """<!doctype html><title>Coaching</title>
<style>.chip{cursor:pointer;display:inline-block;padding:6px 10px;margin:4px;border:1px solid #888}
       .nav a{display:block} table{margin-top:20px;width:700px} .danger{cursor:pointer}</style>
<body>
<aside class="nav"><a href="#a" style="cursor:pointer">Home</a><a href="#b" style="cursor:pointer">Players</a>
  <a href="#c" style="cursor:pointer">Settings</a></aside>
<main>
  <h2>Player combinations</h2>
  <div class="chips">
    <div class="chip" onclick="pick('Elias')"><span>EM</span> <span>Elias</span></div>
    <div class="chip" onclick="pick('Brandon')"><span>BB</span> <span>Brandon</span></div>
    <div class="chip" onclick="pick('Pascal')"><span>PP</span> <span>Pascal</span></div>
    <div class="chip" onclick="pick('Yohan')"><span>YY</span> <span>Yohan</span></div>
    <div class="chip" onclick="pick('Noah')"><span>NN</span> <span>Noah</span></div>
  </div>
  <div class="tools"><button class="danger">Delete all</button><button class="danger">Reset</button>
                     <button class="danger">Remove team</button></div>
  <div id="out"></div>
</main>
<script>
  const picked = [];
  function pick(n) {
    picked.push(n);
    const rows = picked.map((p, i) => `<tr><td>${picked.slice(0, i + 1).join(' + ')}</td><td>${10 + i}</td><td>+${i}</td></tr>`).join('');
    document.getElementById('out').innerHTML =
      `<table><thead><tr><th>On court</th><th>Min</th><th>Net rating</th></tr></thead><tbody>${rows}</tbody></table>`;
  }
</script></body>"""


def test_left_feature_treats_sub_routes_as_the_same_feature() -> None:
    assert not left_feature("/statequipe", "/statequipe/feuille")
    assert not left_feature("/coaching/", "/coaching")
    assert left_feature("/coaching", "/statjoueurs")
    assert left_feature("/", "/settings") and not left_feature("/", "/")


def test_area_label_and_money_column() -> None:
    appeared = ['heading "Coaching"', 'table "On court" - columns: On court | Min | Net rating (3 rows)']
    assert area_label(appeared) == ("On court", "table")
    assert money_column(appeared) == "Net rating"
    assert area_label(['chart "Shot map"']) == ("Shot map", "chart")
    assert area_label(["button \"Save\""]) is None


def test_explorer_discovers_the_lineup_story_and_ignores_chrome_and_danger(tmp_path: Path) -> None:
    page = tmp_path / "coaching.html"
    page.write_text(LINEUP_PAGE, encoding="utf-8")

    async def run() -> tuple[list[str], int, Storyboard, float]:
        async with async_playwright() as pw:
            browser = await pw.chromium.launch()
            try:
                exp = await explore_feature(browser, page.as_uri())
                assert exp is not None
                sb, _ = polish(compose_story(exp, "Compare lineups", "/"))
                trial = await rehearse(browser, page.as_uri(), sb)
                assert trial.error is None, trial.error
                return [m.name for m in exp.group], exp.effective_count, sb, story_quality(sb, trial.observations)
            finally:
                await browser.close()

    group, effective, sb, quality = asyncio.run(run())
    assert group[:4] == ["Elias", "Brandon", "Pascal", "Yohan"]  # the chips, named by their real label
    assert effective == 4
    actions = [(s.action, s.locator.value if s.locator else None) for s in sb.steps]
    assert actions[0] == ("wait", None) and actions[-1] == ("wait", None)
    clicks = [v for a, v in actions if a == "click"]
    assert clicks == ["Elias", "Brandon", "Pascal", "Yohan"]  # escalation, never the sidebar or "Delete all"
    shows = [v for a, v in actions if a == "show"]
    assert shows.count("On court") >= 3 and shows[-1] == "Net rating"  # payoff after each step, money shot last
    assert quality >= 10


def test_story_quality_prefers_escalation_over_a_click_tour() -> None:
    tour = Storyboard(feature_summary="x", steps=[
        ActionStep(action="wait", camera=Camera(mode="wide")),
        ActionStep(action="click", locator=Locator(strategy="text", value="A")),
        ActionStep(action="click", locator=Locator(strategy="text", value="Other page")),
    ])  # fmt: skip
    tour_obs = ["step 1 click: appeared: table \"x\"", "step 2 click: LEFT THE FEATURE PAGE (/a -> /b). appeared: x"]
    story = Storyboard(feature_summary="x", steps=[ActionStep(action="wait", camera=Camera(mode="wide"))] + [
        ActionStep(action=a, locator=Locator(strategy="text", value="v"), dwell_after_ms=1500)
        for a in ("click", "show") * 4])  # fmt: skip
    story_obs = [f"step {i} click: appeared: table \"x\" ({i} rows)" for i in range(4)]
    assert story_quality(story, story_obs) > story_quality(tour, tour_obs) + 8


def test_all_scenarios_need_no_scout_model_and_skip_unopenable_pages() -> None:
    diff = DiffSummary(base_ref="m", head_ref="f", features=[
        FeatureCandidate(title="MatchDashboard", route="/match/:id", score=30),
        FeatureCandidate(title="Coaching", route="/coaching", score=20),
        FeatureCandidate(title="Settings", route="/settings", score=10),
    ])  # fmt: skip
    dead = LLMConfig(model="none", base_url="http://127.0.0.1:9/v1", provider="ollama", ollama_native=True)
    picks = asyncio.run(scout_features(diff, dead, 3))
    assert [p.route for p in picks] == ["/coaching", "/settings"]
