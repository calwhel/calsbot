"""Gold AI Trader Haiku→Opus funnel tests (shadow mode, no live API)."""
import inspect
import os
from unittest.mock import AsyncMock, patch

os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")

from app.gold_ai_trader.config import env_defaults
from app.gold_ai_trader.funnel import (
    SCREEN_SYSTEM_PROMPT,
    funnel_mode_from_env,
    is_false_reject,
    maybe_geometry_prefilter,
    run_funnel_pipeline,
    screen_setup,
)
from app.gold_ai_trader.scanner import Candidate


def test_funnel_mode_defaults_shadow():
    os.environ.pop("GOLD_FUNNEL_MODE", None)
    assert funnel_mode_from_env() == "shadow"
    cfg = env_defaults()
    assert cfg.funnel_mode == "shadow"
    assert cfg.screen_model == "claude-haiku-4-5"


def test_haiku_prompt_over_pass_bias():
    assert "OVER-PASS" in SCREEN_SYSTEM_PROMPT or "over-pass" in SCREEN_SYSTEM_PROMPT.lower()
    assert "When unsure" in SCREEN_SYSTEM_PROMPT


def test_haiku_system_prompt_has_cache_control():
    src = inspect.getsource(screen_setup)
    assert "cache_control" in src
    assert "ephemeral" in src


def test_opus_decide_still_has_cache_control():
    from app.gold_ai_trader import claude

    src = inspect.getsource(claude.decide)
    assert "cache_control" in src


def test_is_false_reject_haiku_skip_opus_take():
    assert is_false_reject("SKIP", "take", 80) is True
    assert is_false_reject("SKIP", "skip", 65) is True
    assert is_false_reject("SKIP", "skip", 40) is False
    assert is_false_reject("PASS", "take", 90) is False


def test_geometry_prefilter_hook_passes_through():
    cand = Candidate(
        type="liquidity_sweep",
        direction="long",
        detail="test",
        quality_atr=1.2,
        sig_key="x",
        raw={},
    )
    ok, reason = maybe_geometry_prefilter(cand, 2650.0, 2.5, rvol=1.5)
    assert ok is True
    assert reason is None


async def _run_shadow_pipeline(screen_action="PASS", opus_action="skip", conf=42):
    cand = Candidate(
        type="liquidity_sweep",
        direction="long",
        detail="sweep reclaim",
        quality_atr=1.1,
        sig_key="k",
        raw={},
    )
    cfg = env_defaults()
    cfg.funnel_mode = "shadow"
    cfg.model = "claude-opus-4-8"

    screen_out = {"screen": screen_action, "reason": "test screen"}
    screen_meta = {"cost_usd": 0.001, "tokens_in": 100, "cache_control_applied": True}
    opus_decision = {
        "action": opus_action,
        "confidence": conf,
        "rationale": "opus rationale",
    }
    opus_meta = {"cost_usd": 0.05, "tokens_in": 500}

    with patch(
        "app.gold_ai_trader.funnel.screen_setup",
        new=AsyncMock(return_value=(screen_out, screen_meta)),
    ), patch(
        "app.gold_ai_trader.funnel.decide",
        new=AsyncMock(return_value=(opus_decision, "reasoning", opus_meta)),
    ), patch(
        "app.gold_ai_trader.funnel.build_screen_context",
        new=AsyncMock(return_value="screen ctx"),
    ):
        return await run_funnel_pipeline(
            candidate=cand,
            price=2650.0,
            atr=2.5,
            rvol=1.4,
            session="london",
            full_context="full ctx",
            cfg=cfg,
            db=None,
        )


def test_shadow_mode_always_calls_opus():
    import asyncio

    result = asyncio.run(_run_shadow_pipeline(screen_action="SKIP", opus_action="take", conf=82))
    assert result.funnel_mode == "shadow"
    assert result.opus_called is True
    assert result.false_reject is True


def test_live_mode_haiku_skip_short_circuits_opus():
    import asyncio

    cand = Candidate(
        type="liquidity_sweep",
        direction="long",
        detail="sweep",
        quality_atr=1.0,
        sig_key="k",
        raw={},
    )
    cfg = env_defaults()
    cfg.funnel_mode = "live"

    with patch(
        "app.gold_ai_trader.funnel.screen_setup",
        new=AsyncMock(return_value=({"screen": "SKIP", "reason": "dead tape"}, {"cost_usd": 0.001})),
    ), patch(
        "app.gold_ai_trader.funnel.decide",
        new=AsyncMock(),
    ) as mock_decide, patch(
        "app.gold_ai_trader.funnel.build_screen_context",
        new=AsyncMock(return_value="screen ctx"),
    ):
        result = asyncio.run(
            run_funnel_pipeline(
                candidate=cand,
                price=2650.0,
                atr=2.5,
                rvol=0.5,
                session="london",
                full_context="full ctx",
                cfg=cfg,
                db=None,
            )
        )
    assert result.opus_called is False
    mock_decide.assert_not_called()
    assert result.decision["action"] == "skip"


def test_screen_setup_dry_run():
    import asyncio

    out, meta = asyncio.run(screen_setup("ctx", dry_run=True))
    assert out["screen"] == "PASS"
    assert meta.get("cache_control_applied") is True
