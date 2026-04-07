"""
tests/test_vishwakarma.py
Basic tests for the Vishwakarma environment.
Run: pytest tests/ -v
"""

import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from vishwakarma_env.server.vishwakarma_environment import VishwakarmaEnvironment
from vishwakarma_env.models import VishwakarmaAction, CrisisType


@pytest.fixture
def env():
    return VishwakarmaEnvironment(factory_id="auto_components_pune", seed=42)


def test_reset_returns_valid_observation(env):
    obs = env.reset()
    assert obs is not None
    assert obs.workers_present > 0
    assert obs.units_target_today > 0
    assert obs.stock_tons > 0
    assert obs.done is False
    assert obs.step == 0


def test_step_increments_step_count(env):
    env.reset()
    obs = env.step(VishwakarmaAction(directive="run_normal"))
    assert obs.step == 1


def test_full_episode_completes(env):
    obs = env.reset()
    steps = 0
    while not obs.done:
        obs = env.step(VishwakarmaAction(directive="run_normal"))
        steps += 1
        assert steps <= 20, "Episode should complete within 16 steps"
    assert obs.done is True
    assert steps == 16


def test_production_increases_each_step(env):
    obs = env.reset()
    prev_units = 0
    for _ in range(5):
        obs = env.step(VishwakarmaAction(directive="run_normal"))
        assert obs.units_produced_today >= prev_units
        prev_units = obs.units_produced_today


def test_overtime_increases_cost(env):
    env.reset()
    obs_normal = env.step(VishwakarmaAction(directive="run_normal"))
    normal_cost = obs_normal.cost_today_INR

    env.reset()
    obs_overtime = env.step(VishwakarmaAction(
        directive="authorize_overtime",
        authorize_overtime_workers=8
    ))
    assert obs_overtime.cost_today_INR > normal_cost


def test_stock_decreases_over_time(env):
    obs = env.reset()
    initial_stock = obs.stock_tons
    for _ in range(4):
        obs = env.step(VishwakarmaAction(directive="run_normal"))
    assert obs.stock_tons < initial_stock


def test_maintenance_call_reduces_repair_eta(env):
    obs = env.reset()
    # Manually take a machine offline
    s = env.state
    s.machine_online["cnc_mill"] = False
    s.maintenance_eta["cnc_mill"] = 4

    obs = env.step(VishwakarmaAction(
        directive="call_maintenance",
        call_maintenance=True
    ))
    # ETA should have decreased
    assert env.state.maintenance_eta.get("cnc_mill", 0) < 4


def test_ordering_stock_increases_stock(env):
    obs = env.reset()
    initial_stock = obs.stock_tons
    obs = env.step(VishwakarmaAction(
        directive="order_stock",
        order_stock_tons=2.0,
        order_stock_supplier="primary"
    ))
    assert obs.stock_tons > initial_stock - 0.5  # minus this step's consumption


def test_reward_is_finite(env):
    obs = env.reset()
    for _ in range(16):
        if obs.done:
            break
        obs = env.step(VishwakarmaAction(directive="run_normal"))
        assert obs.reward is not None
        assert not (obs.reward != obs.reward)  # not NaN
        assert abs(obs.reward) < 100            # sanity bound


def test_state_info_returns_metadata(env):
    env.reset()
    env.step(VishwakarmaAction(directive="run_normal"))
    info = env.state_info()
    assert "episode_id" in info
    assert info["step"] == 1
    assert info["total_steps"] == 16


def test_all_three_factories_load(env):
    for fid in ["auto_components_pune", "pharma_packaging_hyderabad", "textile_mill_surat"]:
        e = VishwakarmaEnvironment(factory_id=fid, seed=0)
        obs = e.reset()
        assert obs.units_target_today > 0, f"Factory {fid} failed to load"
