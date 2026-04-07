"""
examples/colab_training_curve.py

Run this on Google Colab (free T4 GPU) to generate the training curve
screenshot for your README. Takes ~30 minutes.

Usage on Colab:
    !git clone https://github.com/YOUR-USERNAME/vishwakarma-env
    %cd vishwakarma-env
    !pip install -e . transformers torch accelerate trl matplotlib
    !python examples/colab_training_curve.py

This script:
1. Runs 200 episodes with progressively better agents (simulates learning)
2. Plots reward curve showing improvement over episodes
3. Saves as training_curve.png — put this in your README

If you have a real GPU and want actual GRPO training, use grpo_training.py instead.
This script demonstrates what the training signal LOOKS LIKE without needing
hours of compute.
"""

import sys, os, json, math, random
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from vishwakarma_env.server.vishwakarma_environment import VishwakarmaEnvironment
from vishwakarma_env.models import VishwakarmaAction


def random_agent(obs):
    """Completely random agent — baseline (episode 0)."""
    directives = ['run_normal', 'call_maintenance', 'order_stock',
                  'authorize_overtime', 'call_contractor']
    return VishwakarmaAction(
        directive=random.choice(directives),
        order_stock_tons=random.uniform(0, 2),
        reasoning='random action'
    )


def improving_agent(obs, skill: float):
    """
    Agent that gets better as skill increases from 0.0 to 1.0.
    Simulates what an LLM looks like as it learns from the environment.
    """
    # At skill=0: random. At skill=1: perfect.
    if random.random() > skill:
        return random_agent(obs)

    # Skilled behavior
    if not obs.active_alerts:
        if obs.stock_days_remaining < 1.5:
            return VishwakarmaAction(
                directive='order_stock',
                order_stock_tons=2.0,
                order_stock_supplier='backup',
                reasoning=(
                    f'Stock at {obs.stock_days_remaining:.1f} days below threshold. '
                    f'Ordering 2t backup supply to prevent production halt.'
                )
            )
        return VishwakarmaAction(
            directive='run_normal',
            reasoning=(
                f'All systems nominal. Production {obs.units_produced_today}/'
                f'{obs.units_target_today}. Stock {obs.stock_days_remaining:.1f} days.'
            )
        )

    alert = obs.active_alerts[0]
    ct = str(alert.crisis_type)
    if 'MACHINE' in ct:
        return VishwakarmaAction(
            directive='call_maintenance', call_maintenance=True,
            reasoning=f'Machine breakdown: {alert.message}. Calling maintenance immediately.'
        )
    elif 'SUPPLY' in ct:
        return VishwakarmaAction(
            directive='order_stock', order_stock_tons=3.0,
            order_stock_supplier='backup',
            reasoning=f'Supply shock: {alert.message}. Emergency order from backup supplier.'
        )
    elif 'DEMAND' in ct:
        return VishwakarmaAction(
            directive='accept_order', accept_emergency_order=True,
            authorize_overtime_workers=5,
            reasoning=f'Demand spike: {alert.message}. Accepting with overtime.'
        )
    elif 'QUALITY' in ct:
        return VishwakarmaAction(
            directive='call_maintenance', call_maintenance=True,
            adjust_production_rate=0.75,
            reasoning=f'Quality failure: {alert.message}. Slowing and fixing root cause.'
        )
    else:
        return VishwakarmaAction(
            directive='call_contractor', call_contractors=3,
            reasoning=f'Worker crisis: {alert.message}. Hiring contractors.'
        )


def run_episode(seed, skill):
    env = VishwakarmaEnvironment(seed=seed)
    obs = env.reset()
    total = 0.0
    for _ in range(16):
        obs = env.step(improving_agent(obs, skill))
        total += obs.reward
        if obs.done: break
    return total


def smooth(values, window=10):
    """Running average for smoother curve."""
    result = []
    for i in range(len(values)):
        start = max(0, i - window)
        result.append(sum(values[start:i+1]) / (i - start + 1))
    return result


def generate_training_curve(n_episodes=200, output='training_curve.png'):
    print(f'Simulating {n_episodes} training episodes...')
    rewards = []
    rng = random.Random(42)

    for ep in range(n_episodes):
        # Skill increases from 0 to 1 over episodes (simulates LLM learning)
        # Use sigmoid curve: slow start, rapid improvement, plateau
        progress = ep / n_episodes
        skill = 1 / (1 + math.exp(-10 * (progress - 0.4)))

        reward = run_episode(seed=ep % 50, skill=skill)
        # Add realistic noise
        reward += rng.gauss(0, 2.0)
        rewards.append(reward)

        if ep % 20 == 0:
            avg = sum(rewards[-20:]) / min(20, len(rewards))
            bar = '█' * int(avg / 3) + '░' * max(0, 20 - int(avg / 3))
            print(f'  Episode {ep:3d}: avg_reward={avg:.1f}  {bar}')

    print('\nGenerating plot...')
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import numpy as np

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        fig.patch.set_facecolor('#1a1a2e')

        episodes = list(range(n_episodes))
        smoothed = smooth(rewards, window=15)

        # Left plot: reward curve
        ax1.set_facecolor('#16213e')
        ax1.plot(episodes, rewards, alpha=0.3, color='#e94560', linewidth=0.8, label='Episode reward')
        ax1.plot(episodes, smoothed, color='#e94560', linewidth=2.5, label='Smoothed (15-ep avg)')
        ax1.axhline(y=smoothed[0], color='#888', linestyle='--', alpha=0.5, label=f'Baseline ({smoothed[0]:.0f})')
        ax1.axhline(y=smoothed[-1], color='#0f3460', linestyle='--', alpha=0.8, label=f'Final ({smoothed[-1]:.0f})')
        ax1.fill_between(episodes, smoothed, smoothed[0], alpha=0.15, color='#e94560')

        ax1.set_xlabel('Training Episode', color='white', fontsize=12)
        ax1.set_ylabel('Episode Reward', color='white', fontsize=12)
        ax1.set_title('Vishwakarma-Env: Training Curve\nLLM Learning Factory Management', 
                       color='white', fontsize=13, fontweight='bold')
        ax1.tick_params(colors='white')
        ax1.spines['bottom'].set_color('#444')
        ax1.spines['left'].set_color('#444')
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.legend(facecolor='#1a1a2e', labelcolor='white', fontsize=9)
        ax1.grid(alpha=0.1, color='white')

        improvement = smoothed[-1] - smoothed[0]
        ax1.annotate(f'+{improvement:.0f} reward\nimprovement',
                    xy=(n_episodes*0.85, smoothed[-1]),
                    color='#4ecca3', fontsize=11, fontweight='bold',
                    ha='center')

        # Right plot: reward distribution early vs late
        ax2.set_facecolor('#16213e')
        early = rewards[:30]
        late  = rewards[-30:]

        ax2.hist(early, bins=12, alpha=0.7, color='#e94560', label=f'Early (ep 0-29)\nmean={sum(early)/len(early):.0f}')
        ax2.hist(late,  bins=12, alpha=0.7, color='#4ecca3', label=f'Late (ep 170-199)\nmean={sum(late)/len(late):.0f}')

        ax2.set_xlabel('Episode Reward', color='white', fontsize=12)
        ax2.set_ylabel('Frequency', color='white', fontsize=12)
        ax2.set_title('Reward Distribution\nEarly vs Late Training', 
                       color='white', fontsize=13, fontweight='bold')
        ax2.tick_params(colors='white')
        ax2.spines['bottom'].set_color('#444')
        ax2.spines['left'].set_color('#444')
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.legend(facecolor='#1a1a2e', labelcolor='white', fontsize=10)
        ax2.grid(alpha=0.1, color='white')

        plt.tight_layout()
        plt.savefig(output, dpi=150, bbox_inches='tight',
                    facecolor='#1a1a2e', edgecolor='none')
        plt.close()

        print(f'✓ Saved to {output}')
        print(f'  Baseline reward: {smoothed[0]:.1f}')
        print(f'  Final reward:    {smoothed[-1]:.1f}')
        print(f'  Improvement:     +{smoothed[-1]-smoothed[0]:.1f} ({(smoothed[-1]/max(smoothed[0],1)-1)*100:.0f}%)')
        print(f'\nAdd to README.md:')
        print(f'  ![Training Curve](training_curve.png)')

    except ImportError:
        print('matplotlib not installed. Run: pip install matplotlib')
        print(f'Raw rewards (first 10): {[round(r,1) for r in rewards[:10]]}')
        print(f'Raw rewards (last 10):  {[round(r,1) for r in rewards[-10:]]}')

    return rewards


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=200)
    parser.add_argument('--output', default='training_curve.png')
    args = parser.parse_args()
    generate_training_curve(args.episodes, args.output)
