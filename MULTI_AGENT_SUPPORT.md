# Multi-Agent RL Finetuning Support

This document describes the multi-agent (>2 players) support added to the SPIRAL codebase.

## Overview

The codebase has been extended to support RL finetuning on multi-agent games with more than 2 players. Previously, the system was designed exclusively for 2-player zero-sum games. Now it supports N-player games where N ≥ 2.

## Key Changes

### 1. Configurable Number of Players

- Added `num_players` parameter to `SelfPlayArgs` (default: 2 for backward compatibility)
- All game environments now initialize with the configured number of players
- Player ID assignment uses modulo arithmetic: `player_id = actor_id % num_players`

### 2. GameState Updates

The `GameState` class (`spiral/utils.py`) now supports N players:
- Constructor accepts `num_players` parameter
- `players_data` dictionary dynamically created for N players: `{i: [] for i in range(num_players)}`

### 3. Role-Based Baseline EMA

Role baseline tracking extended to N players:
- Each player position (0, 1, ..., N-1) maintains its own EMA baseline
- Enables role-specific reward shaping for multi-player games
- Helps with credit assignment in games where different positions have different strategic advantages

### 4. Trajectory Preparation

Updated to handle N players:
- `player_ids_for_training` dynamically set to `range(num_players)` in self-play mode
- Draw detection generalized: checks if all players receive equal rewards (all zeros)
- Supports training all players simultaneously in self-play

### 5. Evaluation Logic

Multi-player evaluation uses ranking-based outcomes:
- **Win**: Model finishes in 1st place (highest reward) without ties
- **Loss**: Model finishes in last place (lowest reward)
- **Draw**: Model ties for 1st, or finishes in middle ranks, or ties for last
- Added `model_rank` metric (1-indexed ranking)
- `opponent_reward` now represents average reward across all opponent players

### 6. Updated Prompts

Templates updated to reflect "competitive multi-player game" instead of "two-player zero-sum game":
- `apply_qwen3_template`
- `apply_r1_template`
- `apply_llama_instruct_template`

### 7. Reward Handling

Generalized reward structures:
- **Invalid actions**: Penalize acting player (-1.5), distribute positive reward to others (0.5 each)
- **Truncated games**: All players receive 0 reward
- **Terminal states**: Environment-specific rewards (can be non-zero-sum)

## Supported Environments

### Four-Player Chess

A new environment `FourPlayerChess-v1` has been added as a reference implementation:

**Location**: `spiral/envs/FourPlayerChess/env.py`

**Features**:
- Wraps the JAX-based 4-player chess environment from https://github.com/ericyuxuanye/4-player-chess-jax
- Implements TextArena interface for compatibility
- 4 players: Red (0), Blue (1), Yellow (2), Green (3)
- **Action space**: Numeric coordinates matching the board display
  - Format: `((start_row, start_col), (end_row, end_col))`
  - Coordinates: rows 0-13, columns 0-13
  - Matches the board renderer's coordinate system
  - Example moves: `((12, 4), (10, 4))`, `((13, 5), (11, 4))`, `((1, 4), (3, 4))`
  - Alternative formats accepted: `(12, 4, 10, 4)` or `[(12, 4, 10, 4)]`
- Non-zero-sum rewards (captures, checkmates, stalemates)

**Installation**:
```bash
pip install jax jaxlib
pip install git+https://github.com/ericyuxuanye/4-player-chess-jax.git
```

**Usage**:
```bash
python train_spiral.py \
    --env_ids FourPlayerChess-v1 \
    --num_players 4 \
    --use_llm_obs_wrappers False \
    ...
```

See `examples/four_player_chess_config.sh` for a complete configuration example.

### Existing Environments

The following existing environments are compatible:
- **KuhnPoker-v1**: 2 players (original support)
- **SimpleNegotiation-v1**: 2 players (original support)
- **LiarsDice-v1**: Supports N players (was always N-player capable, now fully integrated)
- **TicTacToe-v0**: 2 players (eval only)
- **PigDice-v1**: 2 players
- **TruthAndDeception-v1**: 2 players

To use LiarsDice with more players:
```bash
python train_spiral.py \
    --env_ids LiarsDice-v1 \
    --num_players 4 \
    --use_llm_obs_wrappers True \
    ...
```

## Usage Guide

### Basic Configuration

To train on a multi-player game:

```bash
python train_spiral.py \
    --env_ids <your_env_id> \
    --num_players <N> \
    --use_llm_obs_wrappers <True/False> \
    --use_role_baseline True \
    ...
```

### Key Parameters

- `--num_players N`: Number of players in the game (default: 2)
- `--use_role_baseline True`: Recommended for multi-player games to handle role-specific advantages
- `--role_baseline_ema_gamma 0.95`: Decay rate for role baseline EMA
- `--fixed_opponent ""`: Empty string for full self-play, or agent name for fixed opponents
- `--reward_scaling 1.0`: Scale factor for rewards
- `--gamma 1.0`: Discount factor (1.0 = no discounting, suitable for turn-based games)

### Self-Play vs Fixed Opponents

**Full Self-Play** (recommended for multi-player):
```bash
--fixed_opponent ""
```
All N players are controlled by the online model, enabling full multi-agent self-play.

**Fixed Opponents**:
```bash
--fixed_opponent "random"
```
Only the online model's assigned position is trained. Other positions use fixed agents.

## Adding Custom Multi-Player Environments

To add a new multi-player environment:

### 1. Create Environment Class

```python
# spiral/envs/YourGame/env.py
import textarena as ta

class YourGameEnv(ta.Env):
    def __init__(self):
        # Initialize your game
        pass

    def reset(self, num_players: int, seed: Optional[int] = None):
        # Reset for N players
        self.state = ta.State(
            num_players=num_players,
            min_players=2,
            max_players=10
        )
        # ... initialize game state

    def step(self, action: str) -> Tuple[bool, Dict[str, Any]]:
        # Process action and return (done, info)
        pass

    def close(self) -> Dict[int, float]:
        # Return final rewards: {player_id: reward}
        return {i: score for i, score in enumerate(final_scores)}
```

### 2. Register Environment

```python
# spiral/envs/__init__.py
register(
    id="YourGame-v1",
    entry_point="spiral.envs.YourGame.env:YourGameEnv",
)
```

### 3. Add Action Parser

```python
# spiral/agents/utils.py
def your_game_parse_available_actions(observation: str):
    # Parse and return list of valid actions from observation
    return available_actions

_VALID_ACTION_PARSER = {
    ...
    "YourGame-v1": your_game_parse_available_actions,
}
```

### 4. Train

```bash
python train_spiral.py \
    --env_ids YourGame-v1 \
    --num_players <N> \
    ...
```

## Metrics and Logging

### Training Metrics

- `actor/player_id`: Which player position generated this trajectory
- `actor/final_reward`: Final reward received by the player
- `actor/draw`: Boolean indicating if game ended in a draw
- `actor/game_length`: Total number of turns in the game

### Evaluation Metrics

- `eval/game/<env>/<opponent>/win_rate`: Win rate against opponent
- `eval/game/<env>/<opponent>/loss_rate`: Loss rate
- `eval/game/<env>/<opponent>/draw_rate`: Draw rate
- `eval/game/<env>/<opponent>/model_rank`: Average rank (1 = best)
- `eval/game/<env>/<opponent>/model_reward`: Average reward
- `eval/game/<env>/<opponent>/opponent_reward`: Average opponent reward

## Implementation Notes

### Design Decisions

1. **Backward Compatibility**: Default `num_players=2` ensures existing configurations work unchanged

2. **Flexible Outcomes**: Win/loss/draw classification works for both competitive (zero-sum) and cooperative reward structures

3. **Role Baselines**: Essential for games where different player positions have inherent advantages (e.g., first-mover advantage)

4. **Self-Play**: All N positions train simultaneously, creating a diverse training signal

### Limitations

1. **Sequential Turn-Taking**: Current implementation assumes sequential turns (turn-based games)

2. **Action Space**: Very large action spaces (like 4-player chess) may require environment-specific action selection strategies

3. **Evaluation**: Evaluation assumes all opponents use the same agent type (all random or all same LLM)

## Future Enhancements

Potential areas for extension:

1. **Simultaneous Actions**: Support for games where multiple players act simultaneously

2. **Team-Based Games**: Support for team-based multi-player games (e.g., 2v2)

3. **Population-Based Training**: Maintain a population of diverse agents

4. **Action Masking**: More sophisticated handling of legal action spaces

5. **Curriculum Learning**: Gradually increase number of opponents or game complexity

## References

- Original SPIRAL paper: [Link to paper]
- TextArena documentation: https://github.com/LeonGuertler/TextArena
- 4-Player Chess JAX: https://github.com/ericyuxuanye/4-player-chess-jax
