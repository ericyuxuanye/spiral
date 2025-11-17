#!/bin/bash
# Example configuration for training on 4-player chess
# This script demonstrates how to use the multi-agent RL finetuning
# on games with more than 2 players

# Note: You need to install the 4-player chess JAX environment first:
# pip install git+https://github.com/ericyuxuanye/4-player-chess-jax.git

python train_spiral.py \
    --env_ids FourPlayerChess-v1 \
    --use_llm_obs_wrappers False \
    --num_players 4 \
    --num_envs 1 \
    --eval_env_ids FourPlayerChess-v1 \
    --eval_use_llm_obs_wrappers False \
    --eval_opponent_names random \
    --prompt_template qwen3 \
    --eval_prompt_template qwen3_general \
    --model_path <your_model_path> \
    --save_path ./checkpoints/four_player_chess \
    --rollout_batch_size_per_device 8 \
    --train_batch_size_per_device 4 \
    --eval_batch_size 32 \
    --temperature 0.8 \
    --top_p 0.9 \
    --generate_max_length 512 \
    --max_turns 100 \
    --reward_scaling 1.0 \
    --gamma 1.0 \
    --use_role_baseline True \
    --role_baseline_ema_gamma 0.95 \
    --fixed_opponent "" \
    --num_actors 4 \
    --num_learners 1

# Key parameters for multi-agent games:
# --num_players 4: Set to the number of players in your game
# --env_ids: The environment ID (e.g., FourPlayerChess-v1)
# --use_role_baseline True: Use role-specific baselines for reward shaping
# --fixed_opponent "": Empty for self-play, or "random" to play against random agents
#
# For other multi-player games, you can adjust num_players accordingly:
# - 3-player games: --num_players 3
# - 4-player games: --num_players 4
# - etc.
