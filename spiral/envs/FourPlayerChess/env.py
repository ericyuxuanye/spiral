# Copyright 2025 SPIRAL Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
from typing import Any, Dict, Optional, Tuple

import textarena as ta

try:
    import jax
    import jax.numpy as jnp
    from four_player_chess_jax import FourPlayerChessEnv as JAXChessEnv
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False


class FourPlayerChessEnv(ta.Env):
    """
    TextArena wrapper for 4-player chess JAX environment.
    Adapts the JAX-based 4-player chess to TextArena interface.
    """

    def __init__(self):
        """Initialize the 4-player chess environment."""
        if not JAX_AVAILABLE:
            raise ImportError(
                "JAX and four_player_chess_jax are required for FourPlayerChessEnv. "
                "Install with: pip install jax four-player-chess-jax"
            )

        self.jax_env = JAXChessEnv()
        self.action_pattern = re.compile(r"\[(\d+)\]", re.IGNORECASE)
        self.player_names = ["Red", "Blue", "Yellow", "Green"]
        self.player_colors = {0: "Red", 1: "Blue", 2: "Yellow", 3: "Green"}

    def reset(self, num_players: int = 4, seed: Optional[int] = None):
        """Reset the 4-player chess game to its initial state."""
        if num_players != 4:
            raise ValueError("FourPlayerChess only supports exactly 4 players")

        self.state = ta.State(num_players=4, min_players=4, max_players=4)

        # Initialize JAX environment
        rng = jax.random.PRNGKey(seed if seed is not None else 0)
        jax_state, obs = self.jax_env.reset(rng)

        game_state = {
            "jax_state": jax_state,
            "jax_obs": obs,
            "rng": rng,
            "move_count": 0,
            "active_players": [0, 1, 2, 3],  # Track which players are still active
        }

        self.state.reset(
            seed=seed,
            game_state=game_state,
            player_prompt_function=self._generate_player_prompt,
        )

    def _generate_player_prompt(self, player_id: int, game_state: Dict[str, Any]) -> str:
        """Generate the observation prompt for a given player."""
        jax_state = game_state["jax_state"]

        # Get ASCII board representation
        board_str = self.jax_env.render(jax_state)

        # Get current player info
        current_player = int(jax_state.current_player)
        player_color = self.player_colors[player_id]

        # Get game status
        move_count = game_state["move_count"]

        prompt = (
            f"You are playing 4-Player Chess as {player_color} (Player {player_id}).\n\n"
            f"Board State:\n{board_str}\n\n"
            f"Move #{move_count}\n"
            f"Current player: {self.player_colors[current_player]} (Player {current_player})\n\n"
            "Game Rules:\n"
            "- 4 players take turns clockwise: Red (0) -> Blue (1) -> Yellow (2) -> Green (3)\n"
            "- Capture opponent pieces to earn points\n"
            "- Checkmate an opponent to earn 20 points\n"
            "- Stalemate opponents to earn 10 points × remaining players\n\n"
            "Action Format:\n"
            "- Actions are encoded as integers from 0 to 102,400\n"
            "- Respond with your action in the format: [action_number]\n"
            "- Example: [12345] to play action 12345\n\n"
            "Your action?"
        )

        return prompt

    def step(self, action: str) -> Tuple[bool, Dict[str, Any]]:
        """
        Process a player's action.

        Args:
            action: String containing the action (e.g., "[12345]")

        Returns:
            Tuple of (done, info)
        """
        # Parse action from string
        match = self.action_pattern.search(action)
        if not match:
            return True, {"reason": "Invalid action format. Use [action_number]"}

        try:
            action_int = int(match.group(1))
        except ValueError:
            return True, {"reason": "Invalid action number"}

        # Validate action range
        if action_int < 0 or action_int > 102400:
            return True, {"reason": f"Action {action_int} out of valid range [0, 102400]"}

        # Execute action in JAX environment
        game_state = self.state.game_state
        rng, step_rng = jax.random.split(game_state["rng"])

        try:
            jax_state, obs, reward, done, info = self.jax_env.step(
                step_rng, game_state["jax_state"], action_int
            )

            # Update game state
            game_state["jax_state"] = jax_state
            game_state["jax_obs"] = obs
            game_state["rng"] = rng
            game_state["move_count"] += 1

            # Update current player in TextArena state
            self.state.current_player = int(jax_state.current_player)

            # Check if game is done
            if done:
                return True, info

            return False, info

        except Exception as e:
            return True, {"reason": f"Error executing action: {str(e)}"}

    def close(self) -> Dict[int, float]:
        """
        Get final rewards for all players when the game ends.

        Returns:
            Dictionary mapping player_id to final reward
        """
        jax_state = self.state.game_state["jax_state"]

        # Extract rewards from JAX state
        # The JAX environment should have rewards stored in the state
        rewards = {}
        for player_id in range(4):
            # Convert JAX array to float
            if hasattr(jax_state, 'rewards'):
                rewards[player_id] = float(jax_state.rewards[player_id])
            elif hasattr(jax_state, 'scores'):
                rewards[player_id] = float(jax_state.scores[player_id])
            else:
                # Fallback: use equal rewards
                rewards[player_id] = 0.0

        return rewards
