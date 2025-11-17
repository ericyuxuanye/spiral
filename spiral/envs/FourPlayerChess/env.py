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
    Uses numeric coordinate notation: ((start_row, start_col), (end_row, end_col))
    Coordinates match the board display: rows 0-13, cols 0-13
    Example: ((12, 3), (10, 3)) to move from row 12, col 3 to row 10, col 3
    """

    def __init__(self):
        """Initialize the 4-player chess environment."""
        if not JAX_AVAILABLE:
            raise ImportError(
                "JAX and four_player_chess_jax are required for FourPlayerChessEnv. "
                "Install with: pip install jax four-player-chess-jax"
            )

        self.jax_env = JAXChessEnv()
        # Match coordinate tuples like ((12, 3), (10, 3)) or (12,3,10,3)
        # Pattern matches: ((r1, c1), (r2, c2)) or (r1, c1, r2, c2) or [r1, c1, r2, c2]
        self.action_pattern = re.compile(
            r'\[?\(??\(?\s*(\d+)\s*,\s*(\d+)\s*\)?\s*,?\s*\(?\s*(\d+)\s*,\s*(\d+)\s*\)?\]?'
        )
        self.player_names = ["Red", "Blue", "Yellow", "Green"]
        self.player_colors = {0: "Red", 1: "Blue", 2: "Yellow", 3: "Green"}

        # Create valid square mask for the cross-shaped board
        self.valid_mask = self._create_valid_mask()

    def _create_valid_mask(self):
        """Create valid square mask matching the JAX environment."""
        try:
            import jax.numpy as jnp
            mask = jnp.zeros((14, 14), dtype=jnp.int32)

            # Central 8x8 area
            mask = mask.at[3:11, 3:11].set(1)
            # Red extension (bottom)
            mask = mask.at[11:14, 3:11].set(1)
            # Blue extension (right)
            mask = mask.at[3:11, 11:14].set(1)
            # Yellow extension (top)
            mask = mask.at[0:3, 3:11].set(1)
            # Green extension (left)
            mask = mask.at[3:11, 0:3].set(1)

            return mask
        except:
            return None

    def reset(self, num_players: int = 4, seed: Optional[int] = None):
        """Reset the 4-player chess game to its initial state."""
        if num_players != 4:
            raise ValueError("FourPlayerChess only supports exactly 4 players")

        self.state = ta.State(num_players=4)

        # Initialize JAX environment
        rng = jax.random.PRNGKey(seed if seed is not None else 0)
        jax_state, obs = self.jax_env.reset(rng)

        game_state = {
            "jax_state": jax_state,
            "jax_obs": obs,
            "rng": rng,
            "move_count": 0,
            "active_players": [0, 1, 2, 3],
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

        # Get some example legal moves (for illustration)
        example_moves = self._get_example_moves(player_id)

        prompt = (
            f"You are playing 4-Player Chess as {player_color} (Player {player_id}).\n\n"
            f"Board State:\n{board_str}\n\n"
            f"Move #{move_count}\n"
            f"Current player: {self.player_colors[current_player]} (Player {current_player})\n\n"
            "Game Rules:\n"
            "- 4 players take turns clockwise: Red (0) -> Blue (1) -> Yellow (2) -> Green (3)\n"
            "- Standard chess rules apply with adaptations for 4 players\n"
            "- Capture opponent pieces to earn points (+1 pawn, +3 knight/bishop, +5 rook, +9 queen)\n"
            "- Checkmate an opponent: +20 points\n"
            "- Stalemate opponents: +10 points × remaining players\n\n"
            "Move Format:\n"
            "- Specify moves using board coordinates: ((start_row, start_col), (end_row, end_col))\n"
            "- Coordinates match the board display above (rows 0-13, columns 0-13)\n"
            "- The board shows row numbers on the left (0-13) and column numbers on top (0-13)\n"
            f"- Example moves for your pieces: {example_moves}\n"
            "- Format: ((12, 4), (10, 4)) to move piece from row 12, col 4 to row 10, col 4\n"
            "- You can also use: (12, 4, 10, 4) or [(12, 4, 10, 4)]\n\n"
            "Your move?"
        )

        return prompt

    def _get_example_moves(self, player_id: int) -> str:
        """Get example moves for the given player based on starting positions."""
        # Red (0) pieces are at bottom (rows 12-13)
        # Blue (1) pieces are at right (cols 12-13)
        # Yellow (2) pieces are at top (rows 0-1)
        # Green (3) pieces are at left (cols 0-1)

        examples = {
            0: "((12, 4), (10, 4)), ((12, 5), (11, 5))",  # Red: pawn forward
            1: "((4, 12), (4, 10)), ((5, 12), (5, 11))",  # Blue: pawn forward
            2: "((1, 4), (3, 4)), ((1, 5), (2, 5))",      # Yellow: pawn forward
            3: "((4, 1), (4, 3)), ((5, 1), (5, 2))",      # Green: pawn forward
        }
        return examples.get(player_id, "((12, 4), (10, 4)), ((12, 5), (11, 5))")

    def step(self, action: str) -> Tuple[bool, Dict[str, Any]]:
        """
        Process a player's action.

        Args:
            action: String containing the move, e.g., "((12, 3), (10, 3))" or "(12, 3, 10, 3)"

        Returns:
            Tuple of (done, info)
        """
        # Parse action from string
        match = self.action_pattern.search(action)
        if not match:
            return True, {"reason": f"Invalid move format. Use ((start_row, start_col), (end_row, end_col)). Got: {action}"}

        try:
            start_row = int(match.group(1))
            start_col = int(match.group(2))
            end_row = int(match.group(3))
            end_col = int(match.group(4))

        except (ValueError, IndexError) as e:
            return True, {"reason": f"Invalid coordinate format: {str(e)}"}

        # Validate coordinates are on the board
        if not (0 <= start_row < 14 and 0 <= start_col < 14 and
                0 <= end_row < 14 and 0 <= end_col < 14):
            return True, {"reason": f"Coordinates out of bounds: (({start_row}, {start_col}), ({end_row}, {end_col}))"}

        # Check if squares are valid (not in corners)
        if self.valid_mask is not None:
            if self.valid_mask[start_row, start_col] == 0 or self.valid_mask[end_row, end_col] == 0:
                return True, {"reason": f"Invalid square (corner): (({start_row}, {start_col}), ({end_row}, {end_col}))"}

        # Encode move as action number
        # For simplicity, assume no promotion (promotion_type=0)
        # In a full implementation, you'd parse promotion from the move string
        promotion_type = 0

        try:
            # Use the JAX environment's encode_action function
            from four_player_chess_jax.four_player_chess.utils import encode_action

            action_int = encode_action(
                jnp.int32(start_row),
                jnp.int32(start_col),
                jnp.int32(end_row),
                jnp.int32(end_col),
                jnp.int32(promotion_type),
                self.valid_mask
            )
            action_int = int(action_int)

        except Exception as e:
            return True, {"reason": f"Error encoding move: {str(e)}"}

        # Execute action in JAX environment
        game_state = self.state.game_state
        rng, step_rng = jax.random.split(game_state["rng"])

        try:
            jax_state, obs, reward, done, info = self.jax_env.step(
                step_rng, game_state["jax_state"], action_int
            )

            # Check if move was valid
            if not info.get('move_valid', False):
                return True, {"reason": f"Illegal move: (({start_row}, {start_col}), ({end_row}, {end_col}))"}

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
            return True, {"reason": f"Error executing move: {str(e)}"}

    def close(self) -> Dict[int, float]:
        """
        Get final rewards for all players when the game ends.

        Returns:
            Dictionary mapping player_id to final reward
        """
        jax_state = self.state.game_state["jax_state"]

        # Extract scores from JAX state
        rewards = {}
        for player_id in range(4):
            # The JAX environment stores scores in player_scores
            if hasattr(jax_state, 'player_scores'):
                rewards[player_id] = float(jax_state.player_scores[player_id])
            else:
                # Fallback
                rewards[player_id] = 0.0

        return rewards

