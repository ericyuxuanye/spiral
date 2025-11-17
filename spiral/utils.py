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

from typing import Any, Dict, List


class EMA:
    def __init__(self, decay: float):
        assert 0.0 < decay < 1.0, "Decay must be between 0 and 1"
        self.decay = decay
        self.value = 0.0

    def update(self, x: float) -> float:
        if self.value is None:
            self.value = x
        else:
            self.value = self.decay * self.value + (1 - self.decay) * x
        return self.value

    def get(self) -> float:
        return self.value


class GameState:
    """Class to maintain game state and history."""

    def __init__(self, max_context_length: int = 1024, max_turns: int = 50, num_players: int = 2):
        self.history = []  # Game interaction history
        self.long_history = []  # Game interaction history
        self.max_context_length = max_context_length
        self.max_turns = max_turns
        self.turn_count = 0
        self.num_players = num_players
        # Store player-specific trajectory data for N players
        self.players_data = {i: [] for i in range(num_players)}

    def add_interaction(
        self, player_id: int, observation: str, action: str, thinking: str
    ) -> None:
        """Add a turn interaction to the game history."""
        self.history.append((player_id, observation, action))
        self.long_history.append((player_id, observation, action, thinking))
        self.turn_count += 1

    def get_full_history_text(self) -> str:
        """Get the full game history as text."""
        history_text = []
        for player_id, observation, action in self.history:
            history_text.append(f"Player {player_id} observed:\n{observation}")
            history_text.append(f"Player {player_id} action:\n{action}")

        # Truncate if necessary to stay within max_context_length
        full_text = "\n".join(history_text)
        if len(full_text) > self.max_context_length:
            # Simple truncation strategy - could be improved with smarter summarization
            excess = len(full_text) - self.max_context_length
            full_text = full_text[excess:]

        return full_text

    def is_truncated(self) -> bool:
        """Check if game should be truncated due to max turns."""
        return self.turn_count >= self.max_turns

    def add_trajectory_data(self, player_id: int, data: Dict[str, Any]) -> None:
        """Add trajectory data for a specific player."""
        self.players_data[player_id].append(data)

    def get_player_trajectories(self, player_id: int) -> List[Dict[str, Any]]:
        """Get all trajectory data for a specific player."""
        return self.players_data[player_id]


def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx == None:
        retval = None
    else:
        retval = string[idx : right_brace_idx + 1]

    return retval


def remove_boxed(s):
    left = "\\boxed{"
    try:
        assert s[: len(left)] == left
        assert s[-1] == "}"
        return s[len(left) : -1]
    except:
        return None


def remove_text_boxed(s):
    left = "\\text{"
    try:
        assert s[: len(left)] == left
        assert s[-1] == "}"
        return s[len(left) : -1]
    except:
        return s


def extract_boxed_answer(solution: str) -> str:
    """Extract the answer from inside a LaTeX \\boxed{} command"""
    solution = last_boxed_only_string(solution)
    solution = remove_boxed(solution)
    solution = remove_text_boxed(solution)
    return solution
