
"""
chat_memory.py
--------------
Sliding-window short-term memory for a CLI chatbot.
"""
from typing import List, Tuple, Optional

RoleTurn = Tuple[str, str]  # ("user" or "assistant", text)

class ChatMemory:
    def __init__(self, window_size: int = 4, system_prompt: Optional[str] = None):
        """
        Args:
            window_size: Number of *turns* (user+assistant pairs) to keep.
            system_prompt: Optional instruction prepended to every prompt.
        """
        assert window_size >= 1, "window_size must be >= 1"
        self.window_size = window_size
        self.system_prompt = system_prompt
        self.history: List[RoleTurn] = []

    def add_user(self, text: str):
        self.history.append(("user", text.strip()))

    def add_assistant(self, text: str):
        self.history.append(("assistant", text.strip()))

    def _recent(self) -> List[RoleTurn]:
        """
        Return the last N turns where a 'turn' is a user+assistant pair.
        We implement this by scanning from the end and collecting until
        we have window_size user messages (which start turns).
        """
        if self.window_size <= 0:
            return self.history[:]

        # Collect from the end, counting user messages as turn starters
        recent: List[RoleTurn] = []
        user_count = 0
        for role, text in reversed(self.history):
            recent.append((role, text))
            if role == "user":
                user_count += 1
                if user_count >= self.window_size:
                    break
        recent.reverse()
        return recent

    def build_prompt(self) -> str:
        """
        Build a simple plain-text chat-style prompt for causal LMs.
        """
        lines = []
        if self.system_prompt:
            lines.append(f"System: {self.system_prompt.strip()}")
        for role, text in self._recent():
            prefix = "User" if role == "user" else "Assistant"
            lines.append(f"{prefix}: {text}")
        # Ensure the model continues as Assistant
        lines.append("Assistant:")
        return "\n".join(lines)
