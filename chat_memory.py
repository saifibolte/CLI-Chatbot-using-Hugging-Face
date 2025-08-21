"""
Chat Memory Management Module

This module implements a sliding window memory system for chatbot conversations.
It maintains conversation history while limiting memory usage by keeping only 
the most recent turns within a specified window size.
"""

class SlidingWindowMemory:
    """
    Manages conversation memory using a sliding window approach.
    
    This class stores chat messages in a format compatible with Hugging Face 
    transformers, maintaining a system prompt and a limited number of recent
    conversation turns to prevent memory overflow.
    """

    def __init__(self, max_turns=4, system_prompt=None):
        """
        Initialize the sliding window memory.
        
        """
        # Ensure max_turns is at least 1 to prevent empty conversations
        self.max_turns = max(1, int(max_turns))

        # Set default system prompt if none provided
        self.system_prompt = system_prompt or "You are a helpful assistant."

        # Initialize messages list with system prompt
        # Format follows HuggingFace chat template: [{"role": "system/user/assistant", "content": "..."}]
        self._messages = [{"role": "system", "content": self.system_prompt}]

    @property
    def messages(self):
        """
        Get a copy of all current messages.
        
        """
        return list(self._messages)

    def add_user(self, content):
        """
        Add a user message to the conversation history.
        
        """

        self._messages.append({"role": "user", "content": content})
        self._trim() # Remove old messages if we exceed the window

    def add_assistant(self, content):
        """
        Add an assistant message to the conversation history.
        """

        self._messages.append({"role": "assistant", "content": content})
        self._trim() # Remove old messages if we exceed the window

    def clear(self):
        """
        Clear all conversation history except the system prompt.
        
        This resets the conversation while preserving the system prompt,
        useful for starting fresh conversations with the same assistant behavior.
        """

        self._messages = [{"role": "system", "content": self.system_prompt}]

    def _trim(self):

        """
        Private method to maintain sliding window by removing old messages.
        
        This method preserves the system message (first message) and keeps only
        the most recent conversation turns within the specified window size.
        A "turn" consists of one user message and one assistant response.
        
        Example:
            If max_turns=2, keeps: [system, user1, assistant1, user2, assistant2]
            If we add user3, assistant3, it becomes: [system, user2, assistant2, user3, assistant3]
        """
         # Always keep the system message (first element)
        system = self._messages[:1]

         # Get all conversation messages (everything after system message)
        turns = self._messages[1:]

        # Keep only the most recent turns (2 messages per turn: user + assistant)
        # Negative indexing gets the last 2*max_turns messages
        keep = turns[-(2 * self.max_turns):]

        # Reconstruct messages list with system + recent turns
        self._messages = system + keep
