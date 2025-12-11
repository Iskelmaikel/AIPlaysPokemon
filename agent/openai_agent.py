import base64
import copy
import io
import json
import logging

from openai import OpenAI

from agent.emulator import Emulator
from config import MAX_TOKENS, OPENAI_MODEL, TEMPERATURE, USE_NAVIGATOR


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def get_screenshot_base64(screenshot, upscale=1):
    """Convert PIL image to base64 string."""
    if upscale > 1:
        new_size = (screenshot.width * upscale, screenshot.height * upscale)
        screenshot = screenshot.resize(new_size)

    buffered = io.BytesIO()
    screenshot.save(buffered, format="PNG")
    return base64.standard_b64encode(buffered.getvalue()).decode()


SYSTEM_PROMPT = """You are playing Pokemon Red. You can see the game screen and control the game by executing emulator commands.

Your goal is to play through Pokemon Red and eventually defeat the Elite Four. Make decisions based on what you see on the screen and the memory-based game state description.

At the very beginning of the game, you may see boot or title/intro screens. In those cases, you often need to press START and/or A multiple times to get to a playable state (the main game where you can move the character around). If the game appears idle or you are not yet in control of the player, try pressing START or A to proceed.

Before each action, explain your reasoning briefly, then use the available tools to execute your chosen commands.

The conversation history may occasionally be summarized to save context space. If you see a message labeled "CONVERSATION HISTORY SUMMARY", this contains the key information about your progress so far. Use this information to maintain continuity in your gameplay."""


SUMMARY_PROMPT = """I need you to create a detailed summary of our conversation history up to this point. This summary will replace the full conversation history to manage the context window.

Please include:
1. Key game events and milestones you've reached
2. Important decisions you've made
3. Current objectives or goals you're working toward
4. Your current location and Pokémon team status
5. Any strategies or plans you've mentioned

The summary should be comprehensive enough that you can continue gameplay without losing important context about what has happened so far."""


AVAILABLE_TOOLS = [
    {
        "name": "press_buttons",
        "description": "Press a sequence of buttons on the Game Boy.",
        "input_schema": {
            "type": "object",
            "properties": {
                "buttons": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": ["a", "b", "start", "select", "up", "down", "left", "right"],
                    },
                    "description": "List of buttons to press in sequence. Valid buttons: 'a', 'b', 'start', 'select', 'up', 'down', 'left', 'right'",
                },
                "wait": {
                    "type": "boolean",
                    "description": "Whether to wait for a brief period after pressing each button. Defaults to true.",
                },
            },
            "required": ["buttons"],
        },
    }
]

if USE_NAVIGATOR:
    AVAILABLE_TOOLS.append(
        {
            "name": "navigate_to",
            "description": "Automatically navigate to a position on the map grid. The screen is divided into a 9x10 grid, with the top-left corner as (0, 0). This tool is only available in the overworld.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "row": {
                        "type": "integer",
                        "description": "The row coordinate to navigate to (0-8).",
                    },
                    "col": {
                        "type": "integer",
                        "description": "The column coordinate to navigate to (0-9).",
                    },
                },
                "required": ["row", "col"],
            },
        }
    )


class OpenAIAgent:
    def __init__(self, rom_path, headless=True, sound=False, max_history=60, load_state=None):
        """Initialize the OpenAI-based agent with a feature set similar to SimpleAgent."""
        self.emulator = Emulator(rom_path, headless, sound)
        self.emulator.initialize()
        self.client = OpenAI()
        self.running = True
        self.message_history = [
            {"role": "user", "content": "You may now begin playing."}
        ]
        self.max_history = max_history
        if load_state:
            logger.info(f"Loading saved state from {load_state}")
            self.emulator.load_state(load_state)

    def _openai_tools(self):
        """Convert AVAILABLE_TOOLS into OpenAI tools schema."""
        tools = []
        for tool in AVAILABLE_TOOLS:
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": tool["description"],
                        "parameters": tool["input_schema"],
                    },
                }
            )
        return tools

    def _process_tool_call(self, tool_call):
        """Process a single OpenAI tool_call and return a text summary for a tool message."""
        name = tool_call.function.name
        try:
            arguments = json.loads(tool_call.function.arguments or "{}")
        except json.JSONDecodeError:
            logger.error(f"Failed to decode tool arguments for {name}: {tool_call.function.arguments}")
            arguments = {}

        logger.info(f"Processing tool call: {name}")

        if name == "press_buttons":
            buttons = arguments.get("buttons", [])
            wait = arguments.get("wait", True)  # Default to True for better game state updates
            logger.info(f"[Buttons] Pressing: {buttons} (wait={wait})")

            try:
                # Process the button presses
                result = self.emulator.press_buttons(buttons, wait)
                
                # Get the current game state after the button press
                memory_info = self.emulator.get_state_from_memory()
                location = self.emulator.get_location() or "Unknown location"
                dialog = self.emulator.get_active_dialog() or ""
                
                # Log the state for debugging
                logger.info("[Memory State after action]")
                logger.info(memory_info)
                
                collision_map = self.emulator.get_collision_map()
                if collision_map:
                    logger.info(f"[Collision Map after action]\n{collision_map}")
                
                # Build a detailed response
                response_parts = [
                    f"Action: Pressed buttons: {', '.join(buttons)}",
                    f"Location: {location}",
                ]
                
                if dialog:
                    response_parts.append(f"Dialog: {dialog}")
                
                response_parts.append(f"Game State:\n{memory_info}")
                
                return "\n".join(response_parts)
                
            except Exception as e:
                logger.error(f"Error processing button press: {e}")
                return f"Error processing button press: {str(e)}"

        if name == "navigate_to":
            row = arguments.get("row")
            col = arguments.get("col")
            logger.info(f"[Navigation] Navigating to: ({row}, {col})")

            status, path = self.emulator.find_path(row, col)
            if path:
                for direction in path:
                    self.emulator.press_buttons([direction], True)
                result = f"Navigation successful: followed path with {len(path)} steps. Status: {status}"
            else:
                result = f"Navigation failed: {status}"

            screenshot = self.emulator.get_screenshot()
            _ = get_screenshot_base64(screenshot, upscale=2)
            memory_info = self.emulator.get_state_from_memory()

            logger.info("[Memory State after navigation]")
            logger.info(memory_info)

            collision_map = self.emulator.get_collision_map()
            if collision_map:
                logger.info(f"[Collision Map after navigation]\n{collision_map}")

            return (
                f"Navigation result: {result}\n\n"
                f"Game state after navigation:\n{memory_info}"
            )

        logger.error(f"Unknown tool called: {name}")
        return f"Error: Unknown tool '{name}'"

    def run(self, num_steps=1):
        """Main agent loop using OpenAI's chat.completions and tools."""
        logger.info(f"Starting OpenAI agent loop for {num_steps} steps")

        steps_completed = 0
        while self.running and steps_completed < num_steps:
            try:
                messages = copy.deepcopy(self.message_history)

                # Get current screenshot and game state from memory for the user turn
                screenshot = self.emulator.get_screenshot()
                screenshot_b64 = get_screenshot_base64(screenshot, upscale=2)
                memory_info = self.emulator.get_state_from_memory()

                user_content = [
                    {
                        "type": "text",
                        "text": (
                            "Here is the current game state from memory and a screenshot of the game screen. "
                            "Use the tools to decide what to do next.\n\n"
                            f"GAME STATE:\n{memory_info}"
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{screenshot_b64}",
                        },
                    },
                ]

                messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT})
                messages.append({"role": "user", "content": user_content})

                response = self.client.chat.completions.create(
                    model=OPENAI_MODEL,
                    temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS,
                    messages=messages,
                    tools=self._openai_tools(),
                    tool_choice="auto",
                )

                message = response.choices[0].message

                # Log usage similar to SimpleAgent
                if getattr(response, "usage", None) is not None:
                    logger.info(f"Response usage: {response.usage}")

                # Log assistant text content (handle string or list parts)
                if message.content:
                    if isinstance(message.content, str):
                        logger.info(f"[Text] {message.content}")
                    else:
                        for part in message.content:
                            if part.get("type") == "text":
                                logger.info(f"[Text] {part.get('text', '')}")

                tool_calls = message.tool_calls or []
                for tc in tool_calls:
                    logger.info(f"[Tool] Using tool: {tc.function.name}")

                # Update history with this turn (keep structures close to SimpleAgent)
                self.message_history.append({"role": "user", "content": user_content})
                self.message_history.append({"role": "assistant", "content": message.content or ""})

                # Execute tools, but do not persist tool messages in history.
                # OpenAI expects tool messages only as immediate responses to tool_calls
                # in the same request; keeping them in history causes 400 errors.
                for tc in tool_calls:
                    _ = self._process_tool_call(tc)

                # Summarize history if it grows too large
                if len(self.message_history) >= self.max_history:
                    self.summarize_history()

                steps_completed += 1
                logger.info(f"Completed step {steps_completed}/{num_steps}")

            except KeyboardInterrupt:
                logger.info("Received keyboard interrupt, stopping")
                self.running = False
            except Exception as e:
                logger.error(f"Error in OpenAI agent loop: {e}")
                raise e

        if not self.running:
            self.emulator.stop()

        return steps_completed

    def summarize_history(self):
        """Generate a summary of the conversation history and replace it with the summary."""
        logger.info("[OpenAI Agent] Generating conversation summary...")

        screenshot = self.emulator.get_screenshot()
        screenshot_b64 = get_screenshot_base64(screenshot, upscale=2)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            *copy.deepcopy(self.message_history),
            {"role": "user", "content": SUMMARY_PROMPT},
        ]

        response = self.client.chat.completions.create(
            model=OPENAI_MODEL,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            messages=messages,
        )

        summary_text = response.choices[0].message.content or ""

        logger.info("[OpenAI Agent] Game Progress Summary:")
        logger.info(summary_text)

        # Replace history with a single user message composed of text + image, similar to SimpleAgent
        self.message_history = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            f"CONVERSATION HISTORY SUMMARY (representing {self.max_history} previous messages): {summary_text}"
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{screenshot_b64}",
                        },
                    },
                    {
                        "type": "text",
                        "text": (
                            "You were just asked to summarize your playthrough so far, which is the summary you see above. "
                            "You may now continue playing by selecting your next action."
                        ),
                    },
                ],
            }
        ]

        logger.info("[OpenAI Agent] Message history condensed into summary.")

    def stop(self):
        """Stop the agent and emulator."""
        self.running = False
        self.emulator.stop()
