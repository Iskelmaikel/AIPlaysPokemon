import argparse
import logging
import os
import msvcrt
import threading
import time

from agent.simple_agent import SimpleAgent
from agent.openai_agent import OpenAIAgent

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)

logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="AI Plays Pokemon - Starter Version")
    parser.add_argument(
        "--rom", 
        type=str, 
        default="pokemon.gb",
        help="Path to the Pokemon ROM file"
    )
    parser.add_argument(
        "--steps", 
        type=int, 
        default=10, 
        help="Number of agent steps to run (not used in continuous mode)"
    )
    parser.add_argument(
        "--display", 
        action="store_true", 
        help="Run with display (not headless)"
    )
    parser.add_argument(
        "--sound", 
        action="store_true", 
        help="Enable sound (only applicable with display)"
    )
    parser.add_argument(
        "--max-history", 
        type=int, 
        default=30, 
        help="Maximum number of messages in history before summarization"
    )
    parser.add_argument(
        "--load-state", 
        type=str, 
        default=None, 
        help="Path to a saved state to load"
    )
    parser.add_argument(
        "--provider",
        type=str,
        choices=["anthropic", "openai"],
        default="anthropic",
        help="Which AI provider to use: 'anthropic' (Claude) or 'openai'"
    )
    
    args = parser.parse_args()
    
    # Get absolute path to ROM
    if not os.path.isabs(args.rom):
        rom_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.rom) 
    else:
        rom_path = args.rom
    
    # Check if ROM exists
    if not os.path.exists(rom_path):
        logger.error(f"ROM file not found: {rom_path}")
        print("\nYou need to provide a Pokemon Red ROM file to run this program.")
        print("Place the ROM in the root directory or specify its path with --rom.")
        return
    
    # Create agent
    if args.provider == "openai":
        agent = OpenAIAgent(
            rom_path=rom_path,
            headless=not args.display,
            sound=args.sound if args.display else False,
            max_history=args.max_history,
            load_state=args.load_state,
        )
    else:
        agent = SimpleAgent(
            rom_path=rom_path,
            headless=not args.display,
            sound=args.sound if args.display else False,
            max_history=args.max_history,
            load_state=args.load_state,
        )
    
    # Pre-AI manual phase: keep ticking the emulator and wait for NumPad 8
    print("\nGame initialized. You can play manually now.")
    print("When you want the AI to take over, press NumPad 8 (with NumLock on) in this console.")
    # Control how many emulator frames we advance per loop (affects effective FPS)
    frames_per_step = 1   # 1 frame per ~1/60s -> ~60fps
    sleep_seconds = 1 / 60

    try:
        while True:
            # Advance some frames so the game keeps running
            agent.emulator.tick(frames_per_step)

            # Non-blocking check for NumPad 8 or speed controls in the console
            if msvcrt.kbhit():
                key = msvcrt.getch()
                if key in (b"8",):
                    break
                elif key in (b"-",):
                    print("\n[Speed] Setting emulator to ~60fps")
                    frames_per_step = 1
                    sleep_seconds = 1 / 60
                elif key in (b"=", b"+",):
                    print("\n[Speed] Setting emulator to ~300fps")
                    frames_per_step = 5      # 5 * 60 ~= 300fps
                    sleep_seconds = 1 / 60   # keep same wall-clock rate, more frames per tick

            time.sleep(sleep_seconds)

    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt during manual phase, exiting.")
        agent.stop()
        return

    # Now let the AI take over in a background thread
    def ai_loop():
        try:
            logger.info("AI agent started - will keep playing until you stop it")
            while True:  # Run indefinitely
                agent.run(num_steps=args.steps)  # Use the steps parameter from command line
                time.sleep(0.1)  # Small delay to prevent CPU overuse
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt in AI thread, stopping")
        except Exception as e:
            logger.error(f"Error running agent: {e}")

    ai_thread = threading.Thread(target=ai_loop, daemon=True)
    ai_thread.start()

    # Main loop to keep the emulator running and handle user input
    try:
        print("\nAI is now playing. Press 'q' to quit or use speed controls (-/=)")
        while True:
            agent.emulator.tick(frames_per_step)
            
            # Handle user input for controls
            if msvcrt.kbhit():
                key = msvcrt.getch()
                if key in (b"q", b"Q"):
                    print("\nQuitting...")
                    break
                elif key in (b"-",):
                    print("\n[Speed] Setting emulator to ~60fps")
                    frames_per_step = 1
                    sleep_seconds = 1 / 60
                elif key in (b"=", b"+"):
                    print("\n[Speed] Setting emulator to ~300fps")
                    frames_per_step = 5
                    sleep_seconds = 1 / 60
            
            time.sleep(sleep_seconds)

    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, stopping.")
    finally:
        agent.stop()

if __name__ == "__main__":
    main()