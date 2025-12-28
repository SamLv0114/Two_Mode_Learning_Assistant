"""
Command-line interface for the AI Learning Assistant
"""
import argparse
import sys
from src.pipelines import DailyFeedPipeline, QAAssistant
from src.initialize import initialize


def main():
    parser = argparse.ArgumentParser(
        description="AI Learning Assistant - Two-Mode System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Initialize the system
  python cli.py init
  
  # Run daily feed
  python cli.py daily-feed
  
  # Ask a question
  python cli.py ask "What is attention mechanism?"
  
  # Interactive Q&A mode
  python cli.py interactive
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Init command
    subparsers.add_parser("init", help="Initialize the system")
    
    # Daily feed command
    subparsers.add_parser("daily-feed", help="Run daily feed pipeline")
    
    # Ask command
    ask_parser = subparsers.add_parser("ask", help="Ask a question")
    ask_parser.add_argument("question", help="Your question")
    ask_parser.add_argument("-n", "--context", type=int, default=5, 
                           help="Number of context documents (default: 5)")
    
    # Interactive mode
    subparsers.add_parser("interactive", help="Start interactive Q&A session")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    if args.command == "init":
        initialize()
    
    elif args.command == "daily-feed":
        print("Running daily feed pipeline...")
        pipeline = DailyFeedPipeline()
        result = pipeline.run()
        print(pipeline.format_for_display(result))
    
    elif args.command == "ask":
        assistant = QAAssistant()
        result = assistant.answer_question(args.question, n_context=args.context)
        print(assistant.format_answer(result))
    
    elif args.command == "interactive":
        assistant = QAAssistant()
        assistant.interactive_mode()


if __name__ == "__main__":
    main()

