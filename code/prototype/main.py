"""
Main Entry Point

Run the OGD Fuzzy Retrieval System prototype.
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


def run_prototype():
    """Launch the Streamlit prototype."""
    import subprocess
    app_path = PROJECT_ROOT / "prototype" / "app.py"
    subprocess.run(["streamlit", "run", str(app_path)])


def run_demo():
    """Run a quick demonstration of the fuzzy system."""
    print("=" * 60)
    print("OGD FUZZY RETRIEVAL SYSTEM - DEMONSTRATION")
    print("=" * 60)
    
    # Import and demo fuzzy system
    try:
        from code.fuzzy_system import create_inference_engine
        
        engine = create_inference_engine()
        
        # Test inputs
        test_inputs = {
            "recency": 15,
            "completeness": 0.85,
            "thematic_similarity": 0.75,
            "resource_availability": 5
        }
        
        print("\nInput values:")
        for name, value in test_inputs.items():
            print(f"  {name}: {value}")
        
        result = engine.infer(test_inputs)
        
        print(f"\nRelevance Score: {result.crisp_output:.1f}/100")
        print(f"\nExplanation:\n{result.get_explanation()}")
        
    except ImportError as e:
        print(f"Import error: {e}")
        print("Run 'pip install -r requirements.txt' first.")


def run_collect_data(theme: str = None, limit: int = 100):
    """Collect dataset metadata from opendata.swiss."""
    try:
        from code.data_collection import CKANAPIClient
        
        client = CKANAPIClient()
        
        print(f"Collecting datasets from opendata.swiss...")
        if theme:
            print(f"Theme filter: {theme}")
        
        datasets = client.search_datasets(
            query="*:*" if not theme else theme,
            rows=limit
        )
        
        print(f"Retrieved {len(datasets)} datasets")
        
        # Save to cache
        import json
        cache_path = PROJECT_ROOT.parent / "data" / "datasets_cache.json"
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(datasets, f, ensure_ascii=False, indent=2)
        print(f"Saved to {cache_path}")
        
    except Exception as e:
        print(f"Error: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="OGD Fuzzy Retrieval System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py prototype    # Launch web UI
  python main.py demo         # Run quick demo
  python main.py collect      # Collect OGD metadata
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Prototype command
    proto_parser = subparsers.add_parser("prototype", help="Launch Streamlit prototype")
    
    # Demo command
    demo_parser = subparsers.add_parser("demo", help="Run quick demonstration")
    
    # Collect command
    collect_parser = subparsers.add_parser("collect", help="Collect OGD metadata")
    collect_parser.add_argument("--theme", type=str, help="Theme to filter by")
    collect_parser.add_argument("--limit", type=int, default=100, help="Max datasets")
    
    args = parser.parse_args()
    
    if args.command == "prototype":
        run_prototype()
    elif args.command == "demo":
        run_demo()
    elif args.command == "collect":
        run_collect_data(args.theme, args.limit)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
