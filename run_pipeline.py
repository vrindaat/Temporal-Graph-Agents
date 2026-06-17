#!/usr/bin/env python3
"""
Unified pipeline runner - single entry point for the entire system.

Usage:
  # Run complete pipeline (ingest + query)
  python run_pipeline.py --config configs/pipeline_amazon_reviews.yaml

  # Query existing graph only
  python run_pipeline.py --config configs/pipeline_query_only.yaml --skip-ingestion

  # Interactive mode
  python run_pipeline.py --config configs/pipeline_query_only.yaml --interactive

  # Single query
  python run_pipeline.py --config configs/pipeline_query_only.yaml --entity Apple --year 2021
"""
import argparse
import sys
from pathlib import Path

from src.config.pipeline_config import PipelineConfig
from src.pipeline.orchestrator import PipelineOrchestrator


def interactive_mode(orchestrator: PipelineOrchestrator):
    """Run interactive query loop"""
    entities = orchestrator.get_available_entities()
    print(f"\n[Ready] {len(entities)} entities available")
    print(f"Examples: {entities[:10]}\n")

    while True:
        print("=" * 80)
        entity = input("\n>> Entity (or 'q' to quit): ").strip()

        if entity.lower() == 'q':
            break

        # Find matching entity (case-insensitive)
        matched = next((e for e in entities if e.lower() == entity.lower()), None)
        if not matched:
            matches = [e for e in entities if entity.lower() in e.lower()]
            if matches:
                print(f"  Not found. Similar: {matches[:5]}")
            else:
                print(f"  '{entity}' not in graph.")
            continue

        try:
            y1 = int(input(">> Baseline year: ").strip())
            y2 = int(input(">> Comparison year: ").strip())
        except ValueError:
            print("  Invalid year.")
            continue

        # Run audits
        try:
            result1 = orchestrator.run_audit(matched, y1)
            print(f"\n[Report]:\n{result1['report'][:500]}...")
            print(f"\n[Verdict]: {result1['status']} — {result1['verification']['reasoning']}")

            print()
            result2 = orchestrator.run_audit(matched, y2)
            print(f"\n[Report]:\n{result2['report'][:500]}...")
            print(f"\n[Verdict]: {result2['status']} — {result2['verification']['reasoning']}")

        except Exception as e:
            print(f"\n[ERROR] {e}")
            continue


def single_query_mode(orchestrator: PipelineOrchestrator, entity: str, year: int):
    """Run a single query"""
    result = orchestrator.run_audit(entity, year)

    print("\n" + "=" * 80)
    print(f"REPORT: {entity} ({year})")
    print("=" * 80)
    print(result['report'])

    print("\n" + "=" * 80)
    print(f"VERIFICATION: {result['status']}")
    print("=" * 80)
    print(f"Reasoning: {result['verification']['reasoning']}")
    if result['verification']['issues_found']:
        print(f"Issues: {result['verification']['issues_found']}")


def main():
    parser = argparse.ArgumentParser(
        description="Unified temporal graph agent pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete pipeline
  python run_pipeline.py --config configs/pipeline_amazon_reviews.yaml

  # Query existing graph
  python run_pipeline.py --config configs/pipeline_query_only.yaml --skip-ingestion

  # Interactive mode
  python run_pipeline.py --config configs/pipeline_query_only.yaml --interactive

  # Single query
  python run_pipeline.py --config configs/pipeline_query_only.yaml --entity Apple --year 2021
        """
    )

    parser.add_argument('--config', required=True, help='Path to pipeline config YAML')
    parser.add_argument('--skip-ingestion', action='store_true',
                       help='Skip data ingestion, load existing graph')
    parser.add_argument('--interactive', action='store_true',
                       help='Run in interactive mode')
    parser.add_argument('--entity', help='Entity to query (for single query mode)')
    parser.add_argument('--year', type=int, help='Year to analyze (for single query mode)')

    args = parser.parse_args()

    # Validate arguments
    if not Path(args.config).exists():
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)

    if args.entity and not args.year:
        print("Error: --year required when using --entity")
        sys.exit(1)

    if args.year and not args.entity:
        print("Error: --entity required when using --year")
        sys.exit(1)

    try:
        # Load configuration
        print(f"Loading configuration from {args.config}...")
        config = PipelineConfig.from_yaml(args.config)

        # Create orchestrator
        orchestrator = PipelineOrchestrator(config)

        # Initialize pipeline
        orchestrator.initialize(skip_ingestion=args.skip_ingestion)

        # Choose mode
        if args.entity and args.year:
            # Single query mode
            single_query_mode(orchestrator, args.entity, args.year)
        elif args.interactive:
            # Interactive mode
            interactive_mode(orchestrator)
        else:
            # Show usage
            entities = orchestrator.get_available_entities()
            print(f"\nPipeline ready! Found {len(entities)} entities.")
            print(f"Examples: {entities[:10]}")
            print("\nRun with --interactive for interactive mode")
            print("Or use --entity and --year for single query")

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
