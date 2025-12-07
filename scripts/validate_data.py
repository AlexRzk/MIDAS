#!/usr/bin/env python3
"""
Data quality validation script.

Validates Parquet files for:
- No crossed books
- No negative volumes  
- Monotonic timestamps
- Midprice stability
- Spread bounds
- Price level ordering
"""
import sys
from pathlib import Path
import polars as pl
import json

# Add processor to path
sys.path.insert(0, str(Path(__file__).parent.parent / "processor"))

from processor.validation import DataValidator, DataQualityReport


def validate_directory(data_dir: Path, pattern: str = "*.parquet") -> list[DataQualityReport]:
    """Validate all Parquet files in a directory."""
    validator = DataValidator()
    reports = []
    
    files = sorted(data_dir.glob(pattern))
    
    if not files:
        print(f"No files matching {pattern} found in {data_dir}")
        return reports
    
    print(f"Found {len(files)} files to validate")
    print("=" * 60)
    
    for filepath in files:
        print(f"\nValidating: {filepath.name}")
        
        try:
            df = pl.read_parquet(filepath)
            report = validator.validate_dataframe(df, str(filepath))
            reports.append(report)
            
            # Print summary
            status = "✓ PASS" if report.all_passed else "✗ FAIL"
            print(f"  Status: {status}")
            print(f"  Rows: {report.total_rows:,}")
            print(f"  Quality Score: {report.overall_quality_score:.2%}")
            
            # Print individual check results
            for result in report.results:
                check_status = "✓" if result.valid else "✗"
                print(f"    {check_status} {result.check_name}: {result.message}")
                
        except Exception as e:
            print(f"  ERROR: {e}")
    
    return reports


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate data quality")
    parser.add_argument("--clean", action="store_true", help="Validate clean data")
    parser.add_argument("--features", action="store_true", help="Validate feature data")
    parser.add_argument("--all", action="store_true", help="Validate all data")
    parser.add_argument("--output", "-o", type=str, help="Output JSON report file")
    
    args = parser.parse_args()
    
    if not any([args.clean, args.features, args.all]):
        args.all = True
    
    base_dir = Path(__file__).parent.parent / "data"
    all_reports = []
    
    if args.all or args.clean:
        print("\n" + "=" * 60)
        print("VALIDATING CLEAN DATA")
        print("=" * 60)
        clean_dir = base_dir / "clean"
        if clean_dir.exists():
            reports = validate_directory(clean_dir, "clean_*.parquet")
            all_reports.extend(reports)
    
    if args.all or args.features:
        print("\n" + "=" * 60)
        print("VALIDATING FEATURE DATA")
        print("=" * 60)
        features_dir = base_dir / "features"
        if features_dir.exists():
            reports = validate_directory(features_dir, "features_*.parquet")
            all_reports.extend(reports)
    
    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    
    total_files = len(all_reports)
    passed_files = sum(1 for r in all_reports if r.all_passed)
    
    print(f"Total files validated: {total_files}")
    print(f"Passed: {passed_files}")
    print(f"Failed: {total_files - passed_files}")
    
    if all_reports:
        avg_score = sum(r.overall_quality_score for r in all_reports) / len(all_reports)
        print(f"Average quality score: {avg_score:.2%}")
    
    # Output JSON report if requested
    if args.output:
        output_path = Path(args.output)
        with open(output_path, "w") as f:
            json.dump(
                {
                    "summary": {
                        "total_files": total_files,
                        "passed": passed_files,
                        "failed": total_files - passed_files,
                    },
                    "reports": [r.to_dict() for r in all_reports],
                },
                f,
                indent=2,
            )
        print(f"\nReport saved to: {output_path}")
    
    # Exit with error if any validation failed
    if passed_files < total_files:
        sys.exit(1)


if __name__ == "__main__":
    main()
