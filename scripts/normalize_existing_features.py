#!/usr/bin/env python3
"""
Standalone script to normalize existing feature files.

Use this to normalize features that were generated before normalization was added.

Usage:
    python normalize_existing_features.py
    python normalize_existing_features.py --input-dir /data/features --output-dir /data/features_normalized
"""
import argparse
from pathlib import Path
import polars as pl
import structlog

# Add features module to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from features.features.normalize import FeatureNormalizer
from features.features.feature_schema import validate_dataframe_columns

logger = structlog.get_logger()


def normalize_existing_features(
    input_dir: Path,
    output_dir: Path,
    scaler_dir: Path,
    fit_on_first: bool = True,
    overwrite: bool = False,
):
    """
    Normalize all feature files in a directory.
    
    Args:
        input_dir: Directory with raw feature parquet files
        output_dir: Directory to write normalized features
        scaler_dir: Directory to save/load scalers
        fit_on_first: If True, fit scalers on first file (training data)
        overwrite: If True, overwrite existing normalized files
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    scaler_dir = Path(scaler_dir)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all feature files
    feature_files = sorted(input_dir.glob("features_*.parquet"))
    
    if not feature_files:
        logger.error("no_feature_files_found", path=str(input_dir))
        return
    
    logger.info(
        "found_feature_files",
        count=len(feature_files),
        input_dir=str(input_dir),
        output_dir=str(output_dir),
    )
    
    # Initialize normalizer
    normalizer = FeatureNormalizer(scaler_dir)
    
    # Check if scalers already exist
    manifest_path = scaler_dir / "normalization_manifest.json"
    scalers_exist = manifest_path.exists()
    
    if scalers_exist and not fit_on_first:
        logger.info("loading_existing_scalers", path=str(manifest_path))
        normalizer = FeatureNormalizer.load(scaler_dir)
    
    # Process files
    for i, filepath in enumerate(feature_files):
        output_path = output_dir / f"normalized_{filepath.name}"
        
        # Skip if already exists and not overwriting
        if output_path.exists() and not overwrite:
            logger.info("skipping_existing_file", path=str(output_path))
            continue
        
        logger.info("processing_file", file=filepath.name, progress=f"{i+1}/{len(feature_files)}")
        
        try:
            # Read features
            df = pl.read_parquet(filepath)
            
            logger.info("loaded_dataframe", rows=len(df), columns=len(df.columns))
            
            # Validate columns
            classification = validate_dataframe_columns(df.columns)
            if classification["unknown"]:
                logger.warning(
                    "unknown_features_detected",
                    count=len(classification["unknown"]),
                    features=classification["unknown"][:5],
                )
            
            # Fit or transform
            if i == 0 and fit_on_first and not scalers_exist:
                # First file - fit scalers
                logger.info("fitting_scalers_on_first_file", file=filepath.name)
                df_normalized = normalizer.fit_transform(df, is_training=True)
                normalizer.save()
                logger.info("saved_scalers", path=str(scaler_dir))
            else:
                # Subsequent files - just transform
                df_normalized = normalizer.transform(df)
            
            # Validate normalization
            report = normalizer.validate_normalized_data(df_normalized)
            
            if not report["valid"]:
                logger.warning(
                    "normalization_validation_warnings",
                    file=filepath.name,
                    warnings=report["warnings"][:3],
                )
            else:
                logger.info("normalization_validation_passed", file=filepath.name)
            
            # Write normalized features
            df_normalized.write_parquet(
                output_path,
                compression="zstd",
                compression_level=3,
            )
            
            logger.info("wrote_normalized_file", path=str(output_path), rows=len(df_normalized))
            
        except Exception as e:
            logger.error("processing_error", file=filepath.name, error=str(e))
            continue
    
    logger.info("normalization_complete", processed=len(feature_files))


def main():
    parser = argparse.ArgumentParser(
        description="Normalize existing MIDAS feature files"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="/data/features",
        help="Directory with raw feature files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for normalized features (default: same as input)",
    )
    parser.add_argument(
        "--scaler-dir",
        type=str,
        default="/data/scalers",
        help="Directory to save/load scalers",
    )
    parser.add_argument(
        "--no-fit",
        action="store_true",
        help="Don't fit scalers, only load and transform",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing normalized files",
    )
    
    args = parser.parse_args()
    
    # Configure logging
    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(colors=True),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(20),  # INFO level
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
    )
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir
    scaler_dir = Path(args.scaler_dir)
    
    normalize_existing_features(
        input_dir=input_dir,
        output_dir=output_dir,
        scaler_dir=scaler_dir,
        fit_on_first=not args.no_fit,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
