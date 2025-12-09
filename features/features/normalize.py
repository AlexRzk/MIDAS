"""
Production-ready normalization layer for MIDAS features.

Implements:
- StandardScaler (z-score) for normally distributed features
- RobustScaler (median + IQR) with log1p for heavy-tailed features  
- MinMaxScaler for bounded features
- Proper train/test separation to prevent data leakage
- Scaler persistence for inference consistency

Critical: Normalization is CAUSAL - scalers fitted ONLY on training data.
"""
import polars as pl
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import joblib
import json
import structlog

from .feature_schema import (
    STANDARD_FEATURES,
    ROBUST_FEATURES,
    MINMAX_FEATURES,
    RAW_PASSTHROUGH_FEATURES,
    classify_feature,
    validate_dataframe_columns,
)

logger = structlog.get_logger()


# ============================================================================
# Scaler Statistics Storage
# ============================================================================

@dataclass
class ScalerStats:
    """Statistics for a single feature's scaler."""
    feature_name: str
    scaler_type: str  # "standard", "robust", "minmax"
    
    # Standard scaler
    mean: Optional[float] = None
    std: Optional[float] = None
    
    # Robust scaler
    median: Optional[float] = None
    iqr: Optional[float] = None
    q25: Optional[float] = None
    q75: Optional[float] = None
    
    # MinMax scaler
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    
    # Metadata
    n_samples: Optional[int] = None
    n_nulls: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ScalerStats":
        return cls(**data)


@dataclass
class NormalizationManifest:
    """Manifest of all fitted scalers and normalization metadata."""
    version: str = "1.0.0"
    created_at: Optional[str] = None
    n_samples_fitted: int = 0
    scalers: Dict[str, ScalerStats] = None
    
    def __post_init__(self):
        if self.scalers is None:
            self.scalers = {}
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "created_at": self.created_at,
            "n_samples_fitted": self.n_samples_fitted,
            "scalers": {k: v.to_dict() for k, v in self.scalers.items()},
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NormalizationManifest":
        scalers = {
            k: ScalerStats.from_dict(v) 
            for k, v in data.get("scalers", {}).items()
        }
        return cls(
            version=data.get("version", "1.0.0"),
            created_at=data.get("created_at"),
            n_samples_fitted=data.get("n_samples_fitted", 0),
            scalers=scalers,
        )


# ============================================================================
# Normalization Functions
# ============================================================================

def fit_standard_scaler(series: pl.Series) -> ScalerStats:
    """
    Fit standard scaler (z-score normalization).
    
    Formula: (x - mean) / std
    """
    # Drop nulls for fitting
    clean_series = series.drop_nulls()
    
    if len(clean_series) == 0:
        logger.warning("empty_series_for_standard_scaler", feature=series.name)
        return ScalerStats(
            feature_name=series.name,
            scaler_type="standard",
            mean=0.0,
            std=1.0,
            n_samples=0,
            n_nulls=len(series),
        )
    
    mean = float(clean_series.mean())
    std = float(clean_series.std())
    
    # Prevent division by zero
    if std == 0 or np.isnan(std):
        logger.warning("zero_std_detected", feature=series.name)
        std = 1.0
    
    return ScalerStats(
        feature_name=series.name,
        scaler_type="standard",
        mean=mean,
        std=std,
        n_samples=len(clean_series),
        n_nulls=len(series) - len(clean_series),
    )


def apply_standard_scaler(series: pl.Series, stats: ScalerStats) -> pl.Series:
    """Apply standard scaler transformation."""
    return (series - stats.mean) / stats.std


def fit_robust_scaler(series: pl.Series, apply_log1p: bool = True) -> ScalerStats:
    """
    Fit robust scaler (median + IQR normalization).
    
    Formula: (log1p(x) - median) / IQR
    Note: log1p applied to handle heavy tails and ensure positivity
    """
    # Drop nulls for fitting
    clean_series = series.drop_nulls()
    
    if len(clean_series) == 0:
        logger.warning("empty_series_for_robust_scaler", feature=series.name)
        return ScalerStats(
            feature_name=series.name,
            scaler_type="robust",
            median=0.0,
            iqr=1.0,
            q25=0.0,
            q75=1.0,
            n_samples=0,
            n_nulls=len(series),
        )
    
    # Apply log1p transformation for robust features (sizes, volumes)
    if apply_log1p:
        transformed = clean_series.log1p()
    else:
        transformed = clean_series
    
    # Compute robust statistics
    q25 = float(transformed.quantile(0.25))
    q75 = float(transformed.quantile(0.75))
    median = float(transformed.median())
    iqr = q75 - q25
    
    # Prevent division by zero
    if iqr == 0 or np.isnan(iqr):
        logger.warning("zero_iqr_detected", feature=series.name)
        iqr = 1.0
    
    return ScalerStats(
        feature_name=series.name,
        scaler_type="robust",
        median=median,
        iqr=iqr,
        q25=q25,
        q75=q75,
        n_samples=len(clean_series),
        n_nulls=len(series) - len(clean_series),
    )


def apply_robust_scaler(
    series: pl.Series, 
    stats: ScalerStats,
    apply_log1p: bool = True
) -> pl.Series:
    """Apply robust scaler transformation."""
    if apply_log1p:
        transformed = series.log1p()
    else:
        transformed = series
    
    return (transformed - stats.median) / stats.iqr


def fit_minmax_scaler(series: pl.Series) -> ScalerStats:
    """
    Fit min-max scaler (0-1 normalization).
    
    Formula: (x - min) / (max - min)
    """
    clean_series = series.drop_nulls()
    
    if len(clean_series) == 0:
        logger.warning("empty_series_for_minmax_scaler", feature=series.name)
        return ScalerStats(
            feature_name=series.name,
            scaler_type="minmax",
            min_val=0.0,
            max_val=1.0,
            n_samples=0,
            n_nulls=len(series),
        )
    
    min_val = float(clean_series.min())
    max_val = float(clean_series.max())
    
    # Prevent division by zero
    if max_val == min_val:
        logger.warning("constant_feature_detected", feature=series.name)
        max_val = min_val + 1.0
    
    return ScalerStats(
        feature_name=series.name,
        scaler_type="minmax",
        min_val=min_val,
        max_val=max_val,
        n_samples=len(clean_series),
        n_nulls=len(series) - len(clean_series),
    )


def apply_minmax_scaler(series: pl.Series, stats: ScalerStats) -> pl.Series:
    """Apply min-max scaler transformation."""
    range_val = stats.max_val - stats.min_val
    if range_val == 0:
        range_val = 1.0
    return (series - stats.min_val) / range_val


# ============================================================================
# Main Normalizer Class
# ============================================================================

class FeatureNormalizer:
    """
    Production normalizer for MIDAS features.
    
    Handles:
    - Multi-type normalization (standard, robust, minmax)
    - Proper train/test separation (fit on train, transform on all)
    - Scaler persistence to disk
    - Validation and safety checks
    """
    
    def __init__(self, scaler_dir: Path):
        """
        Args:
            scaler_dir: Directory to save/load fitted scalers
        """
        self.scaler_dir = Path(scaler_dir)
        self.scaler_dir.mkdir(parents=True, exist_ok=True)
        
        self.manifest: Optional[NormalizationManifest] = None
        self.is_fitted = False
    
    def fit(self, df: pl.DataFrame, is_training: bool = True) -> "FeatureNormalizer":
        """
        Fit all scalers on training data.
        
        Args:
            df: Training DataFrame
            is_training: Must be True for fitting
            
        Returns:
            Self for chaining
        """
        if not is_training:
            raise ValueError("fit() should only be called on training data")
        
        logger.info("fitting_normalizers", n_rows=len(df), n_cols=len(df.columns))
        
        # Classify columns
        column_classification = validate_dataframe_columns(df.columns)
        
        # Warn about unknown columns
        if column_classification["unknown"]:
            logger.warning(
                "unknown_features_detected",
                features=column_classification["unknown"][:10],
                count=len(column_classification["unknown"]),
            )
        
        # Create manifest
        from datetime import datetime, timezone
        self.manifest = NormalizationManifest(
            created_at=datetime.now(timezone.utc).isoformat(),
            n_samples_fitted=len(df),
        )
        
        # Fit standard scalers
        for feat in column_classification["standard"]:
            if feat in df.columns:
                stats = fit_standard_scaler(df[feat])
                self.manifest.scalers[feat] = stats
                logger.debug("fitted_standard_scaler", feature=feat, mean=stats.mean, std=stats.std)
        
        # Fit robust scalers
        for feat in column_classification["robust"]:
            if feat in df.columns:
                stats = fit_robust_scaler(df[feat], apply_log1p=True)
                self.manifest.scalers[feat] = stats
                logger.debug("fitted_robust_scaler", feature=feat, median=stats.median, iqr=stats.iqr)
        
        # Fit minmax scalers
        for feat in column_classification["minmax"]:
            if feat in df.columns:
                stats = fit_minmax_scaler(df[feat])
                self.manifest.scalers[feat] = stats
                logger.debug("fitted_minmax_scaler", feature=feat, min=stats.min_val, max=stats.max_val)
        
        self.is_fitted = True
        
        logger.info(
            "normalization_fitting_complete",
            n_scalers=len(self.manifest.scalers),
            standard=len(column_classification["standard"]),
            robust=len(column_classification["robust"]),
            minmax=len(column_classification["minmax"]),
        )
        
        return self
    
    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Apply fitted scalers to data.
        
        Args:
            df: DataFrame to normalize
            
        Returns:
            Normalized DataFrame
        """
        if not self.is_fitted:
            raise RuntimeError("Normalizer must be fitted before transform")
        
        logger.info("transforming_features", n_rows=len(df), n_cols=len(df.columns))
        
        # Work with lazy frame for efficiency
        df_lazy = df.lazy()
        
        expressions = []
        
        for col in df.columns:
            if col not in self.manifest.scalers:
                # Passthrough columns (ts, raw prices, etc.)
                expressions.append(pl.col(col))
                continue
            
            stats = self.manifest.scalers[col]
            
            if stats.scaler_type == "standard":
                # Z-score normalization
                expr = ((pl.col(col) - stats.mean) / stats.std).alias(col)
            
            elif stats.scaler_type == "robust":
                # Robust scaling with log1p
                expr = (
                    ((pl.col(col).log1p() - stats.median) / stats.iqr).alias(col)
                )
            
            elif stats.scaler_type == "minmax":
                # Min-max normalization
                range_val = stats.max_val - stats.min_val
                if range_val == 0:
                    range_val = 1.0
                expr = ((pl.col(col) - stats.min_val) / range_val).alias(col)
            
            else:
                # Unknown scaler type - passthrough
                expr = pl.col(col)
            
            expressions.append(expr)
        
        # Execute transformation
        df_normalized = df_lazy.select(expressions).collect()
        
        logger.info("transformation_complete", n_rows=len(df_normalized))
        
        return df_normalized
    
    def fit_transform(self, df: pl.DataFrame, is_training: bool = True) -> pl.DataFrame:
        """Fit scalers and transform in one step."""
        self.fit(df, is_training=is_training)
        return self.transform(df)
    
    def save(self, filename: str = "normalization_manifest.json"):
        """Save fitted scalers to disk."""
        if not self.is_fitted:
            raise RuntimeError("Cannot save unfitted normalizer")
        
        filepath = self.scaler_dir / filename
        
        with open(filepath, "w") as f:
            json.dump(self.manifest.to_dict(), f, indent=2)
        
        logger.info("saved_normalizer_manifest", path=str(filepath))
    
    @classmethod
    def load(cls, scaler_dir: Path, filename: str = "normalization_manifest.json") -> "FeatureNormalizer":
        """Load fitted scalers from disk."""
        normalizer = cls(scaler_dir)
        
        filepath = normalizer.scaler_dir / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"Manifest not found: {filepath}")
        
        with open(filepath, "r") as f:
            data = json.load(f)
        
        normalizer.manifest = NormalizationManifest.from_dict(data)
        normalizer.is_fitted = True
        
        logger.info(
            "loaded_normalizer_manifest",
            path=str(filepath),
            n_scalers=len(normalizer.manifest.scalers),
        )
        
        return normalizer
    
    def validate_normalized_data(self, df: pl.DataFrame) -> Dict[str, Any]:
        """
        Validate that normalized data has expected properties.
        
        Returns:
            Validation report with checks and warnings
        """
        report = {
            "valid": True,
            "warnings": [],
            "checks": {},
        }
        
        for col in df.columns:
            if col not in self.manifest.scalers:
                continue
            
            stats = self.manifest.scalers[col]
            series = df[col].drop_nulls()
            
            if len(series) == 0:
                continue
            
            col_report = {}
            
            if stats.scaler_type == "standard":
                # Check z-score properties
                mean = float(series.mean())
                std = float(series.std())
                
                col_report["mean"] = mean
                col_report["std"] = std
                
                # Should be close to 0 mean, 1 std
                if abs(mean) > 0.1:
                    report["warnings"].append(f"{col}: mean={mean:.3f} (expected ≈0)")
                    report["valid"] = False
                
                if abs(std - 1.0) > 0.2:
                    report["warnings"].append(f"{col}: std={std:.3f} (expected ≈1)")
            
            elif stats.scaler_type == "robust":
                # Check robust scaling - should be in reasonable range
                min_val = float(series.min())
                max_val = float(series.max())
                
                col_report["min"] = min_val
                col_report["max"] = max_val
                
                # Reasonable range for robust scaling
                if min_val < -20 or max_val > 20:
                    report["warnings"].append(
                        f"{col}: range=[{min_val:.1f}, {max_val:.1f}] (expected ≈[-20, 20])"
                    )
            
            elif stats.scaler_type == "minmax":
                # Check min-max properties
                min_val = float(series.min())
                max_val = float(series.max())
                
                col_report["min"] = min_val
                col_report["max"] = max_val
                
                # Should be in [0, 1]
                if min_val < -0.01 or max_val > 1.01:
                    report["warnings"].append(
                        f"{col}: range=[{min_val:.3f}, {max_val:.3f}] (expected [0, 1])"
                    )
                    report["valid"] = False
            
            report["checks"][col] = col_report
        
        return report


# ============================================================================
# Utility Functions
# ============================================================================

def normalize_features_pipeline(
    df: pl.DataFrame,
    scaler_dir: Path,
    is_training: bool = True,
    save_scalers: bool = True,
) -> pl.DataFrame:
    """
    Convenience function for normalizing features in the pipeline.
    
    Args:
        df: Input DataFrame
        scaler_dir: Directory for scaler storage
        is_training: Whether this is training data (fit scalers)
        save_scalers: Whether to save scalers after fitting
        
    Returns:
        Normalized DataFrame
    """
    normalizer = FeatureNormalizer(scaler_dir)
    
    if is_training:
        df_norm = normalizer.fit_transform(df, is_training=True)
        if save_scalers:
            normalizer.save()
    else:
        normalizer = FeatureNormalizer.load(scaler_dir)
        df_norm = normalizer.transform(df)
    
    return df_norm


if __name__ == "__main__":
    # Test normalization
    import numpy as np
    
    # Create test data
    np.random.seed(42)
    n_samples = 10000
    
    df = pl.DataFrame({
        "ts": np.arange(n_samples) * 1000,
        "ofi": np.random.randn(n_samples) * 10,
        "imbalance": np.random.randn(n_samples) * 0.5,
        "bid_sz_01": np.abs(np.random.randn(n_samples) * 1000),
        "ask_sz_01": np.abs(np.random.randn(n_samples) * 1000),
    })
    
    print("Original data:")
    print(df.describe())
    
    # Fit and transform
    scaler_dir = Path("/tmp/test_scalers")
    normalizer = FeatureNormalizer(scaler_dir)
    df_norm = normalizer.fit_transform(df, is_training=True)
    normalizer.save()
    
    print("\nNormalized data:")
    print(df_norm.describe())
    
    # Validate
    report = normalizer.validate_normalized_data(df_norm)
    print(f"\nValidation: {'PASSED' if report['valid'] else 'FAILED'}")
    if report["warnings"]:
        print("Warnings:")
        for w in report["warnings"]:
            print(f"  - {w}")
