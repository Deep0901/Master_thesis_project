"""
Statistical Analysis for Fuzzy System Calibration

This module analyzes collected OGD metadata to derive data-driven
parameters for the fuzzy membership functions.

Research Purpose:
- Analyze actual distributions of recency, completeness, and resources
- Derive percentile-based breakpoints for membership functions
- Ensure fuzzy system is calibrated to REAL Swiss OGD data
- Support RQ1: How can fuzzy logic model vagueness in OGD metadata?

Author: Deep Shukla
Thesis: Improving Access to Swiss OGD through Fuzzy HCIR
University of Fribourg, Human-IST Institute
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from collections import Counter
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DistributionAnalysis:
    """Results of distribution analysis for a single variable."""
    variable_name: str
    n_samples: int
    n_valid: int
    min_value: float
    max_value: float
    mean: float
    median: float
    std: float
    percentiles: Dict[int, float]  # {10: val, 25: val, 50: val, 75: val, 90: val}
    histogram_bins: List[float]
    histogram_counts: List[int]
    
    # Recommended membership function breakpoints
    recommended_breakpoints: Dict[str, List[float]]


@dataclass
class CalibrationParameters:
    """
    Calibrated parameters for the fuzzy system.
    Based on empirical analysis of Swiss OGD portal data.
    """
    # Recency (days since last update)
    recency_very_recent: Tuple[float, float, float]  # (0, 0, a)
    recency_recent: Tuple[float, float, float, float]  # (a, b, c, d) trapezoidal
    recency_moderate: Tuple[float, float, float, float]
    recency_old: Tuple[float, float, float, float]
    recency_very_old: Tuple[float, float, float]  # (a, b, inf)
    
    # Completeness (0-1 score)
    completeness_low: Tuple[float, float, float]
    completeness_medium: Tuple[float, float, float, float]
    completeness_high: Tuple[float, float, float]
    
    # Resource availability (count)
    resources_limited: Tuple[float, float, float]
    resources_moderate: Tuple[float, float, float, float]
    resources_rich: Tuple[float, float, float]
    
    # Metadata for reproducibility
    data_source: str
    n_datasets_analyzed: int
    analysis_timestamp: str
    

class OGDStatisticalAnalyzer:
    """
    Analyzes Swiss OGD metadata to derive calibration parameters.
    
    This is a critical component for ensuring the fuzzy system
    is grounded in empirical data rather than arbitrary assumptions.
    """
    
    def __init__(self, data_dir: str = "data/raw"):
        """
        Initialize analyzer with data directory.
        
        Args:
            data_dir: Directory containing collected metadata
        """
        self.data_dir = Path(data_dir)
        self.records = []
    
    def load_latest_collection(self) -> int:
        """
        Load the most recent data collection.
        
        Returns:
            Number of records loaded
        """
        json_files = list(self.data_dir.glob("ogd_metadata_*.json"))
        if not json_files:
            raise FileNotFoundError(f"No data files found in {self.data_dir}")
        
        latest_file = max(json_files, key=lambda p: p.stat().st_mtime)
        logger.info(f"Loading data from: {latest_file}")
        
        with open(latest_file, 'r', encoding='utf-8') as f:
            self.records = json.load(f)
        
        logger.info(f"Loaded {len(self.records)} records")
        return len(self.records)
    
    def load_from_file(self, filepath: str) -> int:
        """Load data from specific file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            self.records = json.load(f)
        return len(self.records)
    
    def analyze_recency_distribution(self) -> DistributionAnalysis:
        """
        Analyze distribution of dataset recency (days since modified/created).
        
        This analysis informs the calibration of recency membership functions.
        Uses metadata_created when metadata_modified values are too clustered.
        """
        from datetime import datetime
        
        values_modified = []
        values_created = []
        now = datetime.now()
        
        def parse_date(date_str):
            """Parse ISO date string (Python 3.6 compatible)."""
            try:
                if 'T' in date_str:
                    date_part = date_str.split('T')[0]
                    time_part = date_str.split('T')[1][:8]
                    return datetime.strptime(f"{date_part}T{time_part}", "%Y-%m-%dT%H:%M:%S")
                else:
                    return datetime.strptime(date_str[:10], "%Y-%m-%d")
            except:
                return None
        
        for r in self.records:
            # Parse modification date
            if r.get('metadata_modified'):
                modified = parse_date(r['metadata_modified'])
                if modified:
                    days = (now - modified).days
                    if days >= 0:
                        values_modified.append(days)
            
            # Parse creation date
            if r.get('metadata_created'):
                created = parse_date(r['metadata_created'])
                if created:
                    days = (now - created).days
                    if days >= 0:
                        values_created.append(days)
        
        # Use creation dates if modification dates are too clustered
        values = values_modified
        used_field = 'metadata_modified'
        
        if values_modified:
            modified_std = np.std(values_modified)
            if modified_std < 10 and values_created:  # Very little variance
                values = values_created
                used_field = 'metadata_created'
                logger.info(f"Using {used_field} due to low variance in metadata_modified")
        elif values_created:
            values = values_created
            used_field = 'metadata_created'
        
        if not values:
            raise ValueError("No valid recency values found")
        
        arr = np.array(values)
        
        # Compute percentiles for membership function calibration
        percentiles = {
            p: float(np.percentile(arr, p))
            for p in [5, 10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90, 95]
        }
        
        # Compute histogram
        # Use log scale for better visualization of skewed distribution
        hist_counts, hist_bins = np.histogram(arr, bins=50)
        
        # Recommend breakpoints based on percentiles
        # Very Recent: Top 10% most recently updated
        # Recent: 10-30%
        # Moderate: 30-70%
        # Old: 70-90%
        # Very Old: Bottom 10%
        
        recommendations = {
            'very_recent': [0, 0, percentiles[10]],  # Triangle at 0
            'recent': [percentiles[5], percentiles[20], percentiles[40], percentiles[50]],  # Trapezoid
            'moderate': [percentiles[30], percentiles[50], percentiles[70], percentiles[80]],
            'old': [percentiles[60], percentiles[75], percentiles[90], percentiles[95]],
            'very_old': [percentiles[80], percentiles[90], float(arr.max()) * 1.2]  # Open-ended
        }
        
        return DistributionAnalysis(
            variable_name='recency_days',
            n_samples=len(self.records),
            n_valid=len(values),
            min_value=float(arr.min()),
            max_value=float(arr.max()),
            mean=float(arr.mean()),
            median=float(np.median(arr)),
            std=float(arr.std()),
            percentiles=percentiles,
            histogram_bins=hist_bins.tolist(),
            histogram_counts=hist_counts.tolist(),
            recommended_breakpoints=recommendations
        )
    
    def analyze_completeness_distribution(self) -> DistributionAnalysis:
        """
        Analyze distribution of metadata completeness scores.
        
        Completeness is scored 0-1 based on DCAT-AP CH field population.
        """
        values = [
            r['completeness_score'] 
            for r in self.records 
            if r.get('completeness_score') is not None
        ]
        
        if not values:
            raise ValueError("No valid completeness values found")
        
        arr = np.array(values)
        
        percentiles = {
            p: float(np.percentile(arr, p))
            for p in [10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90]
        }
        
        hist_counts, hist_bins = np.histogram(arr, bins=20, range=(0, 1))
        
        # Recommendations for completeness
        # Low: Bottom 25%
        # Medium: 25-75%
        # High: Top 25%
        
        recommendations = {
            'low': [0, 0, percentiles[30]],  # Triangle starting at 0
            'medium': [percentiles[20], percentiles[40], percentiles[60], percentiles[80]],  # Trapezoid
            'high': [percentiles[70], percentiles[90], 1.0]  # Triangle ending at 1
        }
        
        return DistributionAnalysis(
            variable_name='completeness',
            n_samples=len(self.records),
            n_valid=len(values),
            min_value=float(arr.min()),
            max_value=float(arr.max()),
            mean=float(arr.mean()),
            median=float(np.median(arr)),
            std=float(arr.std()),
            percentiles=percentiles,
            histogram_bins=hist_bins.tolist(),
            histogram_counts=hist_counts.tolist(),
            recommended_breakpoints=recommendations
        )
    
    def analyze_resources_distribution(self) -> DistributionAnalysis:
        """
        Analyze distribution of resource counts per dataset.
        
        Resource availability indicates how many downloadable/accessible
        data formats are provided.
        """
        values = [
            r['num_resources'] 
            for r in self.records 
            if r.get('num_resources') is not None
        ]
        
        if not values:
            raise ValueError("No valid resource count values found")
        
        arr = np.array(values)
        
        percentiles = {
            p: float(np.percentile(arr, p))
            for p in [10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90, 95]
        }
        
        # Count distribution (most datasets have 1-5 resources)
        max_for_hist = min(int(np.percentile(arr, 99)), 50)  # Cap at 99th percentile
        hist_counts, hist_bins = np.histogram(
            arr[arr <= max_for_hist], 
            bins=min(max_for_hist, 25)
        )
        
        # Recommendations
        recommendations = {
            'limited': [0, 0, percentiles[30]],  # 1-2 resources typically
            'moderate': [percentiles[20], percentiles[40], percentiles[60], percentiles[80]],
            'rich': [percentiles[70], percentiles[90], float(arr.max()) * 1.1]
        }
        
        return DistributionAnalysis(
            variable_name='num_resources',
            n_samples=len(self.records),
            n_valid=len(values),
            min_value=float(arr.min()),
            max_value=float(arr.max()),
            mean=float(arr.mean()),
            median=float(np.median(arr)),
            std=float(arr.std()),
            percentiles=percentiles,
            histogram_bins=hist_bins.tolist(),
            histogram_counts=hist_counts.tolist(),
            recommended_breakpoints=recommendations
        )
    
    def analyze_format_distribution(self) -> Dict:
        """
        Analyze distribution of data formats across datasets.
        
        Understanding format distribution helps weight format preferences.
        """
        all_formats = []
        for r in self.records:
            formats = r.get('resource_formats', '')
            if isinstance(formats, str):
                all_formats.extend([f.strip() for f in formats.split(',') if f.strip()])
            elif isinstance(formats, list):
                all_formats.extend(formats)
        
        format_counts = Counter(all_formats)
        
        # Classify formats
        machine_readable = {'CSV', 'JSON', 'XML', 'RDF', 'SPARQL', 'API', 'GeoJSON'}
        open_formats = {'CSV', 'JSON', 'XML', 'ODS', 'TXT', 'HTML'}
        
        analysis = {
            'total_resources': len(all_formats),
            'unique_formats': len(format_counts),
            'top_formats': dict(format_counts.most_common(20)),
            'machine_readable_pct': sum(
                format_counts.get(f, 0) for f in machine_readable
            ) / len(all_formats) if all_formats else 0,
            'open_format_pct': sum(
                format_counts.get(f, 0) for f in open_formats
            ) / len(all_formats) if all_formats else 0
        }
        
        return analysis
    
    def analyze_organization_distribution(self) -> Dict:
        """
        Analyze which organizations publish the most data.
        
        Helps understand authority/reliability scoring.
        """
        org_counts = Counter(
            r.get('organization_name', 'unknown')
            for r in self.records
        )
        
        return {
            'total_organizations': len(org_counts),
            'top_publishers': dict(org_counts.most_common(20)),
            'single_dataset_orgs': sum(1 for c in org_counts.values() if c == 1),
            'prolific_orgs_10plus': sum(1 for c in org_counts.values() if c >= 10)
        }
    
    def generate_calibration_parameters(self) -> CalibrationParameters:
        """
        Generate calibrated fuzzy system parameters from analysis.
        
        These parameters should REPLACE the arbitrary defaults in the
        membership_functions.py file.
        """
        from datetime import datetime
        
        recency = self.analyze_recency_distribution()
        completeness = self.analyze_completeness_distribution()
        resources = self.analyze_resources_distribution()
        
        return CalibrationParameters(
            # Recency breakpoints (in days)
            recency_very_recent=tuple(recency.recommended_breakpoints['very_recent']),
            recency_recent=tuple(recency.recommended_breakpoints['recent']),
            recency_moderate=tuple(recency.recommended_breakpoints['moderate']),
            recency_old=tuple(recency.recommended_breakpoints['old']),
            recency_very_old=tuple(recency.recommended_breakpoints['very_old']),
            
            # Completeness breakpoints (0-1 scale)
            completeness_low=tuple(completeness.recommended_breakpoints['low']),
            completeness_medium=tuple(completeness.recommended_breakpoints['medium']),
            completeness_high=tuple(completeness.recommended_breakpoints['high']),
            
            # Resource count breakpoints
            resources_limited=tuple(resources.recommended_breakpoints['limited']),
            resources_moderate=tuple(resources.recommended_breakpoints['moderate']),
            resources_rich=tuple(resources.recommended_breakpoints['rich']),
            
            # Metadata
            data_source='opendata.swiss',
            n_datasets_analyzed=len(self.records),
            analysis_timestamp=datetime.now().isoformat()
        )
    
    def generate_full_report(self) -> Dict:
        """
        Generate comprehensive statistical report.
        
        This report should be included in the thesis methodology chapter.
        """
        report = {
            'summary': {
                'total_datasets': len(self.records),
                'analysis_date': None  # Set by caller
            }
        }
        
        try:
            recency = self.analyze_recency_distribution()
            report['recency'] = {
                'n_valid': recency.n_valid,
                'mean_days': recency.mean,
                'median_days': recency.median,
                'std_days': recency.std,
                'min_days': recency.min_value,
                'max_days': recency.max_value,
                'percentiles': recency.percentiles,
                'recommended_mf': recency.recommended_breakpoints
            }
        except Exception as e:
            report['recency'] = {'error': str(e)}
        
        try:
            completeness = self.analyze_completeness_distribution()
            report['completeness'] = {
                'n_valid': completeness.n_valid,
                'mean_score': completeness.mean,
                'median_score': completeness.median,
                'min_score': completeness.min_value,
                'max_score': completeness.max_value,
                'percentiles': completeness.percentiles,
                'recommended_mf': completeness.recommended_breakpoints
            }
        except Exception as e:
            report['completeness'] = {'error': str(e)}
        
        try:
            resources = self.analyze_resources_distribution()
            report['resources'] = {
                'n_valid': resources.n_valid,
                'mean_count': resources.mean,
                'median_count': resources.median,
                'min_count': resources.min_value,
                'max_count': resources.max_value,
                'percentiles': resources.percentiles,
                'recommended_mf': resources.recommended_breakpoints
            }
        except Exception as e:
            report['resources'] = {'error': str(e)}
        
        try:
            report['formats'] = self.analyze_format_distribution()
        except Exception as e:
            report['formats'] = {'error': str(e)}
        
        try:
            report['organizations'] = self.analyze_organization_distribution()
        except Exception as e:
            report['organizations'] = {'error': str(e)}
        
        return report
    
    def save_report(self, filepath: str):
        """Save analysis report to JSON file."""
        from datetime import datetime
        
        report = self.generate_full_report()
        report['summary']['analysis_date'] = datetime.now().isoformat()
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Report saved to: {filepath}")


def main():
    """Run statistical analysis on collected data."""
    print("=" * 70)
    print("SWISS OGD STATISTICAL ANALYSIS FOR FUZZY CALIBRATION")
    print("Master Thesis Research - University of Fribourg")
    print("=" * 70)
    
    analyzer = OGDStatisticalAnalyzer(data_dir="data/raw")
    
    try:
        n_records = analyzer.load_latest_collection()
        print(f"\nLoaded {n_records} dataset records for analysis")
    except FileNotFoundError as e:
        print(f"\nNo data found. Please run the collector first:")
        print("  python -m code.data_collection.comprehensive_collector")
        return
    
    print("\n" + "-" * 70)
    print("RECENCY ANALYSIS (Days Since Last Update)")
    print("-" * 70)
    
    recency = analyzer.analyze_recency_distribution()
    print(f"  Valid samples: {recency.n_valid}")
    print(f"  Mean: {recency.mean:.1f} days")
    print(f"  Median: {recency.median:.1f} days")
    print(f"  Std Dev: {recency.std:.1f} days")
    print(f"  Range: {recency.min_value:.0f} - {recency.max_value:.0f} days")
    print(f"\n  Key Percentiles:")
    for p in [10, 25, 50, 75, 90]:
        print(f"    {p}th: {recency.percentiles[p]:.0f} days")
    print(f"\n  Recommended Membership Functions:")
    for term, params in recency.recommended_breakpoints.items():
        print(f"    {term}: {[round(x, 1) for x in params]}")
    
    print("\n" + "-" * 70)
    print("COMPLETENESS ANALYSIS (Metadata Quality Score)")
    print("-" * 70)
    
    completeness = analyzer.analyze_completeness_distribution()
    print(f"  Valid samples: {completeness.n_valid}")
    print(f"  Mean: {completeness.mean:.2%}")
    print(f"  Median: {completeness.median:.2%}")
    print(f"  Range: {completeness.min_value:.2%} - {completeness.max_value:.2%}")
    print(f"\n  Key Percentiles:")
    for p in [10, 25, 50, 75, 90]:
        print(f"    {p}th: {completeness.percentiles[p]:.2%}")
    print(f"\n  Recommended Membership Functions:")
    for term, params in completeness.recommended_breakpoints.items():
        print(f"    {term}: {[round(x, 3) for x in params]}")
    
    print("\n" + "-" * 70)
    print("RESOURCE AVAILABILITY ANALYSIS")
    print("-" * 70)
    
    resources = analyzer.analyze_resources_distribution()
    print(f"  Valid samples: {resources.n_valid}")
    print(f"  Mean: {resources.mean:.1f} resources per dataset")
    print(f"  Median: {resources.median:.0f}")
    print(f"  Range: {resources.min_value:.0f} - {resources.max_value:.0f}")
    print(f"\n  Key Percentiles:")
    for p in [10, 25, 50, 75, 90]:
        print(f"    {p}th: {resources.percentiles[p]:.0f}")
    
    print("\n" + "-" * 70)
    print("FORMAT DISTRIBUTION")
    print("-" * 70)
    
    formats = analyzer.analyze_format_distribution()
    print(f"  Total resources: {formats['total_resources']}")
    print(f"  Unique formats: {formats['unique_formats']}")
    print(f"  Machine-readable: {formats['machine_readable_pct']:.1%}")
    print(f"\n  Top 10 formats:")
    for fmt, count in list(formats['top_formats'].items())[:10]:
        print(f"    {fmt}: {count}")
    
    # Save full report
    report_path = "analytics/statistical_analysis_report.json"
    Path("analytics").mkdir(exist_ok=True)
    analyzer.save_report(report_path)
    print(f"\n  Full report saved to: {report_path}")
    
    # Generate calibration parameters
    print("\n" + "=" * 70)
    print("CALIBRATION PARAMETERS FOR FUZZY SYSTEM")
    print("=" * 70)
    
    params = analyzer.generate_calibration_parameters()
    print(f"\nThese parameters should be used in membership_functions.py:")
    print(f"\n# Recency (days)")
    print(f"RECENCY_VERY_RECENT = {params.recency_very_recent}")
    print(f"RECENCY_RECENT = {params.recency_recent}")
    print(f"RECENCY_MODERATE = {params.recency_moderate}")
    print(f"RECENCY_OLD = {params.recency_old}")
    print(f"RECENCY_VERY_OLD = {params.recency_very_old}")
    
    print(f"\n# Completeness (0-1)")
    print(f"COMPLETENESS_LOW = {params.completeness_low}")
    print(f"COMPLETENESS_MEDIUM = {params.completeness_medium}")
    print(f"COMPLETENESS_HIGH = {params.completeness_high}")
    
    print(f"\n# Resources (count)")
    print(f"RESOURCES_LIMITED = {params.resources_limited}")
    print(f"RESOURCES_MODERATE = {params.resources_moderate}")
    print(f"RESOURCES_RICH = {params.resources_rich}")
    
    print(f"\nBased on analysis of {params.n_datasets_analyzed} datasets")
    print(f"Analysis timestamp: {params.analysis_timestamp}")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
