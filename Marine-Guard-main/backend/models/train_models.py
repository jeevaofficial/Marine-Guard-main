"""
Model Training Script
======================
Standalone script to train both Climate and Marine GRU models.

This script can be run to:
1. Train models for all 14 coastal districts
2. Train for a single district
3. Retrain existing models with new data

Usage:
    python train_models.py --district Chennai
    python train_models.py --all
    
Author: B.Tech AI&DS 
Date: 2026
"""

import argparse
import logging
import os
import sys
import json
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import COASTAL_DISTRICTS, MODEL_SAVE_DIR, SCALER_SAVE_DIR
from models.climate_model import train_climate_model, ClimateGRUModel
from models.marine_model import train_marine_model, MarineGRUModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def train_single_district(district_name: str, days_back: int = 30) -> dict:
    """
    Train both climate and marine models for a single district.
    
    Args:
        district_name: Name of the coastal district
        days_back: Days of historical data for training
        
    Returns:
        Dictionary with training results for both models
    """
    results = {
        "district": district_name,
        "timestamp": datetime.now().isoformat(),
        "climate_model": None,
        "marine_model": None,
        "errors": []
    }
    
    # Train climate model
    logger.info(f"\n{'='*60}")
    logger.info(f"Training CLIMATE model for {district_name}")
    logger.info(f"{'='*60}")
    
    try:
        climate_results = train_climate_model(district_name, days_back)
        results["climate_model"] = climate_results
        logger.info(f"Climate model trained successfully: RMSE={climate_results.get('rmse', 'N/A')}")
    except Exception as e:
        error_msg = f"Climate model training failed: {str(e)}"
        logger.error(error_msg)
        results["errors"].append(error_msg)
    
    # Train marine model
    logger.info(f"\n{'='*60}")
    logger.info(f"Training MARINE model for {district_name}")
    logger.info(f"{'='*60}")
    
    try:
        marine_results = train_marine_model(district_name, forecast_days=7)
        results["marine_model"] = marine_results
        logger.info(f"Marine model trained successfully: MAE={marine_results.get('val_mae', 'N/A')}")
    except Exception as e:
        error_msg = f"Marine model training failed: {str(e)}"
        logger.error(error_msg)
        results["errors"].append(error_msg)
    
    return results


def train_all_districts(days_back: int = 30) -> dict:
    """
    Train models for all 14 coastal districts.
    
    Args:
        days_back: Days of historical data
        
    Returns:
        Dictionary with results for all districts
    """
    all_results = {
        "start_time": datetime.now().isoformat(),
        "districts": {},
        "summary": {
            "total": len(COASTAL_DISTRICTS),
            "successful": 0,
            "failed": 0
        }
    }
    
    for i, district_name in enumerate(COASTAL_DISTRICTS.keys(), 1):
        logger.info(f"\n{'#'*60}")
        logger.info(f"DISTRICT {i}/{len(COASTAL_DISTRICTS)}: {district_name}")
        logger.info(f"{'#'*60}")
        
        try:
            results = train_single_district(district_name, days_back)
            all_results["districts"][district_name] = results
            
            if not results["errors"]:
                all_results["summary"]["successful"] += 1
            else:
                all_results["summary"]["failed"] += 1
                
        except Exception as e:
            logger.error(f"Critical error for {district_name}: {e}")
            all_results["districts"][district_name] = {
                "error": str(e)
            }
            all_results["summary"]["failed"] += 1
    
    all_results["end_time"] = datetime.now().isoformat()
    
    return all_results


def save_training_report(results: dict, output_path: str) -> None:
    """Save training results to JSON file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"Training report saved to {output_path}")


def main():
    """Main entry point for training script."""
    parser = argparse.ArgumentParser(
        description="Train GRU models for Marine Safety Forecasting"
    )
    
    parser.add_argument(
        '--district',
        type=str,
        help='Name of specific district to train (e.g., Chennai)'
    )
    
    parser.add_argument(
        '--all',
        action='store_true',
        help='Train models for all 14 coastal districts'
    )
    
    parser.add_argument(
        '--days',
        type=int,
        default=30,
        help='Days of historical data for training (default: 30)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='data/training_report.json',
        help='Output path for training report'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.district and not args.all:
        parser.print_help()
        print("\nError: Please specify --district <name> or --all")
        sys.exit(1)
    
    if args.district and args.district not in COASTAL_DISTRICTS:
        print(f"Error: Unknown district '{args.district}'")
        print(f"Available districts: {', '.join(COASTAL_DISTRICTS.keys())}")
        sys.exit(1)
    
    # Create model directories
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    os.makedirs(SCALER_SAVE_DIR, exist_ok=True)
    
    # Run training
    print("\n" + "="*60)
    print("MARINE SAFETY FORECASTING - MODEL TRAINING")
    print("="*60)
    
    if args.all:
        print(f"\nTraining models for ALL {len(COASTAL_DISTRICTS)} districts")
        print(f"Historical data: {args.days} days")
        results = train_all_districts(args.days)
    else:
        print(f"\nTraining models for {args.district}")
        print(f"Historical data: {args.days} days")
        results = train_single_district(args.district, args.days)
    
    # Save report
    report_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        args.output
    )
    save_training_report(results, report_path)
    
    # Print summary
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    
    if args.all:
        print(f"\nSummary:")
        print(f"  Total districts: {results['summary']['total']}")
        print(f"  Successful: {results['summary']['successful']}")
        print(f"  Failed: {results['summary']['failed']}")
    
    print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    main()
