"""
NBA Player Data Collection and Preprocessing Pipeline
This script collects player statistics from the NBA API and preprocesses them for neural network training.
"""

import pandas as pd
import numpy as np
from nba_api.stats.endpoints import leaguedashplayerstats
from sklearn.preprocessing import StandardScaler
import json
import time
from pathlib import Path
from download_nba_data import NBADataDownloader

class NBADataCollector:
    """Collects and preprocesses NBA player data for classification."""
    
    def __init__(self, seasons=None, data_dir=None, use_cache=True):
        """
        Initialize the data collector.
        
        Args:
            seasons: List of NBA seasons to use (default: all cached seasons)
            data_dir: Directory to save processed data
            use_cache: Whether to use cached CSV files (recommended)
        """
        self.seasons = seasons
        # Use path relative to this script's location
        if data_dir is None:
            script_dir = Path(__file__).resolve().parent
            self.data_dir = script_dir.parent / 'data'
        else:
            self.data_dir = Path(data_dir).resolve()
        self.data_dir.mkdir(exist_ok=True)
        self.use_cache = use_cache
        
        # Set downloader path relative to script location
        script_dir = Path(__file__).resolve().parent
        self.downloader = NBADataDownloader(data_dir=str(script_dir.parent / 'data' / 'raw'))
        
        # Features to use for classification
        self.stat_features = [
            'GP',           # Games Played
            'MIN',          # Minutes Per Game
            'PTS',          # Points Per Game
            'FG_PCT',       # Field Goal Percentage
            'FG3_PCT',      # 3-Point Percentage
            'FT_PCT',       # Free Throw Percentage
            'REB',          # Rebounds Per Game
            'AST',          # Assists Per Game
            'STL',          # Steals Per Game
            'BLK',          # Blocks Per Game
            'TOV',          # Turnovers Per Game
            'PF',           # Personal Fouls Per Game
            'PLUS_MINUS'    # Plus/Minus
        ]
    
    def collect_player_stats(self):
        """
        Collect player statistics from cached CSV files or NBA API.
        
        Returns:
            DataFrame with player statistics
        """
        if self.use_cache:
            print("Loading player stats from cached CSV files...")
            
            # Check if combined file exists
            script_dir = Path(__file__).resolve().parent
            combined_path = script_dir.parent / 'data' / 'raw' / 'player_stats_combined.csv'
            if combined_path.exists() and self.seasons is None:
                print(f"  Loading combined dataset from {combined_path}")
                df = pd.read_csv(combined_path)
                print(f"  ✓ Loaded {len(df)} player-season records from cache")
                print(f"  ✓ Seasons: {sorted(df['SEASON'].unique())}")
                return df
            
            # Load specific seasons or all available
            if self.seasons:
                print(f"  Loading specific seasons: {self.seasons}")
                df = self.downloader.combine_seasons(self.seasons)
            else:
                # Try to load all available cached seasons
                df = self.downloader.combine_seasons()
            
            if df is not None:
                return df
            else:
                print("  ⚠ No cached data found. Run download_nba_data.py first!")
                print("  Falling back to API download...")
        
        # Fallback: Download from API (single season for compatibility)
        print(f"Downloading from NBA API...")
        season = self.seasons[0] if self.seasons else '2025-26'
        print(f"  Season: {season}")
        
        try:
            stats = leaguedashplayerstats.LeagueDashPlayerStats(
                season=season,
                per_mode_detailed='PerGame',
                measure_type_detailed_defense='Base',
                timeout=30
            )
            
            df = stats.get_data_frames()[0]
            df['SEASON'] = season
            
            print(f"  ✓ Collected stats for {len(df)} players")
            return df
            
        except Exception as e:
            print(f"  ✗ Error collecting stats: {e}")
            return None
    
    def create_tier_labels(self, df):
        """
        Create performance tier labels based on player statistics.
        
        Tiers:
        - 0: Bench (Bottom 20%)
        - 1: Rotation (20-40%)
        - 2: Starter (40-70%)
        - 3: All-Star (70-90%)
        - 4: Elite (Top 10%)
        
        Args:
            df: DataFrame with player statistics
            
        Returns:
            DataFrame with added 'TIER' column
        """
        print("Creating performance tier labels...")
        
        # Filter players with sufficient playing time (>= 15 games, >= 10 mins per game)
        df_filtered = df[(df['GP'] >= 15) & (df['MIN'] >= 10.0)].copy()
        
        # Create composite score based on multiple weighted stats
        # Normalize each stat to 0-1 range before weighting
        df_filtered['PTS_NORM'] = (df_filtered['PTS'] - df_filtered['PTS'].min()) / (df_filtered['PTS'].max() - df_filtered['PTS'].min())
        df_filtered['REB_NORM'] = (df_filtered['REB'] - df_filtered['REB'].min()) / (df_filtered['REB'].max() - df_filtered['REB'].min())
        df_filtered['AST_NORM'] = (df_filtered['AST'] - df_filtered['AST'].min()) / (df_filtered['AST'].max() - df_filtered['AST'].min())
        df_filtered['FG_PCT_NORM'] = (df_filtered['FG_PCT'] - df_filtered['FG_PCT'].min()) / (df_filtered['FG_PCT'].max() - df_filtered['FG_PCT'].min())
        
        # Weighted composite score
        df_filtered['COMPOSITE_SCORE'] = (
            df_filtered['PTS_NORM'] * 0.40 +      # Points weighted heavily
            df_filtered['REB_NORM'] * 0.20 +      # Rebounds
            df_filtered['AST_NORM'] * 0.20 +      # Assists
            df_filtered['FG_PCT_NORM'] * 0.20     # Efficiency
        )
        
        # Assign tiers based on percentiles
        df_filtered['TIER'] = pd.cut(
            df_filtered['COMPOSITE_SCORE'],
            bins=[0, 0.20, 0.40, 0.70, 0.90, 1.0],
            labels=[0, 1, 2, 3, 4],  # Bench, Rotation, Starter, All-Star, Elite
            include_lowest=True
        ).astype(int)
        
        # Display tier distribution
        print("\nTier Distribution:")
        tier_counts = df_filtered['TIER'].value_counts().sort_index()
        tier_names = ['Bench', 'Rotation', 'Starter', 'All-Star', 'Elite']
        for tier, count in tier_counts.items():
            print(f"  {tier_names[tier]}: {count} players ({count/len(df_filtered)*100:.1f}%)")
        
        return df_filtered
    
    def preprocess_features(self, df):
        """
        Preprocess features for neural network training.
        
        Args:
            df: DataFrame with player statistics and tiers
            
        Returns:
            Tuple of (features_normalized, labels, scaler, feature_names, player_names)
        """
        print("\nPreprocessing features...")
        
        # Handle missing values (fill with median for percentage stats)
        for col in ['FG_PCT', 'FG3_PCT', 'FT_PCT']:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median())
        
        # Fill other missing values with 0
        df[self.stat_features] = df[self.stat_features].fillna(0)
        
        # Extract features and labels
        X = df[self.stat_features].values
        y = df['TIER'].values
        player_names = df['PLAYER_NAME'].values
        
        # Normalize features using StandardScaler
        scaler = StandardScaler()
        X_normalized = scaler.fit_transform(X)
        
        print(f"Preprocessed {len(X)} players with {len(self.stat_features)} features")
        
        return X_normalized, y, scaler, self.stat_features, player_names
    
    def save_processed_data(self, X, y, scaler, feature_names, player_names):
        """
        Save processed data and scaler for later use.
        
        Args:
            X: Normalized feature array
            y: Label array
            scaler: Fitted StandardScaler
            feature_names: List of feature names
            player_names: Array of player names
        """
        print("\nSaving processed data...")
        
        # Save features and labels
        np.save(self.data_dir / 'X_train.npy', X)
        np.save(self.data_dir / 'y_train.npy', y)
        np.save(self.data_dir / 'player_names.npy', player_names)
        
        # Save scaler parameters
        scaler_params = {
            'mean': scaler.mean_.tolist(),
            'scale': scaler.scale_.tolist(),
            'feature_names': feature_names
        }
        with open(self.data_dir / 'scaler_params.json', 'w') as f:
            json.dump(scaler_params, f, indent=2)
        
        print(f"Data saved to {self.data_dir}/")
        print(f"  - X_train.npy: {X.shape}")
        print(f"  - y_train.npy: {y.shape}")
        print(f"  - player_names.npy: {len(player_names)} names")
        print(f"  - scaler_params.json")
    
    def run_pipeline(self):
        """
        Run the complete data collection and preprocessing pipeline.
        
        Returns:
            Tuple of (X, y, scaler, feature_names, player_names) or None if failed
        """
        print("="*60)
        print("NBA Player Data Collection & Preprocessing Pipeline")
        print("="*60)
        
        # Step 1: Collect data
        df = self.collect_player_stats()
        if df is None:
            return None
        
        # Small delay to be respectful to NBA API
        time.sleep(1)
        
        # Step 2: Create tier labels
        df_with_tiers = self.create_tier_labels(df)
        
        # Step 3: Preprocess features
        X, y, scaler, feature_names, player_names = self.preprocess_features(df_with_tiers)
        
        # Step 4: Save processed data
        self.save_processed_data(X, y, scaler, feature_names, player_names)
        
        print("\n" + "="*60)
        print("Pipeline completed successfully!")
        print("="*60)
        
        return X, y, scaler, feature_names, player_names


def main():
    """Main function to run the data collection pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description='NBA Player Data Preprocessing')
    parser.add_argument('--seasons', nargs='+', help='Specific seasons to use (e.g., 2023-24 2024-25)')
    parser.add_argument('--no-cache', action='store_true', help='Disable cached data (download from API)')
    args = parser.parse_args()
    
    # Initialize collector with multi-season support
    # Default: use all cached seasons for better model performance
    collector = NBADataCollector(
        seasons=args.seasons,
        use_cache=not args.no_cache
    )
    
    # Run the pipeline
    result = collector.run_pipeline()
    
    if result is not None:
        X, y, scaler, feature_names, player_names = result
        print(f"\nReady for training!")
        print(f"Features shape: {X.shape}")
        print(f"Labels shape: {y.shape}")
        print(f"Number of classes: {len(np.unique(y))}")
        print(f"\nDataset includes player-seasons from multiple years for better generalization.")
    else:
        print("\nPipeline failed. Please check the errors above.")
        print("\nTip: Run 'python download_nba_data.py' first to cache data.")


if __name__ == "__main__":
    main()
