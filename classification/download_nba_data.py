"""
NBA Data Download and Caching Script
Downloads player statistics for multiple seasons and saves them as CSV files.
This avoids repeated API calls and enables multi-season training.
"""

import pandas as pd
from nba_api.stats.endpoints import leaguedashplayerstats
import time
from pathlib import Path
from datetime import datetime


class NBADataDownloader:
    """Downloads and caches NBA player statistics for multiple seasons."""
    
    def __init__(self, data_dir=None):
        """
        Initialize the data downloader.
        
        Args:
            data_dir: Directory to save raw CSV files
        """
        # Use path relative to this script's location
        if data_dir is None:
            script_dir = Path(__file__).resolve().parent
            self.data_dir = script_dir.parent / 'data' / 'raw'
        else:
            self.data_dir = Path(data_dir).resolve()
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Define seasons to download (last 5 seasons)
        self.seasons = [
            '2021-22',
            '2022-23',
            '2023-24',
            '2024-25',
            '2025-26'  # Current season (started Sept 21, 2025)
        ]
    
    def download_season_data(self, season):
        """
        Download player statistics for a specific season.
        
        Args:
            season: NBA season string (e.g., '2023-24')
            
        Returns:
            DataFrame with player statistics or None if failed
        """
        csv_path = self.data_dir / f'player_stats_{season}.csv'
        
        # Check if already downloaded
        if csv_path.exists():
            print(f"  ✓ {season} data already cached at {csv_path}")
            return pd.read_csv(csv_path)
        
        print(f"  Downloading {season} season data...")
        
        try:
            # Fetch data from NBA API
            stats = leaguedashplayerstats.LeagueDashPlayerStats(
                season=season,
                per_mode_detailed='PerGame',
                measure_type_detailed_defense='Base',
                timeout=30
            )
            
            df = stats.get_data_frames()[0]
            
            # Add season column for tracking
            df['SEASON'] = season
            
            # Save to CSV
            df.to_csv(csv_path, index=False)
            
            print(f"  ✓ Downloaded {len(df)} players for {season}")
            print(f"  ✓ Saved to {csv_path}")
            
            # Be respectful to the API - wait 2 seconds between requests
            time.sleep(2)
            
            return df
            
        except Exception as e:
            print(f"  ✗ Error downloading {season}: {e}")
            return None
    
    def download_all_seasons(self):
        """
        Download data for all configured seasons.
        
        Returns:
            Dictionary mapping season to DataFrame
        """
        print("="*70)
        print("NBA Player Data Download - Multi-Season Cache")
        print("="*70)
        print(f"\nDownloading/caching data for seasons: {', '.join(self.seasons)}")
        print(f"Cache directory: {self.data_dir.absolute()}\n")
        
        season_data = {}
        
        for season in self.seasons:
            df = self.download_season_data(season)
            if df is not None:
                season_data[season] = df
            print()  # Blank line between seasons
        
        # Summary
        print("="*70)
        print("Download Summary")
        print("="*70)
        
        if season_data:
            total_players = sum(len(df) for df in season_data.values())
            print(f"\n✓ Successfully cached {len(season_data)} seasons")
            print(f"✓ Total player-season records: {total_players}")
            print(f"\nCached files:")
            for season in season_data.keys():
                csv_path = self.data_dir / f'player_stats_{season}.csv'
                size_kb = csv_path.stat().st_size / 1024
                print(f"  - player_stats_{season}.csv ({size_kb:.1f} KB)")
        else:
            print("\n✗ No data was downloaded")
        
        print("\n" + "="*70)
        
        return season_data
    
    def combine_seasons(self, seasons=None):
        """
        Load and combine multiple seasons into a single DataFrame.
        
        Args:
            seasons: List of seasons to combine (default: all available)
            
        Returns:
            Combined DataFrame with all seasons
        """
        if seasons is None:
            seasons = self.seasons
        
        print(f"\nCombining seasons: {', '.join(seasons)}")
        
        dfs = []
        for season in seasons:
            csv_path = self.data_dir / f'player_stats_{season}.csv'
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                dfs.append(df)
                print(f"  ✓ Loaded {len(df)} players from {season}")
            else:
                print(f"  ✗ {season} not found - run download first")
        
        if dfs:
            combined_df = pd.concat(dfs, ignore_index=True)
            print(f"\n✓ Combined dataset: {len(combined_df)} total records")
            return combined_df
        else:
            print("\n✗ No data to combine")
            return None
    
    def get_latest_season_data(self):
        """
        Load data for the most recent season only.
        
        Returns:
            DataFrame for latest season
        """
        latest_season = self.seasons[-1]
        csv_path = self.data_dir / f'player_stats_{latest_season}.csv'
        
        if csv_path.exists():
            print(f"Loading {latest_season} season data from cache...")
            df = pd.read_csv(csv_path)
            print(f"✓ Loaded {len(df)} players")
            return df
        else:
            print(f"Cache not found for {latest_season}. Downloading...")
            return self.download_season_data(latest_season)


def main():
    """Main function to download and cache NBA data."""
    downloader = NBADataDownloader()
    
    # Download all seasons
    season_data = downloader.download_all_seasons()
    
    # Optional: Create a combined dataset
    if season_data:
        print("\n" + "="*70)
        print("Creating Combined Dataset (All Seasons)")
        print("="*70)
        
        combined_df = downloader.combine_seasons()
        
        if combined_df is not None:
            # Save combined dataset
            script_dir = Path(__file__).resolve().parent
            combined_path = script_dir.parent / 'data' / 'raw' / 'player_stats_combined.csv'
            combined_df.to_csv(combined_path, index=False)
            
            print(f"\n✓ Combined dataset saved to {combined_path}")
            print(f"  Total records: {len(combined_df)}")
            print(f"  Seasons: {combined_df['SEASON'].unique().tolist()}")
            print(f"  File size: {combined_path.stat().st_size / 1024:.1f} KB")
    
    print("\n" + "="*70)
    print("Data download complete! You can now run data_preprocessing.py")
    print("="*70)


if __name__ == "__main__":
    main()
