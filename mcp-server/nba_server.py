from mcp.server.fastmcp import FastMCP
from nba_api.live.nba.endpoints import scoreboard
from nba_api.stats.endpoints import leaguestandings, playercareerstats, commonplayerinfo, leaguedashplayerstats, commonteamroster
from nba_api.stats.static import players, teams
import json
from datetime import datetime
from pathlib import Path
import sys
import torch
import numpy as np
import unicodedata

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from classification.player_classifier_model import load_model, TierLabels

# Initialize FastMCP server
mcp = FastMCP("nba-server")

# Global model and scaler
_model = None
_scaler_params = None
_feature_names = None


def _get_team_roster_data(team_name: str) -> dict:
    """
    Internal helper to get roster data without JSON serialization.
    Returns dict with team info and player list or error.
    """
    try:
        # Find team by name
        nba_teams = teams.find_teams_by_full_name(team_name)
        
        # If not found by full name, try abbreviation
        if not nba_teams:
            nba_teams = teams.find_team_by_abbreviation(team_name)
            if nba_teams:
                nba_teams = [nba_teams]
        
        # If still not found, try nickname (e.g., "Lakers")
        if not nba_teams:
            all_teams = teams.get_teams()
            nba_teams = [t for t in all_teams if team_name.lower() in t['full_name'].lower() or team_name.lower() in t['nickname'].lower()]
        
        if not nba_teams:
            return {"error": f"Team '{team_name}' not found."}
        
        team_id = nba_teams[0]['id']
        team_full_name = nba_teams[0]['full_name']
        
        # Get current roster
        roster = commonteamroster.CommonTeamRoster(team_id=team_id, season='2025-26')
        roster_data = roster.common_team_roster.get_dict()
        
        # Extract player names
        player_list = []
        if roster_data and 'data' in roster_data:
            for player in roster_data['data']:
                # Player name is typically in index 3
                if len(player) > 3:
                    player_list.append(player[3])
        
        return {
            "team": team_full_name,
            "season": "2025-26",
            "players": player_list,
            "player_count": len(player_list)
        }
    except Exception as e:
        return {"error": f"Error fetching team roster: {str(e)}"}


@mcp.tool()
def get_current_games_with_rosters() -> str:
    """
    COMPOSITE TOOL: Get today's games AND the complete rosters for all teams playing.
    
    This single tool provides everything needed to analyze today's matchups:
    - All live/scheduled games for today
    - Complete roster (player names) for each team in each game
    
    Returns:
        JSON string with games and their rosters. Format:
        {
          "games": [
            {
              "game_id": "...",
              "home_team": "Team Name",
              "away_team": "Team Name", 
              "home_roster": ["Player1", "Player2", ...],
              "away_roster": ["Player1", "Player2", ...],
              "status": "...",
              "home_score": X,
              "away_score": Y
            },
            ...
          ],
          "total_games": N
        }
    """
    try:
        # Step 1: Get today's games
        board = scoreboard.ScoreBoard()
        games_data = board.games.get_dict()
        
        result = {
            "games": [],
            "total_games": 0
        }
        
        if not games_data:
            return json.dumps(result, indent=2)
        
        # Step 2: For each game, get both team rosters
        for game in games_data:
            home_team_name = game.get('homeTeam', {}).get('teamName', '')
            away_team_name = game.get('awayTeam', {}).get('teamName', '')
            
            # Get rosters for both teams
            home_roster_data = _get_team_roster_data(home_team_name)
            away_roster_data = _get_team_roster_data(away_team_name)
            
            game_info = {
                "game_id": game.get('gameId', ''),
                "home_team": home_team_name,
                "away_team": away_team_name,
                "home_roster": home_roster_data.get('players', []) if 'error' not in home_roster_data else [],
                "away_roster": away_roster_data.get('players', []) if 'error' not in away_roster_data else [],
                "status": game.get('gameStatus', 0),
                "home_score": game.get('homeTeam', {}).get('score', 0),
                "away_score": game.get('awayTeam', {}).get('score', 0)
            }
            
            result["games"].append(game_info)
        
        result["total_games"] = len(result["games"])
        
        return json.dumps(result, indent=2)
        
    except Exception as e:
        return json.dumps({"error": f"Error fetching games with rosters: {str(e)}"}, indent=2)

@mcp.tool()
def get_live_games() -> str:
    """
    Get the current live games and scores for today.
    Returns a JSON string containing game details.
    """
    try:
        board = scoreboard.ScoreBoard()
        games = board.games.get_dict()
        return json.dumps(games, indent=2)
    except Exception as e:
        return f"Error fetching live games: {str(e)}"

@mcp.tool()
def get_standings() -> str:
    """
    Get the current NBA standings for both conferences.
    Returns a JSON string containing standings.
    """
    try:
        standings = leaguestandings.LeagueStandings()
        data = standings.standings.get_dict()
        return json.dumps(data, indent=2)
    except Exception as e:
        return f"Error fetching standings: {str(e)}"

@mcp.tool()
def get_player_stats(player_name: str) -> str:
    """
    Get career statistics for a specific NBA player.
    Args:
        player_name: The full name of the player (e.g., "LeBron James")
    Returns:
        JSON string with player stats.
    """
    try:
        # Find player by name
        nba_players = players.find_players_by_full_name(player_name)
        if not nba_players:
            return f"Player '{player_name}' not found."
        
        player_id = nba_players[0]['id']
        
        # Get career stats
        career = playercareerstats.PlayerCareerStats(player_id=player_id)
        stats = career.career_totals_regular_season.get_dict()
        
        return json.dumps(stats, indent=2)
    except Exception as e:
        return f"Error fetching player stats: {str(e)}"


@mcp.tool()
def get_team_roster(team_name: str) -> str:
    """
    Get the current roster for an NBA team.
    Args:
        team_name: The team name or abbreviation (e.g., "Lakers", "LAL", "Los Angeles Lakers")
    Returns:
        JSON string with list of player names on the team.
    """
    try:
        # Find team by name
        nba_teams = teams.find_teams_by_full_name(team_name)
        
        # If not found by full name, try abbreviation
        if not nba_teams:
            nba_teams = teams.find_team_by_abbreviation(team_name)
            if nba_teams:
                nba_teams = [nba_teams]
        
        # If still not found, try nickname (e.g., "Lakers")
        if not nba_teams:
            all_teams = teams.get_teams()
            nba_teams = [t for t in all_teams if team_name.lower() in t['full_name'].lower() or team_name.lower() in t['nickname'].lower()]
        
        if not nba_teams:
            return f"Team '{team_name}' not found."
        
        team_id = nba_teams[0]['id']
        team_full_name = nba_teams[0]['full_name']
        
        # Get current roster
        roster = commonteamroster.CommonTeamRoster(team_id=team_id, season='2025-26')
        roster_data = roster.common_team_roster.get_dict()
        
        # Extract player names
        player_list = []
        if roster_data and 'data' in roster_data:
            for player in roster_data['data']:
                # Player name is typically in index 3
                if len(player) > 3:
                    player_list.append(player[3])
        
        result = {
            "team": team_full_name,
            "season": "2025-26",
            "players": player_list,
            "player_count": len(player_list)
        }
        
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error fetching team roster: {str(e)}"


def _load_classification_model():
    """Load the trained classification model and scaler parameters."""
    global _model, _scaler_params, _feature_names
    
    if _model is not None:
        return  # Already loaded
    
    try:
        model_dir = project_root / "data" / "models"
        model_path = model_dir / "player_classifier.pth"
        scaler_path = project_root / "data" / "scaler_params.json"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}. Please train the model first.")
        
        if not scaler_path.exists():
            raise FileNotFoundError(f"Scaler parameters not found at {scaler_path}.")
        
        # Load model
        _model = load_model(model_path, device='cpu')
        
        # Load scaler parameters
        with open(scaler_path, 'r') as f:
            _scaler_params = json.load(f)
            _feature_names = _scaler_params['feature_names']
        
    except Exception as e:
        raise RuntimeError(f"Failed to load classification model: {str(e)}")


def _normalize_features(features_dict):
    """Normalize features using saved scaler parameters."""
    global _scaler_params, _feature_names
    
    # Extract features in correct order
    features = []
    for feature_name in _feature_names:
        features.append(features_dict.get(feature_name, 0.0))
    
    # Convert to numpy array
    features_array = np.array([features])
    
    # Normalize using saved mean and scale
    mean = np.array(_scaler_params['mean'])
    scale = np.array(_scaler_params['scale'])
    
    normalized = (features_array - mean) / scale
    
    return normalized


@mcp.tool()
def aggregate_roster_classifications(team_name: str) -> str:
    """
    AGGREGATOR TOOL: Classify all players on a team's roster and return tier counts + player lists.
    
    This tool provides a complete classification summary for an entire team roster:
    - Gets the team's current roster (2025-26 season)
    - Classifies each player into their performance tier
    - Aggregates counts by tier (Elite, All-Star, Starter, Rotation, Bench)
    - Lists player names grouped by tier
    
    Use this tool when the user asks for:
    - "How many [tier] players does [team] have?"
    - "Give me tier counts for [team]"
    - "Classify all players on [team]"
    - "Show me the roster breakdown for [team]"
    
    Args:
        team_name: The team name or abbreviation (e.g., "Lakers", "Thunder", "OKC")
    
    Returns:
        JSON string with tier counts and player lists by tier.
    """
    try:
        # Load model if not already loaded
        _load_classification_model()
        
        # Get team roster
        roster_data = _get_team_roster_data(team_name)
        
        if "error" in roster_data:
            return json.dumps(roster_data)
        
        team_full_name = roster_data["team"]
        player_names = roster_data["players"]
        
        # Get current season stats for all players
        stats_data = leaguedashplayerstats.LeagueDashPlayerStats(
            season='2025-26',
            per_mode_detailed='PerGame',
            timeout=30
        )
        df = stats_data.get_data_frames()[0]
        
        # Normalize name helper
        def normalize_name(name):
            nfd = unicodedata.normalize('NFD', name)
            return ''.join(char for char in nfd if unicodedata.category(char) != 'Mn').lower()
        
        # Initialize tier buckets
        tier_buckets = {
            "Elite": [],
            "All-Star": [],
            "Starter": [],
            "Rotation": [],
            "Bench": [],
            "Not Found": []
        }
        
        # Classify each player
        for player_name in player_names:
            try:
                search_name = normalize_name(player_name)
                
                # Find player in stats
                player_row = None
                for idx, row in df.iterrows():
                    if normalize_name(row['PLAYER_NAME']) == search_name:
                        player_row = row
                        break
                
                if player_row is None:
                    tier_buckets["Not Found"].append(player_name)
                    continue
                
                # Extract features
                features = {}
                for feature_name in _feature_names:
                    features[feature_name] = float(player_row.get(feature_name, 0.0))
                
                # Normalize and predict
                normalized_features = _normalize_features(features)
                features_tensor = torch.FloatTensor(normalized_features)
                predicted_class, probabilities = _model.predict(features_tensor)
                
                tier_id = int(predicted_class[0])
                tier_name = TierLabels.get_tier_name(tier_id)
                
                # Add to appropriate bucket
                tier_buckets[tier_name].append(player_name)
                
            except Exception as e:
                tier_buckets["Not Found"].append(f"{player_name} (error: {str(e)})")
        
        # Build result
        result = {
            "team": team_full_name,
            "season": "2025-26",
            "total_players": len(player_names),
            "tier_counts": {
                "Elite": len(tier_buckets["Elite"]),
                "All-Star": len(tier_buckets["All-Star"]),
                "Starter": len(tier_buckets["Starter"]),
                "Rotation": len(tier_buckets["Rotation"]),
                "Bench": len(tier_buckets["Bench"])
            },
            "players_by_tier": {
                "Elite": tier_buckets["Elite"],
                "All-Star": tier_buckets["All-Star"],
                "Starter": tier_buckets["Starter"],
                "Rotation": tier_buckets["Rotation"],
                "Bench": tier_buckets["Bench"]
            }
        }
        
        if tier_buckets["Not Found"]:
            result["players_not_classified"] = tier_buckets["Not Found"]
        
        return json.dumps(result, indent=2)
        
    except Exception as e:
        return json.dumps({"error": f"Aggregation failed: {str(e)}"})


@mcp.tool()
def classify_player_tier(player_name: str) -> str:
    """
    Classify an NBA player into a performance tier using a trained neural network.
    
    This tool uses a machine learning model trained on historical player statistics (2021-2026)
    to classify players from the current 2025-26 season into one of five tiers:
    - Elite (Top 10%): Superstar players, MVP candidates
    - All-Star (70-90%): Top-tier players, all-star caliber
    - Starter (40-70%): Consistent starters with solid production
    - Rotation (20-40%): Regular contributors off the bench
    - Bench (Bottom 20%): Limited minutes, developing skills
    
    Args:
        player_name: The full name of the player (e.g., "Stephen Curry")
    
    Returns:
        JSON string with classification results including tier, confidence, and stats.
    """
    try:
        # Load model if not already loaded
        _load_classification_model()
        
        # Get current season player stats (2025-26)
        stats_data = leaguedashplayerstats.LeagueDashPlayerStats(
            season='2025-26',
            per_mode_detailed='PerGame',
            timeout=30
        )
        df = stats_data.get_data_frames()[0]
        
        # Normalize both search name and data names for better matching
        # Remove accents and special characters, convert to lowercase
        import unicodedata
        def normalize_name(name):
            # Normalize unicode to decomposed form, remove accents, lowercase
            nfd = unicodedata.normalize('NFD', name)
            return ''.join(char for char in nfd if unicodedata.category(char) != 'Mn').lower()
        
        search_name = normalize_name(player_name)
        
        # Find player using normalized names
        player_row = None
        for idx, row in df.iterrows():
            if normalize_name(row['PLAYER_NAME']) == search_name:
                player_row = row
                break
        
        if player_row is None:
            # Try partial match as fallback
            for idx, row in df.iterrows():
                if search_name in normalize_name(row['PLAYER_NAME']) or normalize_name(row['PLAYER_NAME']) in search_name:
                    player_row = row
                    break
        
        if player_row is None:
            return f"Player '{player_name}' not found in current season data. Please check the spelling or try a different season."
        
        player_data = player_row
        
        # Extract features for classification
        features = {}
        for feature_name in _feature_names:
            features[feature_name] = float(player_data.get(feature_name, 0.0))
        
        # Normalize features
        normalized_features = _normalize_features(features)
        
        # Convert to tensor
        features_tensor = torch.FloatTensor(normalized_features)
        
        # Predict
        predicted_class, probabilities = _model.predict(features_tensor)
        
        tier_id = int(predicted_class[0])
        tier_name = TierLabels.get_tier_name(tier_id)
        tier_description = TierLabels.get_tier_description(tier_id)
        tier_color = TierLabels.get_tier_color(tier_id)
        
        # Get confidence scores
        probs = probabilities[0].numpy()
        confidence = float(probs[tier_id] * 100)
        
        # Return simplified text summary instead of complex JSON
        # This avoids Gemini content policy issues with detailed structured data
        stats_summary = (
            f"{player_data['PLAYER_NAME']} ({player_data.get('TEAM_ABBREVIATION', 'N/A')}) "
            f"- Classified as {tier_name} tier ({confidence:.1f}% confidence). "
            f"Key stats: {player_data.get('PTS', 0):.1f} PPG, "
            f"{player_data.get('REB', 0):.1f} RPG, "
            f"{player_data.get('AST', 0):.1f} APG in {int(player_data.get('GP', 0))} games."
        )
        
        return stats_summary
        
    except FileNotFoundError as e:
        return f"Model not available: {str(e)}. Please run the training pipeline first."
    except Exception as e:
        return f"Classification failed: {str(e)}"

if __name__ == "__main__":
    # Run the server
    mcp.run()
