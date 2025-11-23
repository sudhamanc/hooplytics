from mcp.server.fastmcp import FastMCP
from nba_api.live.nba.endpoints import scoreboard
from nba_api.stats.endpoints import leaguestandings, playercareerstats, commonplayerinfo
from nba_api.stats.static import players
import json
from datetime import datetime

# Initialize FastMCP server
mcp = FastMCP("nba-server")

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

if __name__ == "__main__":
    # Run the server
    mcp.run()
