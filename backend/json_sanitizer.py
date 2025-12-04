"""
JSON Sanitizer for Gemini API
Converts structured JSON responses to natural language summaries to avoid content policy issues.
"""

import json
from typing import Any


def sanitize_tool_response(tool_name: str, result_text: str) -> str:
    """
    Convert tool response JSON to natural language summary.
    
    This prevents Gemini content policy blocks by ensuring the model receives
    human-readable text instead of raw JSON structures.
    
    Args:
        tool_name: Name of the MCP tool that was called
        result_text: Raw text response from the tool (may be JSON or plain text)
    
    Returns:
        Natural language summary safe for Gemini processing
    """
    try:
        # Try to parse as JSON
        result_data = json.loads(result_text)
        
        # Convert to natural language based on tool type
        if tool_name == "get_live_games":
            return _summarize_live_games(result_data)
        
        elif tool_name == "get_standings":
            return _summarize_standings(result_data)
        
        elif tool_name == "get_player_stats":
            return _summarize_player_stats(result_data)
        
        elif tool_name == "get_team_roster":
            return _summarize_team_roster(result_data)
        
        elif tool_name == "aggregate_roster_classifications":
            return _summarize_roster_classifications(result_data)
        
        elif tool_name == "classify_player_tier":
            # Already returns text format, use as-is
            return result_text
        
        else:
            # Generic: truncate and summarize
            return _generic_summary(result_data)
            
    except json.JSONDecodeError:
        # Not JSON, use text as-is (already formatted by MCP server)
        return result_text[:1000] + "..." if len(result_text) > 1000 else result_text


def _summarize_live_games(data: Any) -> str:
    """Summarize live games data with scores and status."""
    if isinstance(data, list):
        game_count = len(data)
        game_summaries = []
        
        for idx, game in enumerate(data, 1):
            home_team = game.get('homeTeam', {})
            away_team = game.get('awayTeam', {})
            home = home_team.get('teamName', 'Unknown')
            away = away_team.get('teamName', 'Unknown')
            home_score = home_team.get('score', 0)
            away_score = away_team.get('score', 0)
            status = game.get('gameStatusText', 'Scheduled')
            
            # Format: Game 1: TeamA 110 vs TeamB 105 (Final)
            game_summaries.append(f"Game {idx}: {away} {away_score} vs {home} {home_score} ({status})")
        
        return f"{game_count} games today:\n" + "\n".join(game_summaries)
    else:
        return str(data)


def _summarize_standings(data: Any) -> str:
    """Summarize standings data."""
    if isinstance(data, list):
        team_summaries = []
        for team in data[:15]:  # Limit to top 15 teams
            name = team.get('TeamName', 'Unknown')
            wins = team.get('WINS', 0)
            losses = team.get('LOSSES', 0)
            conf = team.get('Conference', '?')
            team_summaries.append(f"{name} ({conf}): {wins}-{losses}")
        
        return f"Standings: {'; '.join(team_summaries)}"
    else:
        return str(data)


def _summarize_player_stats(data: Any) -> str:
    """Summarize player stats data."""
    if isinstance(data, dict):
        name = data.get('name', 'Unknown')
        ppg = data.get('points_per_game', 'N/A')
        rpg = data.get('rebounds_per_game', 'N/A')
        apg = data.get('assists_per_game', 'N/A')
        fg_pct = data.get('field_goal_percentage', 'N/A')
        
        return f"{name} career stats: {ppg} PPG, {rpg} RPG, {apg} APG, {fg_pct}% FG"
    else:
        return str(data)


def _summarize_team_roster(data: Any) -> str:
    """Summarize team roster data."""
    if isinstance(data, dict):
        team = data.get('team', 'Unknown')
        players = data.get('players', [])
        count = data.get('player_count', len(players))
        
        # List all players (comma-separated)
        if players:
            player_list = ', '.join(players[:20])  # Limit to first 20 players
            if count > 20:
                player_list += f" ... and {count - 20} more"
            return f"{team} roster ({count} players): {player_list}"
        else:
            return f"{team} roster: No players found"
    else:
        return str(data)


def _summarize_roster_classifications(data: Any) -> str:
    """Summarize aggregated roster classification data in natural language."""
    if isinstance(data, dict):
        if "error" in data:
            return data["error"]
        
        team = data.get('team', 'Unknown')
        tier_counts = data.get('tier_counts', {})
        players_by_tier = data.get('players_by_tier', {})
        
        # Build concise tier count summary
        elite_count = tier_counts.get('Elite', 0)
        allstar_count = tier_counts.get('All-Star', 0)
        starter_count = tier_counts.get('Starter', 0)
        rotation_count = tier_counts.get('Rotation', 0)
        bench_count = tier_counts.get('Bench', 0)
        
        summary = (
            f"The {team} have {elite_count} Elite, {allstar_count} All-Star, "
            f"{starter_count} Starter, {rotation_count} Rotation, and {bench_count} Bench players."
        )
        
        # Add player lists for non-empty tiers
        tier_details = []
        
        if elite_count > 0:
            elite_players = players_by_tier.get('Elite', [])
            tier_details.append(f"Elite: {', '.join(elite_players)}")
        
        if allstar_count > 0:
            allstar_players = players_by_tier.get('All-Star', [])
            tier_details.append(f"All-Star: {', '.join(allstar_players)}")
        
        if starter_count > 0:
            starter_players = players_by_tier.get('Starter', [])
            tier_details.append(f"Starter: {', '.join(starter_players)}")
        
        if rotation_count > 0:
            rotation_players = players_by_tier.get('Rotation', [])
            # Limit rotation list to avoid verbosity
            if len(rotation_players) > 5:
                tier_details.append(f"Rotation: {', '.join(rotation_players[:5])} and {len(rotation_players) - 5} more")
            else:
                tier_details.append(f"Rotation: {', '.join(rotation_players)}")
        
        if bench_count > 0:
            bench_players = players_by_tier.get('Bench', [])
            # Limit bench list to avoid verbosity
            if len(bench_players) > 5:
                tier_details.append(f"Bench: {', '.join(bench_players[:5])} and {len(bench_players) - 5} more")
            else:
                tier_details.append(f"Bench: {', '.join(bench_players)}")
        
        if tier_details:
            summary += "\n\n" + "\n".join(tier_details)
        
        return summary
    else:
        return str(data)


def _generic_summary(data: Any) -> str:
    """Generic summary for unknown tool types."""
    data_str = str(data)
    if len(data_str) > 500:
        return data_str[:500] + f"... (truncated, {len(data_str)} total chars)"
    return data_str
