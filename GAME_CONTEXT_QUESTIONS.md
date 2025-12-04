# Game Context Classification Questions

## Overview
This document categorizes sample questions by the tools and knowledge sources they utilize:
1. **NBA API Only** - Real-time data from NBA endpoints
2. **Gemini Knowledge Only** - Basketball facts from Gemini's training
3. **NBA API + Gemini** - Live data with contextual interpretation
4. **NBA API + Gemini + Classifier** - Live data + ML predictions + context
5. **Classifier Only** - ML-based player tier classification
6. **Guardrails** - Non-basketball questions that should be rejected

---

## 1. NBA API Only
*Uses: `get_live_games()`, `get_standings()`, `get_player_stats()`*

```
"What are today's NBA games?"
"Show current NBA standings"
```

---

## 2. Gemini Knowledge Only
*Uses: No tools, relies on Gemini's basketball knowledge*

```
"Who won the 2020 NBA championship?"
"How many quarters are in an NBA game?"
```

---

## 3. NBA API + Gemini Together
*Uses: Live data from NBA API + Gemini's contextual knowledge*

```
"Are there any upsets in today's games?"
"Which team has the best home record?"
```

---

## 4. NBA API + Gemini + Classifier
*Uses: Live game data + ML classification + contextual analysis*
*Key Tool: `aggregate_roster_classifications(team_name)` - Efficient roster-wide tier classification*

```
"Give me counts by classification for all the players in the #1 seeded team"
"How many Elite players does the #1 seed have?"
"Which team in the top 5 has the most All-Star tier players?"
"Compare the average tier of Lakers vs Warriors rosters"
```

**Implementation Note:** The new `aggregate_roster_classifications` tool handles roster-wide queries efficiently by:
1. Getting team roster in a single API call
2. Classifying all players server-side
3. Aggregating tier counts and grouping players by tier
4. Returning a concise natural-language summary

This avoids the finish_reason 12 issues that occurred with sequential individual classifications.

---

## 5. Classifier Only
*Uses: `classify_player_tier()` for ML-based tier predictions*

```
"Classify LeBron James"
"Who's higher tier: Embiid or Jokic?"
```

---

## 6. Guardrails (Non-Basketball Questions)
*Should trigger rejection response*

```
"What's the weather today?"
"Who won the Super Bowl?"
```

**Expected Response Format:**
```
"I'm Hoop.io, your NBA assistant! I can only help with NBA-related questions like:
- Current games and scores
- Team standings
- Player statistics  
- Player performance tier classifications

Please ask me about basketball!"
```

---


## Quick Reference by Use Case

| Category | Primary Tools | Example |
|----------|--------------|---------|
| 1. NBA API Only | NBA API | "What are today's NBA games?" |
| 2. Gemini Knowledge Only | Gemini | "Who won the 2020 NBA championship?" |
| 3. NBA API + Gemini Together | NBA API + Gemini | "Are there any upsets in today's games?" |
| 4. NBA API + Gemini + Classifier | NBA API + Gemini + Classifier | "Which game today features the most Elite tier players?" |
| 5. Classifier Only | Classifier | "Classify LeBron James" |
| 6. Guardrails | None (Rejection) | "What's the weather today?" → Rejection |
