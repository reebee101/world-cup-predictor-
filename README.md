# world-cup-predictor
Predicts the remaining 2026 FIFA World Cup knockout bracket (Round of 16 → Final) using a Poisson goal model with maximum-likelihood attack/defense ratings, simulated 20,000 times via Monte Carlo.
How it works

Match history — every 2026 World Cup result so far, plus 2022 results as a secondary prior. 2026 form is weighted 2.5–4.7x heavier than 2022, and knockout matches count more than group games.
Attack/defense ratings, fit by maximum likelihood — instead of a naive goals_scored / average ratio, ratings are solved by iterative proportional fitting (the same core idea behind Dixon-Coles): the attack_i / defense_i values that make the actual observed scorelines most probable, given who each team played. A small prior (equivalent to ~3 "average" matches) keeps ratings sane on small samples. On matches where both teams are alive, this beats the naive ratio method by about +9% log-likelihood.
Match simulation — goals are drawn from a Poisson distribution using both teams' ratings. Knockout draws go to extra time (lower-scoring), then a penalty shootout if still level.
Monte Carlo — the whole remaining bracket (R16 → QF → SF → Final) is simulated 20,000 times to get stable probabilities, not just one predicted scoreline.

Usage
Update MATCHES with new results as they come in, then re-run:
python3 wc2026_predictor.py
Output

Team strength ratings (attack/defense)
Naive-vs-MLE log-likelihood comparison
Expected goals (λ) for each upcoming matchup
Win probabilities for today's Round of 16 matches
Win probabilities for the fixed quarterfinals
Reach-QF / reach-SF / reach-final / win-title probability for all 10 remaining teams

Requirements
Python 3, standard library only (random, math, collections) — no external dependencies.
