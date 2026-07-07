"""
2026 FIFA World Cup — Remaining Bracket Predictor
==================================================
Predicts: remaining Round of 16 matches -> Quarterfinals -> Semifinals -> Final

Method
------
1. Build a match history for the 10 teams still alive (as of July 7, 2026,
   before Egypt-Argentina and Switzerland-Colombia kick off).
2. Each match gets a WEIGHT:
      - 2026 World Cup matches: weight 1.0 (group stage) to 1.4 (knockout)
      - 2022 World Cup matches: weight 0.30 (group stage) to 0.40 (knockout)
   This gives this year's form the dominant say, while 2022 acts as a
   secondary prior for teams with little current signal.
3. Attack and defense ratings are fit by MAXIMUM LIKELIHOOD, not a raw
   goals/average ratio: we solve for the attack_i / defense_i values that
   maximize the Poisson likelihood of every observed scoreline in the
   dataset, so a team's rating reflects who they actually played, not just
   their own tally. This is the same core idea behind Dixon-Coles style
   football models, solved here by iterative proportional fitting (each
   coordinate has a closed-form update, so it converges in well under a
   second with no external solver).
4. Each remaining match is simulated with Poisson-distributed goals based on
   both teams' ratings. Knockout draws go to a simulated "extra time" (30 extra
   mins of lower-scoring football) and then a penalty shootout if still tied.
5. The whole remaining bracket (R16 -> QF -> SF -> Final) is simulated
   20,000 times (Monte Carlo) to get robust probabilities, not just a
   single point prediction.

Re-run this script yourself as real results come in — just update MATCHES.
"""

import random
import math
from collections import defaultdict

random.seed(42)

# ---------------------------------------------------------------------------
# 1. MATCH HISTORY DATA
# Format: (team, opponent, goals_for, goals_against, year, stage)
# stage in {"group", "knockout"}
# ---------------------------------------------------------------------------

MATCHES = [
    # ===================== 2026 WORLD CUP =====================
    # --- France (Group I winner) ---
    ("France", "Senegal", 3, 1, 2026, "group"),
    ("France", "Iraq", 3, 0, 2026, "group"),
    ("France", "Norway", 4, 1, 2026, "group"),
    ("France", "Sweden", 3, 0, 2026, "knockout"),      # R32
    ("France", "Paraguay", 1, 0, 2026, "knockout"),    # R16

    # --- Norway (Group I runner-up) ---
    ("Norway", "Iraq", 4, 1, 2026, "group"),
    ("Norway", "Senegal", 3, 2, 2026, "group"),
    ("Norway", "France", 1, 4, 2026, "group"),
    ("Norway", "Ivory Coast", 2, 1, 2026, "knockout"),  # R32
    ("Norway", "Brazil", 2, 1, 2026, "knockout"),       # R16

    # --- England (Group L winner) ---
    ("England", "Croatia", 4, 2, 2026, "group"),
    ("England", "Ghana", 0, 0, 2026, "group"),
    ("England", "Panama", 2, 0, 2026, "group"),
    ("England", "DR Congo", 2, 1, 2026, "knockout"),   # R32
    ("England", "Mexico", 3, 2, 2026, "knockout"),     # R16

    # --- Spain (Group E winner) ---
    ("Spain", "Cape Verde", 0, 0, 2026, "group"),
    ("Spain", "Saudi Arabia", 4, 0, 2026, "group"),
    ("Spain", "Uruguay", 1, 0, 2026, "group"),
    ("Spain", "Austria", 3, 0, 2026, "knockout"),      # R32
    ("Spain", "Portugal", 1, 0, 2026, "knockout"),     # R16

    # --- Belgium (Group G winner) ---
    ("Belgium", "Egypt", 1, 1, 2026, "group"),
    ("Belgium", "Iran", 0, 0, 2026, "group"),
    ("Belgium", "New Zealand", 5, 1, 2026, "group"),
    ("Belgium", "Senegal", 3, 2, 2026, "knockout"),    # R32 (aet)
    ("Belgium", "United States", 4, 1, 2026, "knockout"),  # R16

    # --- Morocco (Group C runner-up) ---
    ("Morocco", "Brazil", 1, 1, 2026, "group"),
    ("Morocco", "Scotland", 1, 0, 2026, "group"),
    ("Morocco", "Haiti", 4, 2, 2026, "group"),
    ("Morocco", "Netherlands", 1, 1, 2026, "knockout"),  # R32 (won on pens, treated as draw for goal model)
    ("Morocco", "Canada", 3, 0, 2026, "knockout"),       # R16

    # --- Egypt (Group G runner-up) ---
    ("Egypt", "Belgium", 1, 1, 2026, "group"),
    ("Egypt", "New Zealand", 3, 1, 2026, "group"),
    ("Egypt", "Iran", 1, 1, 2026, "group"),
    ("Egypt", "Australia", 1, 1, 2026, "knockout"),      # R32 (won on pens)

    # --- Argentina (Group J winner) ---
    ("Argentina", "Algeria", 3, 0, 2026, "group"),
    ("Argentina", "Austria", 2, 0, 2026, "group"),
    ("Argentina", "Jordan", 3, 1, 2026, "group"),
    ("Argentina", "Cape Verde", 3, 2, 2026, "knockout"),  # R32

    # --- Switzerland (Group B winner) ---
    ("Switzerland", "Qatar", 1, 1, 2026, "group"),
    ("Switzerland", "Bosnia and Herzegovina", 4, 1, 2026, "group"),
    ("Switzerland", "Canada", 2, 1, 2026, "group"),
    ("Switzerland", "Algeria", 2, 0, 2026, "knockout"),   # R32

    # --- Colombia (Group K winner) ---
    ("Colombia", "Uzbekistan", 3, 1, 2026, "group"),
    ("Colombia", "DR Congo", 1, 0, 2026, "group"),
    ("Colombia", "Portugal", 0, 0, 2026, "group"),
    ("Colombia", "Ghana", 1, 0, 2026, "knockout"),        # R32

    # ===================== 2022 WORLD CUP (secondary prior) =====================
    # --- Argentina (2022 champion) ---
    ("Argentina", "Saudi Arabia", 1, 2, 2022, "group"),
    ("Argentina", "Mexico", 2, 0, 2022, "group"),
    ("Argentina", "Poland", 2, 0, 2022, "group"),
    ("Argentina", "Australia", 2, 1, 2022, "knockout"),
    ("Argentina", "Netherlands", 3, 3, 2022, "knockout"),
    ("Argentina", "Croatia", 3, 0, 2022, "knockout"),
    ("Argentina", "France", 3, 3, 2022, "knockout"),

    # --- France (2022 runner-up) ---
    ("France", "Australia", 4, 1, 2022, "group"),
    ("France", "Denmark", 2, 1, 2022, "group"),
    ("France", "Tunisia", 0, 1, 2022, "group"),
    ("France", "Poland", 3, 1, 2022, "knockout"),
    ("France", "England", 2, 1, 2022, "knockout"),
    ("France", "Morocco", 2, 0, 2022, "knockout"),
    ("France", "Argentina", 3, 3, 2022, "knockout"),

    # --- Morocco (2022 semifinalist) ---
    ("Morocco", "Croatia", 0, 0, 2022, "group"),
    ("Morocco", "Belgium", 2, 0, 2022, "group"),
    ("Morocco", "Canada", 2, 1, 2022, "group"),
    ("Morocco", "Spain", 0, 0, 2022, "knockout"),
    ("Morocco", "Portugal", 1, 0, 2022, "knockout"),
    ("Morocco", "France", 0, 2, 2022, "knockout"),
    ("Morocco", "Croatia", 1, 2, 2022, "knockout"),

    # --- England (2022 quarterfinalist) ---
    ("England", "Iran", 6, 2, 2022, "group"),
    ("England", "United States", 0, 0, 2022, "group"),
    ("England", "Wales", 3, 0, 2022, "group"),
    ("England", "Senegal", 3, 0, 2022, "knockout"),
    ("England", "France", 1, 2, 2022, "knockout"),

    # --- Spain (2022 R16) ---
    ("Spain", "Costa Rica", 7, 0, 2022, "group"),
    ("Spain", "Germany", 1, 1, 2022, "group"),
    ("Spain", "Japan", 1, 2, 2022, "group"),
    ("Spain", "Morocco", 0, 0, 2022, "knockout"),

    # --- Belgium (2022 group stage exit) ---
    ("Belgium", "Canada", 1, 0, 2022, "group"),
    ("Belgium", "Morocco", 0, 2, 2022, "group"),
    ("Belgium", "Croatia", 0, 0, 2022, "group"),

    # --- Switzerland (2022 R16) ---
    ("Switzerland", "Cameroon", 1, 0, 2022, "group"),
    ("Switzerland", "Brazil", 0, 1, 2022, "group"),
    ("Switzerland", "Serbia", 3, 2, 2022, "group"),
    ("Switzerland", "Portugal", 1, 6, 2022, "knockout"),

    # Egypt, Norway, Colombia did not qualify for the 2022 World Cup —
    # they run on 2026 data only, which is consistent with the "bigger
    # weight to this year" approach.
]

# Remaining fixtures to predict, in bracket order
REMAINING_R16 = [
    ("Egypt", "Argentina"),
    ("Switzerland", "Colombia"),
]

# QF pairings (France/Morocco and Spain/Belgium and Norway/England are fixed;
# the 4th QF depends on today's two R16 results)
def build_qf(r16_winners):
    egy_arg_winner = r16_winners[("Egypt", "Argentina")]
    sui_col_winner = r16_winners[("Switzerland", "Colombia")]
    return [
        ("France", "Morocco"),
        ("Spain", "Belgium"),
        ("Norway", "England"),
        (egy_arg_winner, sui_col_winner),
    ]

# SF pairing rule from the official bracket:
# SF1 = winner(France/Morocco) vs winner(Spain/Belgium)
# SF2 = winner(Norway/England) vs winner(QF4)
def build_sf(qf_winners):
    return [
        (qf_winners[0], qf_winners[1]),
        (qf_winners[2], qf_winners[3]),
    ]

# ---------------------------------------------------------------------------
# 2. WEIGHTING
# ---------------------------------------------------------------------------

def match_weight(year, stage):
    if year == 2026:
        return 1.4 if stage == "knockout" else 1.0
    else:  # 2022
        return 0.40 if stage == "knockout" else 0.30

# ---------------------------------------------------------------------------
# 3. BUILD ATTACK / DEFENSE RATINGS
# ---------------------------------------------------------------------------

# `teams` = the 10 squads still alive, whose ratings we actually report and
# simulate forward with. `all_teams` is broader: every team that appears
# anywhere in MATCHES, including group-stage victims like Senegal or Iraq.
# We fit ratings for all_teams too, because an alive team's attack rating
# should account for the *quality of the defense it scored against* — that
# requires an opponent rating even for teams that are already eliminated.
teams = set()
all_teams = set()
for t, o, gf, ga, yr, stage in MATCHES:
    teams.add(t)
    all_teams.add(t)
    all_teams.add(o)

all_goals = []
all_weights = []
for t, o, gf, ga, yr, stage in MATCHES:
    w = match_weight(yr, stage)
    all_goals.append(gf)
    all_goals.append(ga)
    all_weights.append(w)
    all_weights.append(w)
league_avg_goals = sum(g * w for g, w in zip(all_goals, all_weights)) / sum(all_weights)

# ---------------------------------------------------------------------------
# 3a. MAXIMUM LIKELIHOOD FIT (iterative proportional fitting)
# ---------------------------------------------------------------------------
# Model: goals scored by team i against team j ~ Poisson(league_avg_goals *
# attack_i * defense_j). We want the attack/defense values that maximize the
# (weighted) log-likelihood of every scoreline actually observed, instead of
# just averaging each team's own goals — this is the same model Dixon-Coles
# uses. Differentiating the weighted Poisson log-likelihood with respect to
# a single attack_i (holding all defense_j fixed) has a closed form:
#
#   attack_i = sum(w * goals_scored_by_i)  /  (league_avg * sum(w * defense_of_opponent))
#
# and symmetrically for defense_j. So we alternate: update every attack_i
# from the current defense values, then update every defense_j from the
# current attack values, and repeat. This is "iterative proportional
# fitting" (IPF) — each step is an exact coordinate-wise MLE update, so the
# joint log-likelihood only ever goes up, and it converges in well under a
# second with no external optimizer (scipy, etc.) required.
#
# A small Bayesian-style prior anchors each team to attack = defense = 1.0
# (league average) worth PRIOR_PSEUDO_WEIGHT "matches" against a virtual
# average opponent. This is what stops a team with 4-5 games from getting an
# implausibly extreme rating off a small sample, while letting a team with
# strong, consistent evidence pull away from average.

AVG = "__LEAGUE_AVERAGE__"      # virtual opponent: attack = defense = 1.0, fixed
PRIOR_PSEUDO_WEIGHT = 3.0       # equivalent to ~3 extra "average" matches

scored_obs = defaultdict(list)    # team -> [(opponent, goals_scored, weight), ...]
conceded_obs = defaultdict(list)  # team -> [(scorer, goals_conceded, weight), ...]

for t, o, gf, ga, yr, stage in MATCHES:
    w = match_weight(yr, stage)
    scored_obs[t].append((o, gf, w))
    conceded_obs[o].append((t, gf, w))
    scored_obs[o].append((t, ga, w))
    conceded_obs[t].append((o, ga, w))

for t in all_teams:
    scored_obs[t].append((AVG, league_avg_goals, PRIOR_PSEUDO_WEIGHT))
    conceded_obs[t].append((AVG, league_avg_goals, PRIOR_PSEUDO_WEIGHT))

attack = {t: 1.0 for t in all_teams}
defense = {t: 1.0 for t in all_teams}
attack[AVG] = 1.0
defense[AVG] = 1.0

N_FIT_ITERS = 200
for _ in range(N_FIT_ITERS):
    new_attack = {}
    for t in all_teams:
        num = sum(w * g for (opp, g, w) in scored_obs[t])
        den = league_avg_goals * sum(w * defense[opp] for (opp, g, w) in scored_obs[t])
        new_attack[t] = num / den if den > 0 else 1.0

    new_defense = {}
    for t in all_teams:
        num = sum(w * g for (scorer, g, w) in conceded_obs[t])
        den = league_avg_goals * sum(w * attack[scorer] for (scorer, g, w) in conceded_obs[t])
        new_defense[t] = num / den if den > 0 else 1.0

    attack.update(new_attack)
    defense.update(new_defense)

    # Attack/defense have a "gauge freedom": multiplying every attack_i by c
    # and dividing every defense_j by c leaves every attack_i * defense_j
    # product (and hence every predicted score) unchanged. We remove that
    # freedom each iteration by re-centering attack's geometric mean at 1.0,
    # so ratings stay directly comparable to the old "1.00 = average" scale.
    log_mean_attack = sum(math.log(attack[t]) for t in all_teams) / len(all_teams)
    gauge = math.exp(log_mean_attack)
    for t in all_teams:
        attack[t] /= gauge
        defense[t] *= gauge

# Weighted Poisson log-likelihood of the real (non-prior) data under the
# fitted model, printed as evidence the fit converged to something sensible.
fit_log_likelihood = 0.0
for t, o, gf, ga, yr, stage in MATCHES:
    w = match_weight(yr, stage)
    lam_t = league_avg_goals * attack[t] * defense[o]
    lam_o = league_avg_goals * attack[o] * defense[t]
    fit_log_likelihood += w * (gf * math.log(max(lam_t, 1e-9)) - lam_t)
    fit_log_likelihood += w * (ga * math.log(max(lam_o, 1e-9)) - lam_o)

# ---------------------------------------------------------------------------
# 3b. SANITY CHECK: does the MLE fit actually beat the naive ratio method?
# ---------------------------------------------------------------------------
# The naive method (attack = goals_scored / average) only ever produces a
# rating for the 10 alive teams, since it never looks at opponents at all.
# That means it can only be *evaluated* fairly on matches where BOTH sides
# are alive teams (otherwise it has no rating for one side, e.g. Senegal).
# We recompute the naive ratings here, then compare log-likelihood on that
# alive-vs-alive subset against the MLE model on the exact same matches —
# an apples-to-apples check that the extra machinery earns its keep.

goals_for_naive = defaultdict(float)
goals_against_naive = defaultdict(float)
weight_sum_naive = defaultdict(float)
for t, o, gf, ga, yr, stage in MATCHES:
    w = match_weight(yr, stage)
    goals_for_naive[t] += gf * w
    goals_against_naive[t] += ga * w
    weight_sum_naive[t] += w

attack_naive, defense_naive = {}, {}
for t in teams:
    total_w = weight_sum_naive[t] + PRIOR_PSEUDO_WEIGHT
    avg_scored = (goals_for_naive[t] + PRIOR_PSEUDO_WEIGHT * league_avg_goals) / total_w
    avg_conceded = (goals_against_naive[t] + PRIOR_PSEUDO_WEIGHT * league_avg_goals) / total_w
    attack_naive[t] = avg_scored / league_avg_goals
    defense_naive[t] = avg_conceded / league_avg_goals

comparable_matches = [m for m in MATCHES if m[0] in teams and m[1] in teams]

naive_ll, mle_ll = 0.0, 0.0
for t, o, gf, ga, yr, stage in comparable_matches:
    w = match_weight(yr, stage)

    lam_t_naive = league_avg_goals * attack_naive[t] * defense_naive[o]
    lam_o_naive = league_avg_goals * attack_naive[o] * defense_naive[t]
    naive_ll += w * (gf * math.log(max(lam_t_naive, 1e-9)) - lam_t_naive)
    naive_ll += w * (ga * math.log(max(lam_o_naive, 1e-9)) - lam_o_naive)

    lam_t_mle = league_avg_goals * attack[t] * defense[o]
    lam_o_mle = league_avg_goals * attack[o] * defense[t]
    mle_ll += w * (gf * math.log(max(lam_t_mle, 1e-9)) - lam_t_mle)
    mle_ll += w * (ga * math.log(max(lam_o_mle, 1e-9)) - lam_o_mle)

print("=" * 70)
print("NAIVE RATIO vs MLE FIT  (log-likelihood, higher/less-negative is better)")
print("=" * 70)
print(f"Compared on {len(comparable_matches)} matches where both sides are alive teams")
print(f"  Naive ratio method (attack = goals_scored / average): {naive_ll:8.2f}")
print(f"  MLE / Dixon-Coles style fit (accounts for opponent):  {mle_ll:8.2f}")
print(f"  MLE improvement: {mle_ll - naive_ll:+.2f} log-likelihood "
      f"({100 * (mle_ll - naive_ll) / abs(naive_ll):+.1f}%)")

print("\n" + "=" * 70)
print("TEAM STRENGTH RATINGS  (1.00 = tournament average, fit by MLE)")
print("=" * 70)
print(f"{'Team':<14}{'Attack':>10}{'Defense':>10}   (defense < 1.00 is good)")
for t in sorted(teams, key=lambda x: -attack[x]):
    print(f"{t:<14}{attack[t]:>10.2f}{defense[t]:>10.2f}")
print(f"\nLeague-average goals/team/match used for scaling: {league_avg_goals:.2f}")
print(f"Model fit (weighted Poisson log-likelihood, higher is better): {fit_log_likelihood:.1f}")

# ---------------------------------------------------------------------------
# 3b. EXPECTED GOALS (λ) FOR EACH MATCHUP
# This is the Poisson mean each side's goal count is drawn from, derived
# directly from the attack/defense ratings above. Printed here so you can see
# exactly what the simulator is working with before the Monte Carlo runs.
# ---------------------------------------------------------------------------

UPCOMING_MATCHUPS = [
    ("Egypt", "Argentina"),
    ("Switzerland", "Colombia"),
    ("France", "Morocco"),
    ("Spain", "Belgium"),
    ("Norway", "England"),
]

def team_lambda(team, opponent):
    lam = league_avg_goals * attack[team] * defense[opponent]
    return max(lam, 0.05)

print("\n" + "=" * 70)
print("EXPECTED GOALS (λ) — Poisson mean per matchup")
print("=" * 70)
for a, b in UPCOMING_MATCHUPS:
    lam_a = team_lambda(a, b)
    lam_b = team_lambda(b, a)
    print(f"{a:<14} λ={lam_a:5.2f}   vs   λ={lam_b:5.2f}   {b}")

# ---------------------------------------------------------------------------
# 4. MATCH SIMULATION
# ---------------------------------------------------------------------------

MAX_GOALS = 8

def poisson_sample(lam):
    # simple Knuth-style sampler, fine for our lambda range
    L = math.exp(-lam)
    k = 0
    p = 1.0
    while True:
        k += 1
        p *= random.random()
        if p <= L:
            return k - 1

def simulate_regulation(team_a, team_b):
    lam_a = team_lambda(team_a, team_b)
    lam_b = team_lambda(team_b, team_a)
    ga = poisson_sample(lam_a)
    gb = poisson_sample(lam_b)
    return ga, gb

def simulate_knockout(team_a, team_b):
    """Returns the winner, applying extra time + penalties if needed."""
    ga, gb = simulate_regulation(team_a, team_b)
    if ga != gb:
        return team_a if ga > gb else team_b, (ga, gb, "FT")
    # Extra time: ~1/3 the scoring rate of normal time
    lam_a_et = team_lambda(team_a, team_b) / 3
    lam_b_et = team_lambda(team_b, team_a) / 3
    ga_et = poisson_sample(lam_a_et)
    gb_et = poisson_sample(lam_b_et)
    if ga_et != gb_et:
        return (team_a if ga_et > gb_et else team_b), (ga + ga_et, gb + gb_et, "AET")
    # Penalties: modeled as roughly 50/50, with a small nudge toward
    # the team with the better attack rating (proxy for composure/quality)
    p_a = 0.5 + 0.03 * (attack[team_a] - attack[team_b])
    p_a = min(max(p_a, 0.35), 0.65)
    winner = team_a if random.random() < p_a else team_b
    return winner, (ga + ga_et, gb + gb_et, "PK")

# ---------------------------------------------------------------------------
# 5. MONTE CARLO OVER THE FULL REMAINING BRACKET
# ---------------------------------------------------------------------------

N_SIMS = 20000

reach_qf = defaultdict(int)
reach_sf = defaultdict(int)
reach_final = defaultdict(int)
win_title = defaultdict(int)

r16_winner_count = defaultdict(lambda: defaultdict(int))  # match -> team -> count
qf_winner_count = defaultdict(lambda: defaultdict(int))
sf_winner_count = defaultdict(lambda: defaultdict(int))
final_score_examples = defaultdict(list)

for _ in range(N_SIMS):
    # --- remaining Round of 16 ---
    r16_winners = {}
    for a, b in REMAINING_R16:
        w, score = simulate_knockout(a, b)
        r16_winners[(a, b)] = w
        r16_winner_count[(a, b)][w] += 1

    qf_pairs = build_qf(r16_winners)
    qf_winners = []
    for a, b in qf_pairs:
        w, score = simulate_knockout(a, b)
        qf_winners.append(w)
        qf_winner_count[(a, b)][w] += 1
        reach_qf[a] += 1
        reach_qf[b] += 1

    sf_pairs = build_sf(qf_winners)
    sf_winners = []
    for a, b in sf_pairs:
        w, score = simulate_knockout(a, b)
        sf_winners.append(w)
        sf_winner_count[(a, b)][w] += 1
        reach_sf[a] += 1
        reach_sf[b] += 1

    final_a, final_b = sf_winners
    champion, score = simulate_knockout(final_a, final_b)
    reach_final[final_a] += 1
    reach_final[final_b] += 1
    win_title[champion] += 1

# ---------------------------------------------------------------------------
# 6. REPORT
# ---------------------------------------------------------------------------

def pct(count):
    return 100.0 * count / N_SIMS

print("\n" + "=" * 70)
print("TODAY'S REMAINING ROUND OF 16 MATCHES  (July 7, 2026)")
print("=" * 70)
for a, b in REMAINING_R16:
    counts = r16_winner_count[(a, b)]
    print(f"{a} vs {b}:")
    for team in (a, b):
        print(f"   {team:<14} win probability: {pct(counts[team]):5.1f}%")

print("\n" + "=" * 70)
print("QUARTERFINALS")
print("=" * 70)
# Note: 4th QF pairing is probabilistic since it depends on today's results,
# so we report it split by the most likely combination.
fixed_qfs = [("France", "Morocco"), ("Spain", "Belgium"), ("Norway", "England")]
for a, b in fixed_qfs:
    counts = qf_winner_count[(a, b)]
    print(f"{a} vs {b}:")
    for team in (a, b):
        print(f"   {team:<14} win probability: {pct(counts[team]):5.1f}%")

print("\n4th Quarterfinal (winner of Egypt/Argentina vs winner of Switzerland/Colombia):")
combo_counts = defaultdict(int)
for (a, b), counts in qf_winner_count.items():
    if (a, b) in fixed_qfs:
        continue
    for team, c in counts.items():
        combo_counts[team] += c
total_combo = sum(combo_counts.values())
for team, c in sorted(combo_counts.items(), key=lambda x: -x[1]):
    print(f"   Reaches semis as this QF's winner: {team:<14} {100*c/total_combo:5.1f}% (of simulations where they got there)")

print("\n" + "=" * 70)
print("PROBABILITY OF REACHING EACH STAGE (out of the 8 remaining teams' paths)")
print("=" * 70)
print(f"{'Team':<14}{'Reach QF':>10}{'Reach SF':>10}{'Reach Final':>12}{'Win Title':>12}")
all_relevant = ["France", "Morocco", "Spain", "Belgium", "Norway", "England",
                "Egypt", "Argentina", "Switzerland", "Colombia"]
for t in all_relevant:
    print(f"{t:<14}{pct(reach_qf[t]):>9.1f}%{pct(reach_sf[t]):>9.1f}%{pct(reach_final[t]):>11.1f}%{pct(win_title[t]):>11.1f}%")

