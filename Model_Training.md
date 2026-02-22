# Machine Learning Model Training

## Overview

The recommendation engine is powered by a supervised machine learning model trained using historical ranked match drafts collected through Riot Games APIs.

The model learns relationships between champion selections, role assignments, and match outcomes under specific League of Legends game patches.

---

## Training Dataset

Training samples consist of Ranked Solo Queue matches collected from high-skill tiers.

### Ranked Tiers Included

* Challenger
* Grandmaster
* Master
* Diamond
* Emerald

Each dataset row represents a complete draft composition.

### Dataset Schema

```
match_id
patch

blue_top
blue_jungle
blue_mid
blue_adc
blue_support

red_top
red_jungle
red_mid
red_adc
red_support

blue_win
```

---

### Target Variable

```
blue_win
```

Binary Classification:

* **1 → Blue Team Victory**
* **0 → Red Team Victory**

---

# Feature Sources

Model training combines three primary datasets.

---

## 1. Draft Composition Dataset

Core champion selections from both teams.

Captures:

* Champion combinations.
* Team composition structure.
* Role assignments.

Example Draft:

```
Lee Sin — Jungle
Ahri — Mid
Jinx — ADC
```

This enables the model to learn champion synergy patterns.

---

## 2. Champion Patch Statistics

Generated using matches played during a specific patch.

### Dataset Structure

```
champion
games
wins
winrate
```

### Purpose

Captures patch meta strength.

Example:

If a champion shows consistently high winrate during a patch, predictions reflect that advantage.

---

## 3. Champion Role Statistics

Generated separately from patch matches.

### Dataset Structure

```
champion
role
games
wins
winrate
```

### Purpose

Captures role-specific effectiveness.

Example:

* Lux Support ≠ Lux Mid performance.

---

# Feature Comparisons Learned by Model

The XGBoost model implicitly learns several drafting relationships.

---

## Champion Synergy

Evaluates how champions perform together.

Examples:

* Jungle + Mid roaming combinations.
* ADC + Support lane pairings.

Example:

```
Rakan + Xayah synergy.
```

---

## Role Matchups

Evaluates lane counter relationships.

Examples:

* Top vs Top matchup strength.
* Mid vs Mid matchup advantages.

Counter picks significantly influence win probability.

---

## Patch Meta Strength

Patch statistics introduce bias toward strong champions.

Example:

* Overperforming champions increase predicted win probability.

This allows the model to adapt across patches.

---

## Cross Team Composition Interaction

Evaluates overall team strategy interactions.

Examples:

* Engage vs Poke compositions.
* Scaling vs Early aggression.

Example:

```
Malphite engage composition vs squishy poke team.
```

---

## Team Balance

The model implicitly observes:

* Damage distribution.
* Frontline presence.
* Utility combinations.

These patterns are learned directly from match outcomes.

---

# Model Selection

## Algorithm

XGBoost Gradient Boosted Decision Trees.

### Reasons for Selection

* Strong performance on structured tabular datasets.
* Captures nonlinear feature interactions.
* Robust against sparse categorical encoding.

---

# Prediction Objective

Given:

```
9 champions selected.
```

The system predicts:

```
Probability Blue Team Wins.
```

---

# Recommendation Generation

When a role is locked and nine champions are selected:

The system simulates candidate champions for the remaining slot.

### Process

```
Insert candidate champion
        ↓
Predict win probability
        ↓
Compute delta improvement
```

Top candidates are ranked based on win probability improvement.
