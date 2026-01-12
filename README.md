# Stress Computation Service

## Rule-Based Stress Estimation for Education & Healthcare Pilots**

## 1. Overview

This service provides **deterministic, auditable stress estimation** for two distinct pilots:

* **Education pilot** (students, academic scheduling)
* **Healthcare pilot** (nursing staff, shift scheduling)

The system combines:

* physiological signals (HR, HRV, steps),
* self-reported stress,
* contextual / scheduling constraints,
* optional machine-learning outputs,

into a **single stress score (0–100)** with:

* transparent rule logic,
* explicit overrides for high-risk situations,
* confidence estimation for data quality.

> **Important**
> Machine Learning is treated as **one signal**, not as the decision authority.
> Final stress scores are always rule-governed.

## 2. Design Philosophy

### Why rule-based?

* Stress estimation affects **scheduling and wellbeing**
* Decisions must be **explainable**
* Edge cases (exam weeks, night shifts) must not be averaged away
* Missing data must degrade safely

### Core principles

* **Personal baselines**, not population averages
* **Physiology first**, context second, perception third
* **Overrides beat averages**
* **Confidence is explicit**, not implicit

## 3. High-Level Architecture

```
biosignals ─┐
            ├─► Normalization ─► Physiological Stress (PS)
ML model ───┘

questionnaires ─► Self-Reported Stress (SRS)

context ─► Academic Load (ALC) / Contextual Load (CLS)

PS + SRS + ALC/CLS
        └─► Overrides
                └─► Final Aggregation
                        └─► Confidence
```
Each pilot implements the **same pattern**, but with **pilot-specific rules and weights**.

## 4. Endpoints

### 4.1 Education Pilot

```
POST /stress/compute
```

Used for **students** during lectures, labs, exams, or self-study.

### 4.2 Healthcare Pilot

```
POST /stress/compute/healthcare
```

Used for **nursing staff** during work shifts.

---

## 5. Education Pilot — Rules Summary

### 5.1 Inputs

#### Baseline (per student)

* `hr_base` – resting HR mean
* `hrv_base` – RMSSD median
* `steps_base` – median daily steps

#### Session data

* `hr_session`
* `hrv_session` (optional)
* `steps_session`
* `sleep_last_24h` (optional)

#### Questionnaires

* `pre_sr` ∈ {0,1,2}
* `post_sr` ∈ {0,1,2}
* `weekly_sr` ∈ {0,1,2}

#### Academic context

* deadlines within 72h
* hours until next exam
* consecutive sessions
* credit overload
* work hours
* commute duration

---

### 5.2 Core Scores

#### Physiological Stress (PS)

* HR and HRV normalized **per student**
* HRV weighted more heavily than HR
* Physical activity reduces false stress detection
* ML probability used if available, fallback otherwise

#### Self-Reported Stress (SRS)

* Weighted:

  * 50% post-session
  * 35% pre-session
  * 15% weekly

#### Academic Load (ALC)

* Deadline pressure
* Exam proximity
* Sleep debt
* Scheduling density
* External workload

---

### 5.3 Overrides (Education)

Examples:

* Exam < 48h **and** high SRS ⇒ PS ≥ 80
* HRV collapse ⇒ PS + 8
* ≥ 3 deadlines + dense schedule ⇒ ALC + 8
* Severe sleep deprivation ⇒ global offset

Overrides ensure **non-linear risk escalation is not averaged out**.

---

### 5.4 Final Aggregation

```
Overall Stress =
0.50 × PS
0.25 × SRS
0.25 × ALC
```

Classification:

* `0–33` → Low
* `34–66` → Moderate
* `67–100` → High

---

## 6. Healthcare Pilot — Rules Summary

### 6.1 Inputs

#### Baseline (per nurse)

* `hr_base`
* `hrv_base`
* `steps_base` (per 8h shift)

#### Shift data

* `hr_shift`
* `hrv_shift` (optional)
* `steps_shift`

#### Questionnaires

* `pre_sr`, `post_sr`, `weekly_sr`

#### Scheduling context

* `shift_type`: day / evening / night
* `pref_match`: preference satisfied
* consecutive shifts
* hours since last shift
* overtime hours

---

### 6.2 Core Scores

#### Physiological Stress (PS)

Same structure as education pilot, tuned for **occupational fatigue**.

#### Self-Reported Stress (SRS)

Healthcare-specific point mapping:

* Post-shift stress has highest impact
* Reflects acute exhaustion

#### Contextual Load Score (CLS)

Accounts for:

* night shifts
* preference mismatch
* short recovery windows
* consecutive shifts
* overtime

### 6.3 Overrides (Healthcare)

Examples:

* High PS + high post-shift SR ⇒ force high classification
* HRV collapse ⇒ PS + 8
* ≥ 4 night shifts with mismatch ⇒ CLS + 10

Designed to capture **burnout risk**, not just momentary stress.

### 6.4 Final Aggregation

```
Overall Stress =
0.50 × PS
0.30 × SRS
0.20 × CLS
```

---

## 7. Confidence & Data Quality (Both Pilots)

Each response includes a **confidence score ∈ [0.4, 1.0]**.
Confidence penalties:

* HRV missing
* HR missing > 10%
* ML model unavailable (fallback used)
* Missing post-session / post-shift questionnaire
* Missing sleep data (education)

Rule:

```
confidence < 0.6 → needs_review = true
```

This prevents silent reliance on low-quality data.

## 8. API Inputs

### 8.1 Education Input (simplified)

```json
{
  "dataframe_split": { "columns": [...], "data": [...] },
  "hr_base": 65,
  "hrv_base": 42,
  "steps_base": 7000,
  "hr_session": 82,
  "hrv_session": 28,
  "steps_session": 3200,
  "sleep_last_24h": 5.5,
  "pre_sr": 1,
  "post_sr": 2,
  "weekly_sr": 1,
  "deadlines_72h": 2,
  "exam_hours_until": 36,
  "back_to_back_sessions": 3,
  "credit_overload": 3,
  "work_hours_week": 12,
  "commute_minutes_day": 40
}
```

### 8.2 Healthcare Input (simplified)

```json
{
  "dataframe_split": { "columns": [...], "data": [...] },
  "hr_base": 62,
  "hrv_base": 48,
  "steps_base": 9000,
  "hr_shift": 88,
  "hrv_shift": 30,
  "steps_shift": 11000,
  "pre_sr": 1,
  "post_sr": 2,
  "weekly_sr": 1,
  "shift_type": "night",
  "pref_match": false,
  "consecutive_shifts": 4,
  "hours_since_last_shift": 10,
  "overtime_hours": 3
}
```

## 9. API Outputs (Both Pilots)

```json
{
  "stress": {
    "physiological_stress": 82.4,
    "self_reported_stress": 66.5,
    "academic_load": 41.0,
    "overall_stress": 72.6,
    "classification": "high",
    "overrides": ["exam<48h_and_high_SRS"]
  },
  "confidence": 0.82,
  "needs_review": false
}
```

## 10. Versioning & Governance

* Rules are **versioned**, not silently changed
* ML models can be updated **without changing decision logic**
* All outputs are explainable post-hoc

This is intentional and required for **clinical, educational, and EU-regulated environments**.

## 11. Future Extensions

* Stress accumulation over time
* Burnout trajectory detection (healthcare)
* Adaptive scheduling actions
* Pilot-specific dashboards
* Rule weight learning (without removing rules)
