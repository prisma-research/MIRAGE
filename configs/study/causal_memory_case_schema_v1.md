# Causal Memory Case Schema v1

## Status Note (2026-03-29)

This schema records an internal, generation-oriented representation.

It still keeps explicit fields for:

- history prefix
- compact-memory variants
- condition-aware internals

because these are useful for:

- procedural data generation
- optional sanity checks
- future appendix analyses

However, the current canonical public-facing experiment framing has already shifted to:

- `T1/T2/T3`
- `M0/M1/M2/M3`
- `T × M` as the main matrix

So this schema should be read as:

- **generation internals**

not as the final paper-facing symbolic system.

**T × M ↔ Internal mapping**: The public experiment uses `T1/T2/T3` (task types) × `M0/M1/M2/M3` (memory quality).
Internal generation fields like `C1_history_prefix`, `C2_memory_*`, timeline events are pipeline intermediates.
The `conditions` block maps to `M0/M1/M2/M3` for the public experiment; `C0/C1` are optional sanity checks only.

## Case Graph Structure (Generation Internals)

Each case is fully determined by a **user profile** + **timeline** + **page spec**.
All surface data (history prefix, compact memory, gold action) are derived deterministically.

```
case
├── case_id: str                      # e.g. "booking_seat_t2_003"
├── domain: str                       # "booking" | "food" | "sharing"
├── task_type: str                    # "T1" | "T2" | "T3"
│
├── user_profile                      # latent: who is the user?
│   ├── stable_preferences: dict      # long-term stable facts
│   │   e.g. {"seat": "aisle", "reason": "knee injury"}
│   ├── updated_preferences: dict     # only for T2: new values covering old
│   │   e.g. {"seat_old": "window", "seat_new": "aisle", "update_reason": "knee injury"}
│   ├── context_rules: dict           # only for T3: conditional boundaries
│   │   e.g. {"rule": "no_work_to_family", "scope": "sharing"}
│   └── general_traits: list[str]     # 3-5 stable general facts (populate memory)
│       e.g. ["Avoids reminders after 21:30", "Prefers concise confirmations"]
│
├── timeline                          # latent: what happened over time?
│   ├── events: list[Event]
│   │   ├── Event(time="t1", type="establish", state_changes={...}, interaction_text="...")
│   │   ├── Event(time="t2", type="update", state_changes={...}, interaction_text="...")
│   │   └── Event(time="t3", type="rule", state_changes={...}, interaction_text="...")
│   └── test_timepoint: str           # always after latest event
│
├── conditions                        # surface: what agent sees per condition
│   ├── C0_input: None                # no history, no memory
│   ├── C1_history_prefix: str        # chronological replay of events ≤ timepoint
│   ├── C2_memory_correct: str        # MEMORY.md — current effective user state
│   ├── C2_memory_stale: str          # MEMORY.md — missing t2 update (T2 only)
│   └── C2_memory_noisy: str          # MEMORY.md — correct + irrelevant traits (T3 only)
│
├── page_spec                         # latent: what page is shown?
│   ├── template_id: str              # e.g. "seat_selector"
│   ├── slots: dict                   # template-specific values
│   └── rendered_image_path: str      # path to PNG
│
├── task_prompt: str                  # question about the page
├── decision_space: list[str]         # valid actions (always includes ASK_USER)
│
├── gold_action: dict                 # per-condition correct action
│   ├── C0: str                       # often ASK_USER (insufficient info for personalization)
│   ├── C1: str                       # specific action (full history available)
│   ├── C2-correct: str               # = C1 gold (compact memory sufficient)
│   ├── C2-stale: str                 # = C1 gold (still objectively correct)
│   └── C2-noisy: str                 # = C1 gold (noise should not change correct answer)
│
├── stale_aligned_action: str         # template-derived: what a stale-trusting agent would pick
│                                     # = determine_gold(stale_state, page_spec)
├── noisy_aligned_action: str | null  # template-derived: what an over-personalizing agent picks
│                                     # T3 out-of-scope: = rule's constrained_action (wrongly applied)
│                                     # T1/T2: = random non-gold from decision_space, or null
│
└── t3_scope: str | null              # T3 only: "in_scope" or "out_of_scope"
```

## MEMORY.md Content Structure

```markdown
# Memory

## General
- [general_trait_1]
- [general_trait_2]
- [general_trait_3]

## Current Preferences
- [task_relevant_preference_1]
- [task_relevant_preference_2]

## Boundaries
- [rule_or_constraint]  (if applicable)
```

All content is **user information state**, not conversation summary.

## Field Generation Rules

| Field | Generation Method |
|-------|-------------------|
| user_profile | Template × random sampling from domain-specific pools |
| timeline | Fixed 3-stage structure per task type |
| C1_history_prefix | `format_history_prefix(timeline, timepoint)` |
| C2_memory_correct | `compile_memory(timeline, timepoint)` |
| C2_memory_stale | `compile_memory(timeline, timepoint, skip_update=True)` |
| C2_memory_noisy | `compile_memory(...) + general_traits` extended with extra irrelevant |
| page_spec.slots | From user state + domain template |
| rendered_image | `render_page(template_id, slots)` → PNG |
| task_prompt | Domain template with slot interpolation |
| gold_action | `{C0: determine_c0_gold(...), C1: determine_gold(current_state, ...), ...}` |
| stale_aligned_action | `determine_gold(stale_state, page_spec)` — template-derived |
| noisy_aligned_action | T3 out-of-scope: `rule.constrained_action`; T1/T2: `sample_non_gold(decision_space, gold)` or null |
| t3_scope | T3 only: "in_scope" if page matches rule scope, "out_of_scope" otherwise |

## Event Schema

```json
{
  "time": "t1",
  "type": "establish",
  "state_changes": {"seat_pref": "window", "seat_reason": "likes the view"},
  "interaction_text": "User: I'm booking a flight next week. I usually like window seats.\nAgent: Noted, I'll keep that in mind for your bookings."
}
```

## Timeline Templates by Task Type

### T1 (Stable Preference)
```
t1: preference established → remains stable → test at t_now
```
Only t1 event. Memory and history both reflect the same stable fact.

### T2 (Preference Update)
```
t1: old preference established
t2: new preference overrides old
test at t_now (after t2)
```
C2-correct has new preference. C2-stale has old preference.

### T3 (Selective / Context-Bounded) — Paired Cases

T3 generates **paired cases** from each context rule: one in-scope, one out-of-scope.

```
t1: general preference established
t3: context rule introduced (scoped boundary)
test at t_now: TWO cases generated:
  - case_A (in-scope):  page matches rule scope → rule should apply
  - case_B (out-of-scope): page similar but outside rule scope → rule should NOT apply
```

#### Pairing Structure

Each T3 rule template defines:
- `scope_conditions`: what makes a page fall within the rule (e.g., content=work AND target=family)
- `in_scope_slots`: page slots that satisfy scope conditions
- `out_scope_slots`: page slots that do NOT satisfy scope conditions (but look similar)
- `constrained_action`: the action the rule dictates (used as gold for in-scope, as noisy_aligned for out-of-scope)
- `default_action`: the normal action without the rule (used as gold for out-of-scope)

#### Example Rule: "Do not share work content to family groups"

| | In-Scope Case | Out-of-Scope Case |
|---|---|---|
| Page content | Work screenshot | Personal photo |
| Share target | Family group | Family group |
| Rule applies? | Yes | No (personal content, not work) |
| Gold action | COPY_LINK | SHARE |
| noisy_aligned | n/a (in-scope uses rule correctly) | COPY_LINK (wrongly applying work rule to personal content) |
| t3_scope | "in_scope" | "out_of_scope" |

#### OPR Measurement
Over-personalization is measured **only on out-of-scope cases**:
- If agent picks `constrained_action` on out-of-scope page → over-personalization error
- `noisy_aligned_action = constrained_action` for all out-of-scope T3 cases (template-derived, not hand-assigned)

## Trait Pool (Two-Layer, Domain-Safe)

Traits are split into **background** (always safe) and **domain-specific irrelevant** (safe for the given domain only).
This prevents a booking case from accidentally containing seat-related "general" traits that become task-relevant.

### Layer A: Background Traits (domain-neutral, always safe)

```python
BACKGROUND_TRAITS = [
    "Prefers concise confirmations for routine tasks.",
    "Usually prefers email summaries over long calls.",
    "Prefers direct, technical communication.",
    "Does not like unsolicited recommendations.",
    "Dislikes auto-playing media.",
    "Prefers dark mode on all devices.",
    "Working hours: 09:00–22:00 (Shanghai).",
    "Uses metric units.",
]
```

### Layer B: Domain-Specific Irrelevant Traits

```python
DOMAIN_IRRELEVANT = {
    "booking": [
        # Safe for booking: nothing about seats, travel, legroom, flights
        "Avoids scheduling reminders after 21:30.",
        "Prefers dark roast coffee, no sugar.",
        "Gym schedule: Tue/Thu/Sat mornings.",
        "Subscribes to The Pragmatic Engineer newsletter.",
        "Preferred lunch spot: cafeteria on 3F.",
    ],
    "food": [
        # Safe for food: nothing about dietary restrictions, allergies, cuisine
        "Prefers window seats on long flights.",
        "Default sharing mode: copy-link.",
        "Avoids meetings before 10:00.",
        "Personal photo backup: iCloud.",
        "Calendar sync: yes for work, no for personal events.",
    ],
    "sharing": [
        # Safe for sharing: nothing about share policies, privacy, social media
        "Seat preference: aisle (knee injury).",
        "Avoids spicy food (stomach issues).",
        "Gym schedule: Tue/Thu/Sat mornings.",
        "Prefers dark roast coffee, no sugar.",
        "Follows NBA and Premier League scores.",
    ],
}
```

### Sampling Rule

```python
def sample_domain_safe_traits(domain: str, k: int = 3) -> list[str]:
    pool = BACKGROUND_TRAITS + DOMAIN_IRRELEVANT[domain]
    return random.sample(pool, k=min(k, len(pool)))
```

**Constraint**: A trait must NOT appear in both "General" and "Current Preferences" for the same case.

## Example: Complete Case (Booking, T2)

```json
{
  "case_id": "booking_seat_t2_001",
  "domain": "booking",
  "task_type": "T2",
  "user_profile": {
    "stable_preferences": {},
    "updated_preferences": {
      "seat_old": "window",
      "seat_old_reason": "likes the view",
      "seat_new": "aisle",
      "seat_new_reason": "knee injury since March"
    },
    "context_rules": {},
    "general_traits": [
      "Avoids reminders after 21:30.",
      "Prefers concise confirmations.",
      "Uses metric units."
    ]
  },
  "timeline": {
    "events": [
      {
        "time": "t1",
        "type": "establish",
        "state_changes": {"seat_pref": "window", "seat_reason": "likes the view"},
        "interaction_text": "User: I prefer window seats when I fly.\nAgent: Got it, window seat preference noted."
      },
      {
        "time": "t2",
        "type": "update",
        "state_changes": {"seat_pref": "aisle", "seat_reason": "knee injury, needs legroom"},
        "interaction_text": "User: Actually, I need to switch to aisle seats from now on. My knee has been bothering me and I need the extra legroom.\nAgent: Understood, I've updated your preference to aisle seats."
      }
    ],
    "test_timepoint": "t_now"
  },
  "C1_history_prefix": "[2026-03-10] User: I prefer window seats when I fly.\nAgent: Got it, window seat preference noted.\n\n---\n\n[2026-03-20] User: Actually, I need to switch to aisle seats from now on. My knee has been bothering me and I need the extra legroom.\nAgent: Understood, I've updated your preference to aisle seats.",
  "C2_memory_correct": "# Memory\n\n## General\n- Avoids reminders after 21:30.\n- Prefers concise confirmations.\n- Uses metric units.\n\n## Current Preferences\n- Seat preference: aisle (knee injury, needs extra legroom; updated 2026-03-20).\n\n## Boundaries\n- (none currently recorded)",
  "C2_memory_stale": "# Memory\n\n## General\n- Avoids reminders after 21:30.\n- Prefers concise confirmations.\n- Uses metric units.\n\n## Current Preferences\n- Seat preference: window (likes the view).\n\n## Boundaries\n- (none currently recorded)",
  "page_spec": {
    "template_id": "seat_selector",
    "slots": {"options": ["window", "aisle", "middle"], "flight": "SH→BJ", "date": "2026-04-05"}
  },
  "rendered_image_path": "data/generated_personal_ui/booking_seat_t2_001.png",
  "task_prompt": "I'm booking a flight from Shanghai to Beijing. Which seat type should I select?",
  "decision_space": ["WINDOW", "AISLE", "MIDDLE", "ASK_USER"],
  "gold_action": {
    "C0": "ASK_USER",
    "C1": "AISLE",
    "C2-correct": "AISLE",
    "C2-stale": "AISLE",
    "C2-noisy": "AISLE"
  },
  "stale_aligned_action": "WINDOW",
  "noisy_aligned_action": "WINDOW",
  "t3_scope": null
}
```
