"""Domain templates for the T × M personal memory study.

Each domain defines:
  - slot pools for user-state sampling
  - task templates for T1/T2/T3
  - page template specs
  - update pools (for T2)
  - rule templates (for T3 paired selective-use cases)
  - decision spaces
  - gold-action derivation logic

Domains: booking, food, sharing.
"""

from __future__ import annotations
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Trait pools (two-layer, domain-safe)
# ---------------------------------------------------------------------------

BACKGROUND_TRAITS: list[str] = [
    "Prefers concise confirmations for routine tasks.",
    "Usually prefers email summaries over long calls.",
    "Prefers direct, technical communication.",
    "Does not like unsolicited recommendations.",
    "Dislikes auto-playing media.",
    "Prefers dark mode on all devices.",
    "Working hours: 09:00–22:00 (Shanghai).",
    "Uses metric units.",
]

DOMAIN_IRRELEVANT: dict[str, list[str]] = {
    "booking": [
        "Avoids scheduling reminders after 21:30.",
        "Prefers dark roast coffee, no sugar.",
        "Gym schedule: Tue/Thu/Sat mornings.",
        "Subscribes to The Pragmatic Engineer newsletter.",
        "Preferred lunch spot: cafeteria on 3F.",
    ],
    "food": [
        "Prefers window seats on long flights.",
        "Default sharing mode: copy-link.",
        "Avoids meetings before 10:00.",
        "Personal photo backup: iCloud.",
        "Calendar sync: yes for work, no for personal events.",
    ],
    "sharing": [
        # Generic non-sharing traits safe for sharing domain
        "Seat preference: aisle (knee injury).",
        "Avoids spicy food (stomach issues).",
        "Gym schedule: Tue/Thu/Sat mornings.",
        "Prefers dark roast coffee, no sugar.",
        "Follows NBA and Premier League scores.",
    ],
}

# Sharing-specific domain-adjacent distractors (separate pool for M3).
# Tagged with category so mitigation (M3-mit) can do structured pruning:
#   "sharing_adjacent" = domain-adjacent, KEPT by distractor pruning
#   "general" = not sharing-related, PRUNED by distractor pruning
SHARING_DISTRACTORS_TAGGED: list[dict] = [
    {"text": "Likes sharing travel photos with family group chats.", "tag": "sharing_adjacent"},
    {"text": "Usually forwards restaurant screenshots to close friends.", "tag": "sharing_adjacent"},
    {"text": "Comfortable sharing non-work content publicly on social media.", "tag": "sharing_adjacent"},
    {"text": "Prefers WeChat Moments for personal life updates.", "tag": "sharing_adjacent"},
    {"text": "Often shares interesting articles via Twitter.", "tag": "sharing_adjacent"},
    {"text": "Sends meeting notes to team via email, not chat.", "tag": "sharing_adjacent"},
    {"text": "Enjoys posting weekend hiking photos on social media.", "tag": "sharing_adjacent"},
]

# Flat list for backward compat
SHARING_DISTRACTORS: list[str] = [d["text"] for d in SHARING_DISTRACTORS_TAGGED]

# Tag ontology for distractor items.
# Convention: "{domain}_adjacent" for domain-specific distractors.
# Generator MUST use these exact tags. Mitigation filter consumes them.
DISTRACTOR_TAG_DOMAIN_ADJACENT = "{domain}_adjacent"  # template
DISTRACTOR_TAG_GENERAL = "general"

# Relevance tags that M3-mit should KEEP per domain.
# Items matching these tags are domain-adjacent and preserved.
# Items tagged "general" are pruned as irrelevant noise.
DOMAIN_RELEVANT_TAGS: dict[str, set[str]] = {
    "sharing": {"sharing_adjacent"},
    "booking": {"booking_adjacent"},
    "food": {"food_adjacent"},
}

# ---------------------------------------------------------------------------
# Domain: Booking / Seat Selection
# ---------------------------------------------------------------------------

BOOKING_TEMPLATE = {
    "domain": "booking",
    "page_template_id": "seat_selector",
    "decision_space": ["WINDOW", "AISLE", "MIDDLE", "ASK_USER"],

    # T1: stable preference
    "t1_stable_pool": [
        {"value": "aisle", "reason": "knee injury, needs extra legroom"},
        {"value": "aisle", "reason": "prefers quick access to restroom on long flights"},
        {"value": "window", "reason": "likes the view and leaning against the wall to sleep"},
        {"value": "window", "reason": "gets anxious in the middle, prefers wall side"},
    ],

    # T2: preference update
    "t2_update_pool": [
        {"old": "window", "old_reason": "likes the view",
         "new": "aisle", "new_reason": "knee injury since March, needs extra legroom"},
        {"old": "window", "old_reason": "likes leaning against wall",
         "new": "aisle", "new_reason": "started doing more short-haul flights, prefers quick exit"},
        {"old": "aisle", "old_reason": "prefers quick access",
         "new": "window", "new_reason": "knee recovered, back to preferring window views"},
    ],

    "task_prompts": [
        "I'm booking a flight. Which seat type should I select?",
        "Help me choose a seat for my upcoming flight.",
        "Which seat type would be best for me on this flight?",
    ],
}

# ---------------------------------------------------------------------------
# Domain: Food / Dietary
# ---------------------------------------------------------------------------

FOOD_TEMPLATE = {
    "domain": "food",
    "page_template_id": "dish_order",
    "decision_space": [],  # filled per page from dish names + ASK_USER

    "t1_stable_pool": [
        {"restriction": "no_spicy", "reason": "chronic stomach issues"},
        {"restriction": "no_dairy", "reason": "lactose intolerant"},
        {"restriction": "no_shellfish", "reason": "allergic to shellfish"},
        {"restriction": "vegetarian", "reason": "personal dietary choice"},
    ],

    "t2_update_pool": [
        {"old_restriction": "none", "old_reason": "no restrictions",
         "new_restriction": "no_spicy", "new_reason": "developed stomach issues in March"},
        {"old_restriction": "none", "old_reason": "no restrictions",
         "new_restriction": "no_dairy", "new_reason": "diagnosed lactose intolerant recently"},
        {"old_restriction": "no_spicy", "old_reason": "stomach issues",
         "new_restriction": "none", "new_reason": "stomach recovered, doctor cleared all food"},
    ],

    "task_prompts": [
        "I'm ordering lunch. Which dish should I pick?",
        "Help me choose from this menu.",
        "Which of these dishes would be best for me?",
    ],
}

# ---------------------------------------------------------------------------
# Domain: Sharing / Privacy
# ---------------------------------------------------------------------------

SHARING_TEMPLATE = {
    "domain": "sharing",
    "page_template_id": "share_sheet",
    "decision_space": [],  # filled per page from share targets + ASK_USER

    "t1_stable_pool": [
        {"rule": "copy_link_for_work", "detail": "Use Copy Link for work-related content."},
        {"rule": "wechat_for_team", "detail": "Share work content via WeChat to the team."},
    ],

    # T3 rule templates (paired selective-use cases)
    "t3_rule_templates": [
        {
            "rule": "Do not share work content to family groups.",
            "scope": {"content_type": "work"},
            "constrained_action": "COPY_LINK",
            "default_action": "SHARE",
            "in_scope_slots": {"content_type": "work_screenshot", "target": "family_group"},
            "out_scope_slots": {"content_type": "personal_photo", "target": "family_group"},
        },
        {
            "rule": "Do not share work screenshots on public social media.",
            "scope": {"content_type": "work", "target_visibility": "public"},
            "constrained_action": "COPY_LINK",
            "default_action": "SHARE",
            "in_scope_slots": {"content_type": "work_screenshot", "target": "twitter"},
            "out_scope_slots": {"content_type": "personal_blog_post", "target": "twitter"},
        },
    ],

    "task_prompts": [
        "A share sheet appeared. How should I share this?",
        "Which sharing option should I use for this content?",
    ],
}

ALL_DOMAINS = {
    "booking": BOOKING_TEMPLATE,
    "food": FOOD_TEMPLATE,
    "sharing": SHARING_TEMPLATE,
}
