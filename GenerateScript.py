import random

# COMMAND SET
COMMANDS = ["het", "ly", "mo", "eo", "ba", "ndb", "hettox", "hetstab", "hetmet",
            "lyklein", "lymittel", "lygross", "lymag", "moly"]

# NUMBER SET
NUMBERS = ["zwei", "drei", "vier", "f\u00fcnf"]

# HELPER FUNCTION TO GENERATE NORMAL COMMAND
def generate_item():
    category = random.choice(COMMANDS)

    if random.random() < 0.20:
        number = random.choice(NUMBERS)
        return f"{number} {category}"

    return category

# CONFUSION FAMILIES
MO_FAMILY = ["mo", "moly"]
LY_FAMILY = ["ly", "lyklein", "lymittel", "lygross", "lymag"]
HET_FAMILY = ["het", "hettox", "hetstab", "hetmet"]

# HELPER FUNCTION FOR IN-FAMILY CONFUSIONS
def generate_pair_patterns(group):
    patterns = []

    for first in group:
        for second in group:
            patterns.append([first, second])

    return patterns

# HELPER FUNCTION FOR CROSS-FAMILY CONFUSIONS
def generate_cross_patterns(group_a, group_b):
    patterns = []

    for a in group_a:
        for b in group_b:
            patterns.extend([[a, b], [b, a], [a, b, a], [b, a, b]])

    return patterns

# CONFUSION SET
CONFUSION_PATTERNS = []

# IN-FAMILY CONFUSIONS
CONFUSION_PATTERNS += generate_pair_patterns(HET_FAMILY)
CONFUSION_PATTERNS += generate_pair_patterns(LY_FAMILY)
CONFUSION_PATTERNS += generate_pair_patterns(MO_FAMILY)

# CROSS-FAMILY CONFUSIONS
CONFUSION_PATTERNS += generate_cross_patterns(MO_FAMILY, LY_FAMILY)

# DOUBLE UP ON IMPORTANT CONFUSION PHRASES
CONFUSION_PATTERNS += [["mo", "ly"], ["moly"], ["mo", "ly", "moly"], ["moly", "mo", "ly"],
                       ["mo", "ly", "mo", "ly"], ["moly", "moly"], ["mo", "moly", "ly"],
                       ["ly", "mo", "moly"]]

# GENERATE NORMAL SCRIPT
def generate_phrase(min_items=5, max_items=15):
    items = []
    number_of_items = random.randint(min_items, max_items)

    for _ in range(number_of_items):
        items.append(generate_item())

    return " ".join(items)

# GENERATE CONFUSION PHRASE
def generate_confusion_phrase():
    special = random.choice(CONFUSION_PATTERNS)

    before = [generate_item() for _ in range(random.randint(1, 3))]

    after = [generate_item() for _ in range(random.randint(1, 3))]

    return " ".join(before + special + after)

# GENERATE SCRIPT
def get_script():
    if random.random() < 0.30:
        return generate_confusion_phrase()

    return generate_phrase()