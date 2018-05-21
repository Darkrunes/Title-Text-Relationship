from enum import Enum


class Stances(Enum):
    AGREES = 0
    DISAGREES = 1
    DISCUSSES = 2
    UNRELATED = 4

    stances = {
        'agrees': AGREES,
        'disagrees': DISAGREES,
        'discusses': DISCUSSES,
        'unrelated': UNRELATED,
    }


