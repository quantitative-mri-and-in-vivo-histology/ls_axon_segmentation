from enum import Enum


class DataTag(str, Enum):
    INPUT = "input"
    TARGET = "target"


class ClassTag(str, Enum):
    BACKGROUND = "background"
    IGNORE = "ignore"
