from enum import Enum

class DataType(Enum):
    e = "Extract"
    c = "Calculate"

class VerticalBarType(Enum):
    sim = "simple"
    stk = "stacked"
    grp = "grouped"

class BarType(Enum):
    ver = "verticalbar"
    hor = "horizontalbar"
    und = "Undetermined"

class LegendPosition(Enum):
    hor = "Horizontal"
    ver = "Vertical"
    err = "Legend error"

class ModelUsed(Enum):
    inh = "InHouse"
    llm = "Vision Model"


