from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class Location:
    trap: int
    site: int
    def __repr__(self): return f"({self.trap}, {self.site})"

@dataclass
class QubitDecl:
    id: int
    location: Location

@dataclass
class Operation:
    pass

@dataclass
class GateOp(Operation):
    type: str
    targets: List[int]
    params: List[float]

@dataclass
class MovementOp(Operation):
    type: str
    targets: List[int]
    src: Optional[Location] = None
    dst: Optional[Location] = None

@dataclass
class MeasureOp(Operation):
    targets: List[int]
    type: str = "MEASURE"

@dataclass
class ResetOp(Operation):
    targets: List[int]
    type: str = "RESET"

@dataclass
class BarrierOp(Operation):
    targets: List[int]
    type: str = "BARRIER"

@dataclass
class Program:
    declarations: List[QubitDecl]
    instructions: List[Operation]