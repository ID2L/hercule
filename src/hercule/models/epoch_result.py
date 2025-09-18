from typing import Literal

from pydantic import BaseModel


class EpochResult(BaseModel):
    reward: float
    steps_number: int
    final_state: Literal["terminated", "truncated"]
