from pydantic import BaseModel, Field
from typing import Optional, Dict, Any

class ProcessQuery(BaseModel):
    user_id: str = Field(default="demo-user")
    sex: int     # 0=female,1=male
    goal_3200_s: int
    goal_push: int
    goal_sit: int

class ProcessResult(BaseModel):
    height_m: float
    weight_kg: float
    class_idx: int
    class_name: str
    plan: Dict[str, Any]
    uploadId: str
    predId: str
    planId: str
