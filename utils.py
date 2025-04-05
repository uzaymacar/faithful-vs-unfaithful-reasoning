from typing import Any, Dict, Literal, Optional, TypedDict

class Question(TypedDict):
    q_str: str
    q_str_open_ended: str
    x_name: str
    y_name: str
    x_value: Any
    y_value: Any

class DatasetParams(TypedDict):
    prop_id: str
    comparison: Literal["gt", "lt"]
    answer: Literal["YES", "NO"]
    max_comparisons: int
    uuid: str
    suffix: Optional[str]

class QsDataset(TypedDict):
    question_by_qid: Dict[str, Question]
    params: DatasetParams | Dict[str, Any]

class CoTData(TypedDict):
    qid: str
    prop_id: str
    comparison: str
    ground_truth_answer: Literal["YES", "NO"]
    question_text: str
    generated_cot: str
    eval_final_answer: Literal["YES", "NO", "UNKNOWN", "REFUSED", "FAILED_EVAL"] | None
    eval_equal_values: bool | None
    is_faithful_iphr: bool | None