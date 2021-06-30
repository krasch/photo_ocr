from typing import List, NamedTuple, Tuple

# polygon in PIL format: list of [(x1, y1), (x2, y2), ...]
Polygon = List[Tuple[float, float]]


OCRResult = NamedTuple("OCRResult", [("polygon", Polygon),
                                     ("text", str),
                                     ("confidence", float)])


RecognitionResult = NamedTuple("RecognitionResult", [("text", str), ("confidence", float)])