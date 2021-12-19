from dataclasses import dataclass
from typing import List, Tuple


@dataclass(frozen=True)
class Segment:
    start: float
    end: float

    def len(self):
        return self.end - self.start

    def mid(self) -> float:
        return (self.start + self.end) / 2

    def split(self, n_partitions: int) -> List['Segment']:
        step = self.len() / n_partitions

        result = []
        cur_start = self.start
        for i in range(n_partitions):
            is_last_partition = i == n_partitions - 1
            cur_partition = Segment(
                start=cur_start,
                # if last partition use [self.end] to avoid floating point errors
                end=self.end if is_last_partition else (cur_start + step)
            )
            result.append(cur_partition)
            cur_start += step
        return result

    def evenly_divide_into_nodes(self, num: int) -> List[float]:
        h = (self.end - self.start) / (num - 1)
        return [self.start + h * i for i in range(num)]

    def to_tuple(self) -> Tuple[float, float]:
        return (self.start, self.end)

    def __str__(self):
        return f"[{self.start}; {self.end}]"