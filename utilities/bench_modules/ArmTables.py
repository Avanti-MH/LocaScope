"""The stage 2 cache entries the stage 2 retriever bench writes, a slide at a time.

An entry is what the pipeline's own stage 2 writes for a phase -- `output` and the tables the phase holds
(`BenchCommon.phase_roles`: `tile_sims`, `truth`, `truth_sim`, `rerank_scores`) under `.../stage1=oracle/stage2/`,
named and recorded as the pipeline names and records it (`BenchCommon.phase_ids`, `phase_record`) and filled by
the same writer (`BenchCommon.write_phase`) -- so the Demo, the plots, stage 3 and a pipeline run read an arm of
this bench exactly as a pipeline run's. Two things fill them:

    the streamed sliding-window arms     `bench_stage2_retriever.stage2_tables`
    every other phase                     `RetrieverArms.ArmEngine`

AN ENTRY HOLDS EVERY LEVEL OF ITS SLIDE. It is written once, when the slide's last level is done
(`SlideTables.flush`): written level by level, the first level would make it a hit and the others would
never be written. So a slide whose entries are missing or do not reach every level holding a FoV the arms are scored on
(`entries_complete`) is run again from its first level; that is how an interrupted slide, or an entry
written by a run that only kept level 0, gets repaired.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from BenchCommon import Tables, status


# ── what a slide has ──────────────────────────────────────────────────────────

def entry_levels(entry: Any, entry_id: str) -> Optional[Set[int]]:
    """The levels the `output` table of an entry has a row for; None when the entry has no such table. A FoV's
    list is written only when it was searched at its own level, so these are the FoVs' levels."""
    path: Path = entry.path('output', entry_id, '.csv')
    if not path.is_file():
        return None
    with open(path, newline='') as handle:
        return {int(float(r['level'])) for r in csv.DictReader(handle) if r.get('level') not in (None, '')}


def entries_complete(entry: Any, wanted: Dict[str, Dict[str, Any]], levels: Set[int],
                     roles: Dict[str, Sequence[str]]) -> bool:
    """True when every wanted entry is there, with all the tables it holds (`roles`, by entry), and its list
    reaches every level in `levels` (the levels of the slide holding a FoV the arms are scored on)."""
    for entry_id, rec in wanted.items():
        if status(entry, entry_id, rec, roles[entry_id]) != 'hit':
            return False
        have: Optional[Set[int]] = entry_levels(entry, entry_id)
        if have is None or not levels <= have:
            return False
    return True


# ── a slide's tables ──────────────────────────────────────────────────────────

class SlideTables:
    """The tables of one slide's entries while its levels run, written together when the slide is done. Each
    entry gets the tables it holds (`roles`, by entry)."""

    def __init__(self, entry: Any, wanted: Dict[str, Dict[str, Any]], roles: Dict[str, Sequence[str]]) -> None:
        self.entry: Any = entry
        self._tables: Dict[str, Tables] = {i: Tables(entry, i, rec, roles[i]) for i, rec in wanted.items()}

    def has(self, entry_id: str) -> bool:
        """Is the entry one of those this sink writes."""
        return entry_id in self._tables

    def tables(self, entry_id: str) -> Tables:
        return self._tables[entry_id]

    def flush(self) -> Tuple[int, List[str]]:
        """Write every entry that got rows; `(written, ids of the entries that got none)`. An arm that failed
        has none, and an empty entry would make it look done."""
        written: int = 0
        empty: List[str] = []
        for entry_id, tables in self._tables.items():
            if not tables.rows['output']:
                empty.append(entry_id)
                continue
            tables.write()
            written += 1
        return written, empty


__all__ = ['SlideTables', 'entries_complete', 'entry_levels']
