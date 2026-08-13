from pathlib import Path
from tempfile import TemporaryDirectory

from declan.active_sensing_movie_information.run_backimage_temporal_remapping_pilot import (
    figure4_candidate_source_rows,
)


def test_figure4_candidate_source_rows_prioritizes_observation_rows_then_candidate_ids():
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "candidate_sets.csv"
        path.write_text(
            "\n".join(
                [
                    "observation_source_row,candidate_ids",
                    "10,source_row:10;source_row:20",
                    "10,source_row:10;source_row:30",
                    "40,",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        assert figure4_candidate_source_rows(path) == [10, 40, 20, 30]
