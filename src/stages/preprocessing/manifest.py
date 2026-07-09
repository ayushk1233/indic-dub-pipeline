import json
from pathlib import Path


class ManifestWriter:
    def write(
        self,
        chunk_paths: list[Path],
        segments: list[tuple[float, float]],
        output_path: Path,
    ) -> Path:
        """
        Write a JSON manifest describing generated speech chunks.
        """

        manifest = []

        for chunk_path, (start, end) in zip(chunk_paths, segments):
            manifest.append(
                {
                    "chunk_path": str(chunk_path),
                    "start_ts": start,
                    "end_ts": end,
                }
            )

        with output_path.open("w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        return output_path
