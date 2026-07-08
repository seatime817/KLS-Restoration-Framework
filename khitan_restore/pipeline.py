"""Top-level pipeline orchestration."""

from __future__ import annotations

from pathlib import Path

from .config import PipelineConfig
from .io_utils import ensure_dir, resolve_path, write_json
from .refinement import run_refinement
from .restoration import run_component_restoration
from .segmentation import run_segmentation
from .super_resolution import run_super_resolution


class KhitanRestorationPipeline:
    """Run the full blurry-text restoration pipeline with one config."""

    def __init__(self, config: PipelineConfig):
        self.config = config

    def _search_roots(self) -> list[str]:
        roots = list(self.config.search_roots)
        dataset = getattr(self.config, "dataset", None)
        if dataset is not None:
            for candidate in [
                getattr(dataset, "root_dir", None),
                getattr(dataset, "data_dir", None),
                getattr(dataset, "components_dir", None),
                getattr(dataset, "checkpoints_dir", None),
            ]:
                if candidate:
                    roots.append(candidate)
        if self.config.config_base_dir:
            roots.insert(0, self.config.config_base_dir)
        deduped: list[str] = []
        seen: set[str] = set()
        for root in roots:
            if root in seen:
                continue
            seen.add(root)
            deduped.append(root)
        return deduped

    def run(
        self,
        *,
        input_path: str | None = None,
        output_root: str | None = None,
    ) -> dict:
        search_roots = self._search_roots()
        base_dir = self.config.config_base_dir
        raw_input_path = input_path or self.config.input_path
        original_input = resolve_path(raw_input_path, search_roots=search_roots, base_dir=base_dir)

        raw_output_root = output_root or self.config.output.root_dir
        resolved_output_root = resolve_path(
            raw_output_root,
            search_roots=search_roots,
            base_dir=base_dir,
            allow_missing=True,
        )
        ensure_dir(resolved_output_root)
        write_json(self.config.to_dict(), resolved_output_root / "resolved_config.json")

        current_input: str | Path = original_input
        summary: dict[str, object] = {
            "input_path": str(original_input),
            "output_root": str(resolved_output_root),
            "steps": {},
        }
        print(f"[pipeline] input={original_input}")
        print(f"[pipeline] output_root={resolved_output_root}")

        if self.config.super_resolution.enabled:
            print("[pipeline] stage=super_resolution start")
            super_resolution_dir = ensure_dir(
                resolved_output_root / self.config.output.super_resolution_dir
            )
            super_resolution_manifest = run_super_resolution(
                current_input,
                super_resolution_dir,
                self.config.super_resolution,
                search_roots=search_roots,
                base_dir=base_dir,
            )
            summary["steps"]["super_resolution"] = super_resolution_manifest
            current_input = Path(super_resolution_manifest["output_image_dir"])
            print("[pipeline] stage=super_resolution done")

        if self.config.segmentation.enabled:
            print("[pipeline] stage=segmentation start")
            segmentation_dir = ensure_dir(resolved_output_root / self.config.output.segmentation_dir)
            segmentation_manifest = run_segmentation(
                current_input,
                segmentation_dir,
                self.config.segmentation,
                search_roots=search_roots,
                base_dir=base_dir,
            )
            summary["steps"]["segmentation"] = segmentation_manifest
            current_input = segmentation_manifest["segment_json"]
            print("[pipeline] stage=segmentation done")
        elif self.config.restoration.enabled:
            current_input = resolve_path(
                current_input,
                search_roots=search_roots,
                base_dir=base_dir,
            )
            if Path(current_input).suffix.lower() != ".json":
                raise ValueError(
                    "Segmentation is disabled, so the restoration input must be a segment_results.json file."
                )

        if self.config.restoration.enabled:
            print("[pipeline] stage=restoration start")
            restoration_dir = ensure_dir(resolved_output_root / self.config.output.restoration_dir)
            restoration_manifest = run_component_restoration(
                current_input,
                restoration_dir,
                self.config.restoration,
                search_roots=search_roots,
                base_dir=base_dir,
            )
            summary["steps"]["restoration"] = restoration_manifest
            current_input = restoration_manifest["results_path"]
            print("[pipeline] stage=restoration done")

        if self.config.refinement.enabled:
            if not self.config.restoration.enabled:
                raise ValueError("Refinement requires the restoration stage to be enabled.")
            print("[pipeline] stage=refinement start")
            refinement_dir = ensure_dir(resolved_output_root / self.config.output.refinement_dir)
            refinement_manifest = run_refinement(
                current_input,
                refinement_dir,
                self.config.refinement,
                search_roots=search_roots,
                base_dir=base_dir,
            )
            summary["steps"]["refinement"] = refinement_manifest
            print("[pipeline] stage=refinement done")

        manifest_path = write_json(
            summary,
            resolved_output_root / self.config.output.manifest_name,
        )
        summary["manifest_path"] = str(manifest_path)
        return summary
