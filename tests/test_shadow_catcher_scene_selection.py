import sys
import types
import unittest
from pathlib import Path

sys.modules.setdefault("torch", types.ModuleType("torch"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import test_shadow_catcher_p1p3 as shadow


class ShadowCatcherSceneSelectionTest(unittest.TestCase):
    def test_default_scene_selection_runs_every_scene(self):
        self.assertEqual(shadow.resolve_scene_selection("all"), list(shadow.ALL_SCENE_TAGS))

    def test_scene_selection_accepts_aliases_and_full_names(self):
        self.assertEqual(
            shadow.resolve_scene_selection("D,E,J_area_light"),
            [
                "D_hard_shadow",
                "E_soft_shadow",
                "J_area_light",
            ],
        )

    def test_scene_selection_accepts_json_list_values(self):
        self.assertEqual(
            shadow.resolve_scene_selection(["gs_only", "H", "I_point_plus_dir"]),
            [
                "gs_only",
                "H_point_light",
                "I_point_plus_dir",
            ],
        )

    def test_scene_selection_expands_shadow_diff_dependencies(self):
        self.assertEqual(
            shadow.resolve_scene_selection("shadow_diff"),
            [
                "B_catcher_nolight",
                "D_hard_shadow",
            ],
        )

    def test_scene_selection_rejects_unknown_names(self):
        with self.assertRaisesRegex(ValueError, "Unknown scene"):
            shadow.resolve_scene_selection("D,unknown")

    def test_assertion_dependencies_require_all_check_scenes_without_skip_assert(self):
        selected = shadow.resolve_scene_selection("D,E,J")

        with self.assertRaisesRegex(ValueError, "--skip_assert"):
            shadow.validate_assertion_scene_dependencies(selected, skip_assert=False)

    def test_assertion_dependencies_allow_partial_runs_with_skip_assert(self):
        selected = shadow.resolve_scene_selection("D,E,J")

        shadow.validate_assertion_scene_dependencies(selected, skip_assert=True)


if __name__ == "__main__":
    unittest.main()
