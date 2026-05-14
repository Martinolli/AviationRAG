import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from aviationrag.config import (  # noqa: E402
    DEFAULT_MANIFEST_PATH,
    MANIFEST_DRY_RUN_ENV,
    MANIFEST_INTEGRATION_ENV,
    MANIFEST_PATH_ENV,
    ManifestIntegrationSettings,
    get_manifest_integration_settings,
    get_manifest_path,
    is_manifest_dry_run_enabled,
    is_manifest_integration_enabled,
    parse_bool_env,
)


class TestManifestConfig(unittest.TestCase):
    def test_defaults_are_disabled(self):
        env = {}

        self.assertFalse(is_manifest_integration_enabled(env))
        self.assertFalse(is_manifest_dry_run_enabled(env))

    def test_default_manifest_path(self):
        self.assertEqual(get_manifest_path({}), Path("data/manifest/documents.jsonl"))
        self.assertEqual(DEFAULT_MANIFEST_PATH, Path("data/manifest/documents.jsonl"))

    def test_true_like_values_enable_integration(self):
        for value in ("1", "true", "TRUE", "yes", "on", " On "):
            with self.subTest(value=value):
                env = {MANIFEST_INTEGRATION_ENV: value}
                self.assertTrue(is_manifest_integration_enabled(env))

    def test_false_like_values_disable_integration(self):
        for value in ("0", "false", "FALSE", "no", "off", " Off "):
            with self.subTest(value=value):
                env = {MANIFEST_INTEGRATION_ENV: value}
                self.assertFalse(is_manifest_integration_enabled(env))

    def test_unknown_values_fall_back_to_default(self):
        self.assertFalse(parse_bool_env("maybe", default=False))
        self.assertTrue(parse_bool_env("maybe", default=True))
        self.assertFalse(is_manifest_integration_enabled({MANIFEST_INTEGRATION_ENV: "maybe"}))

    def test_custom_manifest_path_is_respected(self):
        env = {MANIFEST_PATH_ENV: "tmp/sample/documents.jsonl"}

        self.assertEqual(get_manifest_path(env), Path("tmp/sample/documents.jsonl"))

    def test_settings_returns_expected_fields(self):
        env = {
            MANIFEST_INTEGRATION_ENV: "yes",
            MANIFEST_DRY_RUN_ENV: "true",
            MANIFEST_PATH_ENV: "custom/manifest.jsonl",
        }

        settings = get_manifest_integration_settings(env)

        self.assertIsInstance(settings, ManifestIntegrationSettings)
        self.assertTrue(settings.enabled)
        self.assertTrue(settings.dry_run)
        self.assertEqual(settings.manifest_path, Path("custom/manifest.jsonl"))

    def test_injected_env_does_not_require_real_environment_mutation(self):
        injected = {
            MANIFEST_INTEGRATION_ENV: "true",
            MANIFEST_DRY_RUN_ENV: "true",
            MANIFEST_PATH_ENV: "injected/documents.jsonl",
        }

        with patch.dict(
            os.environ,
            {
                MANIFEST_INTEGRATION_ENV: "false",
                MANIFEST_DRY_RUN_ENV: "false",
                MANIFEST_PATH_ENV: "real-env/documents.jsonl",
            },
            clear=False,
        ):
            settings = get_manifest_integration_settings(injected)

        self.assertTrue(settings.enabled)
        self.assertTrue(settings.dry_run)
        self.assertEqual(settings.manifest_path, Path("injected/documents.jsonl"))


if __name__ == "__main__":
    unittest.main()
