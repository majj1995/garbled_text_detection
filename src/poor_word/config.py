from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class PathsConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="POOR_WORD_", frozen=True)

    repo_root: Path = Field(default_factory=Path.cwd)
    raw_dir: Path = Path("data/raw")
    generated_dir: Path = Path("data/generated")
    artifacts_dir: Path = Path("artifacts")
    models_dir: Path = Path("models")

    def resolve(self, path: Path) -> Path:
        return path if path.is_absolute() else self.repo_root / path

    @property
    def raw_path(self) -> Path:
        return self.resolve(self.raw_dir)

    @property
    def generated_path(self) -> Path:
        return self.resolve(self.generated_dir)

    @property
    def artifacts_path(self) -> Path:
        return self.resolve(self.artifacts_dir)

    @property
    def models_path(self) -> Path:
        return self.resolve(self.models_dir)

    def create_runtime_dirs(self) -> tuple[Path, Path, Path]:
        directories = (self.generated_path, self.artifacts_path, self.models_path)
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        return directories
