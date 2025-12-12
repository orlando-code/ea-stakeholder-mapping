from pathlib import Path  # noqa


def get_repo_dir():
    """Get repository directory as an anchor for other file paths"""
    return Path(__file__).parent.parent


repo_dir = get_repo_dir()
data_dir = repo_dir / "data"
