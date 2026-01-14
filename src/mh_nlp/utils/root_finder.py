import sys
from pathlib import Path


def get_repository_root(add_to_sys_path: bool = False) -> Path:
    """
    Locate the root directory of the repository by searching for known project markers
    such as 'pyproject.toml' or '.git'. Optionally add it to sys.path.

    Parameters
    ----------
    add_to_sys_path : bool, optional
        If True, the resolved repository root will be added to sys.path. Default is False.

    Returns
    -------
    Path
        The resolved repository root path.

    Raises
    ------
    FileNotFoundError
        If no project root markers are found in parent directories.
    """
    current = Path(__file__).resolve().parent

    # Files or folders that indicate the root of a project (Git, Poetry...)
    root_markers = {"pyproject.toml", ".git"}

    for parent in [current, *current.parents]:
        if any((parent / marker).exists() for marker in root_markers):
            if add_to_sys_path:
                sys.path.append(str(parent))
            return parent

    raise FileNotFoundError(
        "Repository root not found. No 'pyproject.toml' or '.git' directory detected."
    )
