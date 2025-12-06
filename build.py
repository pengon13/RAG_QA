"""
Build script for packaging the PySide6 GUI into a single executable.
Requires pyinstaller to be installed in the active environment.
"""

import subprocess
import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).parent
    dist = root / "dist"
    dist.mkdir(exist_ok=True)

    cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        "app_gui.py",
        "--name",
        "RAG_GUI",
        "--console",
        # Collect only required PySide6 modules to keep bundle size/build time down
        "--collect-submodules",
        "PySide6.QtCore",
        "--collect-submodules",
        "PySide6.QtGui",
        "--collect-submodules",
        "PySide6.QtWidgets",
        "--collect-submodules",
        "qdrant_client",
        "--hidden-import",
        "jaraco.text",
        "--hidden-import",
        "platformdirs",
        "--add-data",
        ".env;.env",
        "--icon",
        str(root / "icon.png"),
    ]

    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd, cwd=root)
    print("\nBuild complete. Executable located in dist/RAG_GUI.exe")


if __name__ == "__main__":
    main()
