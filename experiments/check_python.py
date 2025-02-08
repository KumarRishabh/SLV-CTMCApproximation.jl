import sys
import site
import os

def check_environment():
    print(f"Python Version: {sys.version}")
    print(f"\nPython Executable: {sys.executable}")
    print(f"\nPython Path:\n{os.linesep.join(sys.path)}")
    print(f"\nSite Packages:\n{os.linesep.join(site.getsitepackages())}")

if __name__ == "__main__":
    check_environment()

# VS Code settings for Python
