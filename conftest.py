from pathlib import Path
import sys

# Add the project root to Python path so retrieval_engine package can be found
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
