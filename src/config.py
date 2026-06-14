from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

# ---- Paths ----
ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = ROOT_DIR / "data" / "DataSet.csv"
OUTPUTS_DIR = ROOT_DIR / "outputs"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = OUTPUTS_DIR / "reports"

SELECTED_FEATURES_PATH = OUTPUTS_DIR / "selected_features.json"
RISK_SCORES_PATH = OUTPUTS_DIR / "risk_scores.csv"

# ---- Target column ----
TARGET_COL = "F3924"
ID_COL = "Unnamed: 0"

# ---- Reproducibility ----
RANDOM_STATE = 42

# ---- Feature triage settings ----
MAX_MISSING_RATIO = 0.90
TOP_N_FEATURES = 80

# ---- Train/test split ----
TEST_SIZE = 0.2
N_FOLDS = 5

# ---- Leaked columns ----
LEAKED_COLUMNS = ["F3912", "F2230"]

# ---- Groq LLM ----
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "llama-3.3-70b-versatile"

# In config.py add this line:
DECISION_THRESHOLD = 0.35