import os
import sys

# Modules in src/ import each other by bare name (as main.py runs them).
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
