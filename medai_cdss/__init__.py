"""MedAI-nEXUS CDSS: explainable, FHIR-interoperable clinical risk prediction."""

import os
import warnings

# joblib probes physical cores with a Windows tool (wmic) that is often missing and prints a
# traceback. Giving it a core count below the logical count (~physical cores) skips the probe.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(max(1, (os.cpu_count() or 2) // 2)))
warnings.filterwarnings("ignore", message="Could not find the number of physical cores")

__version__ = "2.0.0"
