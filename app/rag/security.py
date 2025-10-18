import os
import re
from typing import Dict

ALLOW_PII = os.getenv("ALLOW_PII", "false").lower() == "true"

PII_COLUMN_RE = re.compile(r"(email|e-?mail|phone|ssn|social|account|card|cvv)", re.I)


def mask_row(row: Dict) -> Dict:
	if ALLOW_PII:
		return row
	masked = {}
	for k, v in row.items():
		if PII_COLUMN_RE.search(str(k)):
			masked[k] = "***MASKED***"
		else:
			masked[k] = v
	return masked
