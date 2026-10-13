"""Excel workbooks for reviewers. Presentation only: callers wrap this in try/except and never block on it."""
import io
from typing import Dict

import pandas as pd


def write_xlsx(storage, path: str, sheets: Dict[str, pd.DataFrame]) -> None:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        for name, df in sheets.items():
            df.to_excel(writer, sheet_name=name[:31], index=False)
    storage.write_bytes(path, buf.getvalue())
