import duckdb
import re
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional


class DataLoader:
    @staticmethod
    def load_duckdb_data(
            duckdb_path: str,
            table: Optional[str] = None,
    ) -> List[str]:
        if not Path(duckdb_path).exists():
            raise FileExistsError(f"DuckDB file not found: {duckdb_path}")

        conn: duckdb.DuckDBPyConnection = duckdb.connect(duckdb_path)

        tbl = table if table else conn.execute("SHOW TABLES").fetchone()[0]

        df: pd.DataFrame = conn.execute(f"SELECT * FROM {tbl}").df()

        texts: List[str] = []

        for _, row in df.iterrows():
            txt = " ".join(str(v) for v in row if isinstance(v, str))
            # Clean boilerplate text
            txt = re.sub(r'https?://\S+', '', txt)
            texts.append(txt)

        conn.close()

        return texts


if __name__ == "__main__":
    # Example usage
    from autocomplete_system.config import DUCKDB_PATH
    loader = DataLoader()
    texts = loader.load_duckdb_data(duckdb_path=DUCKDB_PATH)
    print(texts[:5])  # Print first 5 loaded texts
