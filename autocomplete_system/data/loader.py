import duckdb
import re
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional


class DataLoader:
    """
    DataLoader class to load data from a crawled DuckDB database.

    This class provides methods to load text data from a DuckDB database file.
    """
    @staticmethod
    def load_duckdb_data(
            duckdb_path: str,
            table: Optional[str] = None,
    ) -> List[str]:
        """
        Load text data from a DuckDB database file.

        Args:
            duckdb_path (str): Path to the DuckDB database file.
            table (Optional[str]): Name of the table to load data from. If None, the first table is used.
        """
        # Make sure the DuckDB file exists
        if not Path(duckdb_path).exists():
            raise FileExistsError(f"DuckDB file not found: {duckdb_path}")

        # Connect to the DuckDB database
        conn: duckdb.DuckDBPyConnection = duckdb.connect(duckdb_path)

        # Get the table name if not provided
        tbl = table if table else conn.execute("SHOW TABLES").fetchone()[0]

        # Fetch all data from the specified table
        df: pd.DataFrame = conn.execute(f"SELECT * FROM {tbl}").df()

        # Iterate over the DataFrame rows and concatenate text columns
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
    import random

    from autocomplete_system.config import DUCKDB_PATH

    loader = DataLoader()
    texts = loader.load_duckdb_data(duckdb_path=DUCKDB_PATH)
    # Print 5 random texts
    for text in random.sample(texts, 5):
        print(text)
        print()
