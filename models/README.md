# Model Placeholders

This directory is reserved for language-specific SQLite vector stores derived
from fastText vectors.

Expected local filenames:

- `cc.pt.300.sqlite`
- `cc.en.300.sqlite`
- `cc.es.300.sqlite`

The production files are intentionally not stored in git because each file is
well above 50 MB. Use `analysis/build_vector_store.py` to create compatible
local stores from `.vec` or `.vec.gz` fastText files.
