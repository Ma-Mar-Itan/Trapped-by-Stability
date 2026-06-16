# Notebooks

Exploratory / presentation notebooks live here. They should import the
pipeline package rather than re-implementing analysis logic:

```python
import sys; sys.path.insert(0, "../src")
from radiomics_stability.clustering import pca, kmeans
```

Keep reproducible analysis in `src/radiomics_stability/`; use notebooks only for
interactive exploration and figure inspection. Notebook scratch output is
gitignored.
