# Vanilla Steel Assignment

## Structure of the repository

```
├─ task_1/ 
│  ├─  supplier_data1.xlsx
│  └─ supplier_data2.xlsx
├─ task_2/ 
│  ├─ reference_properties.tsv
│  └─ rfq.csv
├─ plots/
│   └─ decision_comparison_tree.png
├─ deliverables/
│   └─ inventory_dataset.csv
│   └─ top3.csv
├─ TaskA.ipynb
├─ TaskB.ipynb
├─ README.md
└─ requirements.txt
```

## Creation of venv

```bash
python3 -m venv dataenv
source dataenv/bin/activate
pip install -r requirements.txt
```

## Task A.1 — Clean & Join
Goal:
- Clean and normalize both datasets (e.g., unify thickness/width formats, standardize names).
- Handle missing/inconsistent values.
- Join into a single table called inventory_dataset.
- Document your assumptions.

Deliverable: inventory_dataset.csv

**TaskA.ipynb**
- **Datasets loaded:** `task_1/supplier_data1.xlsx` and `task_1/supplier_data2.xlsx` (50 rows each). Initial `info()` showed no NaNs, but several numeric fields contained zeros that behave like missing values.
- **Column normalization:** Renamed columns to snake_case and aligned semantics across files.
  - Supplier 1 mapping: `Quality/Choice→quality_choice`, `Grade→grade`, `Finish→finish`, `Thickness (mm)→thickness_mm`, `Width (mm)→width_mm`, `Description→defect_desc`, `Gross weight (kg)→sup1_weight_kg`, `RP02→rp02`, `RM→rm`, `Quantity→sup1_quantity`, `AG→ag`, `AI→ai`.
  - Supplier 2 mapping: `Material→material`, `Description→coating_desc`, `Article ID→article_id`, `Weight (kg)→sup2_weight_kg`, `Quantity→sup2_quantity`, `Reserved→reserved`.
  - Additional cleanup for Supplier 2: extracted final token of `coating_desc` and lowercased (oiled/painted/not), normalized `reserved` to lowercase with underscores.
- **Missing-value analysis (zeros as proxies):**
  - Found substantial zeros in `sup1_quantity` and `ag`, and notable zeros in `rp02` and `rm`.
  - Identified 15 rows where `sup1_quantity == 0` but `sup1_weight_kg > 0` (a quantity/weight inconsistency). Computed a `quantity_weight_conflict` flag for these.
  - Explored associations: numeric–numeric correlations were weak; categorical–numeric (eta-squared) and categorical–categorical (Cramér’s V) also weak, suggesting limited structure given small sample size.
  - Computed `unit_weight = sup1_weight_kg / sup1_quantity` for rows with `sup1_quantity > 0`. Distribution is wide; ANOVA by `grade` and `finish` showed no significant differences (p > 0.2), reinforcing weak group effects.
- **Imputation approach (applied to Supplier 1 during exploration):**
  - Imputed `sup1_quantity` for `quantity_weight_conflict` rows using median `unit_weight` by `(grade, finish)`, falling back to grade-only median, then global median.
  - Converted exact zeros in mechanical properties (`rp02`, `rm`, `ag`) to NaN prior to any imputation logic; considered defect-aware handling for “Sollmasse (Gewicht) unterschritten”.
- **Join decision:** No reliable natural key exists across the two files. Overlapping fields ( weights, quantities) are not unique identifiers, and `Description` has different semantics in each dataset. Therefore, no merge was performed. Instead, both tables were column-aligned and concatenated to produce a unified inventory table, saved as `deliverables/inventory_dataset.csv`.

## Task B
