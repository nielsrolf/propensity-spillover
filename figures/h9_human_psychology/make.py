"""§4.4 / Appendix H: human-psychology (H9) predictor.

  boxplot.py   h9_prediction_boxplot_rep.png   observed SFT spillover by H9 prediction level (Fig. 7)
  matrix.py    h9_predicted_matrix.png         the H9 predicted spillover matrix (App. H)
  analysis.py  h9_prediction_table_rep.tex     the H9 spec's evidence table restricted to the battery,
               plus the numbers quoted in §4.4 / App. H (Tier-1 rho, cross-model agreement, errors)

    python figures/h9_human_psychology/make.py
"""
import pathlib
import runpy

HERE = pathlib.Path(__file__).resolve().parent
for script in ("boxplot.py", "matrix.py", "analysis.py"):
    runpy.run_path(str(HERE / script), run_name="__main__")
