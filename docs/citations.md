# Citation

If you use SmallMLP in academic work, please cite the software and (when available) the preprint.

---

## Software

```bibtex
@software{emelyanov2026smallmlp,
  author       = {Emelyanov, Ilya},
  title        = {{SmallMLP}: Learned-bandwidth Nadaraya-Watson regression
                  with prediction intervals for small nonlinear data},
  year         = {2026},
  publisher    = {GitHub},
  version      = {0.1.0},
  url          = {https://github.com/nsdmlk/smallmlp},
  note         = {Non-parametric regression with weighted conformal prediction}
}
```

---

## Preprint

A preprint describing the method and benchmarks is in preparation.

**Status:** draft.

**Expected:** arXiv `cs.LG`, 2026.

When the preprint is posted, its BibTeX will appear here. Check the [GitHub repository](https://github.com/nsdmlk/smallmlp) for updates.

---

## Earlier work

SmallMLP is a follow-up to the author's earlier work on robust gradient boosting for small data. If you use both, please cite both.

```bibtex
@software{emelyanov2025smallgbm,
  author       = {Emelyanov, Ilya},
  title        = {{SmallGBM}: Gradient boosting with robust leaf regularization
                  for small-sample tabular data},
  year         = {2025},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.21934674},
  url          = {https://doi.org/10.5281/zenodo.21934674}
}
```

**SmallGBM** — gradient boosting for small tabular data ($n < 1000$). Outperforms XGBoost, LightGBM, and RandomForest on 27 benchmark datasets by using robust leaf weights (median + adaptive shrinkage) and stochastic split selection.

Repository: [github.com/nsdmlk/smallgbm](https://github.com/nsdmlk/SmallGBM)

---

## How to cite in text

**For the software:**

> We used SmallMLP (Emelyanov, 2026) for non-parametric regression with weighted conformal prediction intervals.

**For the method:**

> SmallMLP learns a per-feature kernel bandwidth via leave-one-out optimization and produces calibrated prediction intervals through weighted conformal prediction (Emelyanov, 2026).

---

## BibTeX management

If you use a reference manager (Zotero, Mendeley, BibDesk), import the BibTeX block directly.

For LaTeX documents:

```latex
\bibliographystyle{plain}
\bibliography{refs}
```

where `refs.bib` contains the entry above.

For `natbib`:

```latex
\usepackage{natbib}
...
\citep{emelyanov2026smallmlp}
```

---

## Contact

For questions about citation or to report a paper using SmallMLP, contact:

- **GitHub issues:** [github.com/nsdmlk/smallmlp/issues](https://github.com/nsdmlk/smallmlp/issues)
- **Email:** Nsdmlk@yandex.ru
- **Telegram: KantervilleGhost**

> We are happy to hear about applications of SmallMLP in scientific research.
