@echo off
dvc add data/raw/aviation_corpus.pkl
git add data/raw/aviation_corpus.pkl.dvc
git commit -m "Update aviation corpus data"
git push origin main
dvc push