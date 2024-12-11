@echo off
dvc add data/processed/aviation_corpus.json
git add data/processed/aviation_corpus.json.dvc
git commit -m "Update aviation data"
git push origin main
dvc push