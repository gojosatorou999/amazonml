PY := PYTHONPATH=src python
.PHONY: setup synth eda block score decide submit validate all clean

setup:          ## create venv and install pinned, CPU-only deps
	python -m venv .venv
	.venv/Scripts/python -m pip install -U pip
	.venv/Scripts/python -m pip install -r requirements.txt

synth:          ## generate the synthetic stand-in dataset (used until real data lands)
	$(PY) -m ber.synth --out data --seed 0

eda:            ## profile the dataset + verify the one-to-many assignment assumption
	$(PY) -m ber.cli eda

block:          ## candidate generation -> output/candidate_pairs.tsv + blocking report
	$(PY) -m ber.cli block

score:          ## feature build + model inference -> calibrated p(match)
	$(PY) -m ber.cli score

decide:         ## expected-F0.5 set selection -> output/matching_results.tsv
	$(PY) -m ber.cli decide

submit: block score decide validate

validate:       ## Amazon-provided format checker; a precondition on every submission
	python utils/validate_submission.py \
	  --matching output/matching_results.tsv \
	  --candidate output/candidate_pairs.tsv \
	  --test-dir data/test

all: synth eda submit

clean:
	rm -rf artifacts/* output/*
