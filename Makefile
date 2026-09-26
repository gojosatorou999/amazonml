PY    := PYTHONPATH=src .venv/Scripts/python
DATA  ?= data
TEAM  ?= team
.PHONY: app dev test setup synth synth-hard eda train predict loco validate scaling all package serve clean

setup:          ## create venv and install pinned, CPU-only deps
	python -m venv .venv
	.venv/Scripts/python -m pip install -U pip
	.venv/Scripts/python -m pip install -r requirements.txt

synth:          ## synthetic stand-in dataset (dev harness only; never used for the submission)
	$(PY) -m ber.synth --out data --seed 0

synth-hard:     ## harder synthetic set: chain branches, co-located businesses, heavier noise
	$(PY) -m ber.synth --out data_hard --seed 0 --hard

eda:            ## profile the data + verify the one-owner-per-record assumption
	$(PY) -m ber.cli eda --data $(DATA)

train:          ## cross-fitted training -> artifacts/ + reports/cv_report.md
	$(PY) -m ber.cli train --data $(DATA)

predict:        ## blocking -> matching -> output/{candidate_pairs,matching_results}.tsv
	$(PY) -m ber.cli predict --data $(DATA)

loco:           ## leave-one-country-out validation -> reports/loco_report.md
	$(PY) -m ber.cli loco --data $(DATA)

validate:       ## our checker, then Amazon's if present (theirs is authoritative)
	$(PY) -m ber.cli validate --data $(DATA)
	@if [ -f utils/validate_submission.py ]; then \
	  python utils/validate_submission.py --matching output/matching_results.tsv \
	    --candidate output/candidate_pairs.tsv --test-dir $(DATA)/test; fi

scaling:        ## blocking scaling benchmark -> reports/scaling.md
	$(PY) -m ber.eval.scaling --data $(DATA)

all: eda train predict validate

package:        ## build <TEAM>_submission.zip in the structure the organisers specified
	$(PY) -m ber.export.package --team $(TEAM)

app:            ## build the React review console into app/dist (needs Node 18+)
	cd app && npm ci && npm run build

serve:          ## API + review console at http://127.0.0.1:8000 (run `make predict` and `make app` first)
	$(PY) -m ber.service.app

dev:            ## hot-reloading console on :5173, proxying /api to `make serve` on :8000
	cd app && npm run dev

clean:
	rm -rf artifacts/* output/*

test:           ## behavioural tests for every claim the docs make (stdlib unittest)
	$(PY) -m unittest discover -s tests -v
