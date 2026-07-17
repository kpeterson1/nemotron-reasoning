SPLIT ?= dev_frozen
CONFIG ?= configs/train/lora_baseline.yaml
RUN1 ?=
RUN2 ?=
ADAPTER ?=

.PHONY: eval compare smoke-test package train

eval:
	python -m src.evaluation.run_eval --split $(SPLIT) --config configs/eval/default.yaml

compare:
	python -m src.evaluation.compare --run1 $(RUN1) --run2 $(RUN2)

smoke-test:
	pytest tests/

package:
	python -m src.packaging.make_submission --adapter-dir $(ADAPTER) --output submission/submission.zip

train:
	python -m src.training.train_lora --config $(CONFIG)

check-no-reference:
	@grep -rn "reference_solvers\|tonghuikang" src/ scripts/ notebooks/ && echo "FAIL: reference code leaked into production" && exit 1 || echo "OK: no reference leaks"
