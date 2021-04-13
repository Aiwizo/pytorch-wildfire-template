#!/usr/bin/env sh

set -o errexit
set -o nounset

. './test/env.sh'

cd "$PROJECT_PATH"

poetry run guild run train max_epochs=2 n_batches_per_epoch=2 batch_size=4 eval_batch_size=4 cuda=false -y
poetry run guild run retrain max_epochs=1 n_batches_per_epoch=2 batch_size=4 eval_batch_size=4 cuda=false -y
poetry run guild run evaluate cuda=false eval_batch_size=4 -y
