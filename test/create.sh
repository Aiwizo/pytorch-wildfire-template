#!/usr/bin/env sh

set -o errexit
set -o nounset

. './test/env.sh'

mkdir -p "$TEST_DIR" && cd "$TEST_DIR"

poetry run cookiecutter ../. \
    --no-input --overwrite-if-exists \
    repository_name="$REPOSITORY_NAME"

cd "$REPOSITORY_NAME"
poetry install
