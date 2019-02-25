#!/bin/bash
set -Eeuxo pipefail

docker-compose build ml-tests
docker-compose run --rm ml-tests