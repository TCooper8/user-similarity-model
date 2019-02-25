#!/bin/bash
set -Eeuxo pipefail

docker-compose up -d --build ml
docker-compose logs -f ml