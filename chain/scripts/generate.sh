#!/usr/bin/env bash
# Generate validator keys and the QBFT genesis with Besu's own operator tool,
# then lay them out for docker-compose. Wipes chain/data (a fresh chain).
# Topology: full mesh. Each validator gets a static-nodes.json listing the other
# three; discovery is disabled in docker-compose, so no node relays for others.
# Usage: bash chain/scripts/generate.sh        (BESU_VERSION overrides the image tag)
set -euo pipefail
cd "$(dirname "$0")/.."
export MSYS_NO_PATHCONV=1                      # Git Bash on Windows: keep /work as-is
HOST_DIR="$(pwd -W 2>/dev/null || pwd)"        # Windows path under Git Bash, POSIX path elsewhere
IMAGE="hyperledger/besu:${BESU_VERSION:-26.8.1}"

rm -rf data
mkdir -p data
docker run --rm --user root -v "${HOST_DIR}:/work" --entrypoint /opt/besu/bin/besu "${IMAGE}" \
  operator generate-blockchain-config \
  --config-file=/work/qbftConfigFile.json \
  --to=/work/data/networkFiles \
  --private-key-file-name=key

# One validator per member, in address order of the generated keys.
members=(power mci digikala maker)
ips=(172.28.0.11 172.28.0.12 172.28.0.13 172.28.0.14)   # must match docker-compose.yml
i=0
for d in data/networkFiles/keys/*/; do
  m=${members[$i]}
  mkdir -p "data/$m"
  cp "$d/key" "$d/key.pub" "data/$m/"
  basename "$d" > "data/$m/address"
  i=$((i + 1))
done
cp data/networkFiles/genesis.json data/genesis.json

enode() { echo "enode://$(sed 's/^0x//' "data/${members[$1]}/key.pub" | tr -d '\r\n')@${ips[$1]}:30303"; }
for i in 0 1 2 3; do
  peers=()
  for j in 0 1 2 3; do [ "$i" != "$j" ] && peers+=("\"$(enode "$j")\""); done
  (IFS=,; echo "[${peers[*]}]") > "data/${members[$i]}/static-nodes.json"
done

echo "BESU_VERSION=${BESU_VERSION:-26.8.1}" > .env

echo "validators:"
for m in "${members[@]}"; do echo "  $m $(cat "data/$m/address")"; done
