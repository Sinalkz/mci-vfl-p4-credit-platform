#!/usr/bin/env bash
# The S1 network without Docker (M4's E3 run on the MacBook Air): the same Besu version, genesis
# (chain/data/genesis.json) and node keys (chain/data/<member>/key), and the same node flags as
# chain/docker-compose.yml. Only networking and paths differ: the four nodes share 127.0.0.1, each
# with its own p2p port (30303-30306) and its JSON-RPC port on the host as docker-compose maps it
# (power 8545, mci 8546, digikala 8547, maker 8548); each node's static-nodes.json lists the other
# three at 127.0.0.1. Everything the nodes write (data, logs, JVM temp files, pids) stays under $WORK.
# Usage:
#   BESU_HOME=<besu-26.8.1 dir> JAVA_HOME=<Java 21+> WORK=<dir> bash chain/scripts/native-besu.sh up
#   WORK=<dir> bash chain/scripts/native-besu.sh down
# `up` wipes $WORK/data first: a fresh chain, as `docker compose down` then `up`.
set -euo pipefail
CHAIN="$(cd "$(dirname "$0")/.." && pwd)"
: "${WORK:?set WORK}"
members=(power mci digikala maker)

down() {
  for m in "${members[@]}"; do
    f="$WORK/pids/$m.pid"
    [ -f "$f" ] || continue
    kill "$(cat "$f")" 2>/dev/null || true
    rm -f "$f"
  done
  sleep 3
}

up() {
  : "${BESU_HOME:?set BESU_HOME}" "${JAVA_HOME:?set JAVA_HOME}"
  export JAVA_HOME
  down
  rm -rf "$WORK/data"
  mkdir -p "$WORK/data" "$WORK/logs" "$WORK/pids" "$WORK/tmp"
  enode() { echo "enode://$(sed 's/^0x//' "$CHAIN/data/${members[$1]}/key.pub" | tr -d '\r\n')@127.0.0.1:$((30303 + $1))"; }
  for i in 0 1 2 3; do
    m=${members[$i]}
    mkdir -p "$WORK/data/$m"
    peers=()
    for j in 0 1 2 3; do [ "$i" != "$j" ] && peers+=("\"$(enode "$j")\""); done
    (IFS=,; echo "[${peers[*]}]") > "$WORK/data/$m/static-nodes.json"
    # BESU_OPTS: -Xmx1g as docker-compose; JVM temp files in $WORK/tmp; no hsperfdata in the system temp dir
    BESU_OPTS="-Xmx1g -Djava.io.tmpdir=$WORK/tmp -XX:-UsePerfData" nohup "$BESU_HOME/bin/besu" \
      --genesis-file="$CHAIN/data/genesis.json" \
      --node-private-key-file="$CHAIN/data/$m/key" \
      --data-path="$WORK/data/$m/db" \
      --p2p-host=127.0.0.1 \
      --p2p-port=$((30303 + i)) \
      --rpc-http-enabled \
      --rpc-http-host=127.0.0.1 \
      --rpc-http-port=$((8545 + i)) \
      --rpc-http-api=ETH,NET,WEB3,QBFT,ADMIN,TXPOOL \
      --rpc-http-cors-origins='*' \
      --host-allowlist='*' \
      --min-gas-price=0 \
      --static-nodes-file="$WORK/data/$m/static-nodes.json" \
      --discovery-enabled=false \
      --logging=INFO > "$WORK/logs/$m.log" 2>&1 &
    echo $! > "$WORK/pids/$m.pid"
  done
  # wait until every node has its three peers and blocks are being produced
  for _ in $(seq 1 90); do
    ok=0
    for i in 0 1 2 3; do
      r=$(curl -s -m 2 -H 'content-type: application/json' -d '{"jsonrpc":"2.0","id":1,"method":"net_peerCount","params":[]}' "http://127.0.0.1:$((8545 + i))" || true)
      case "$r" in *'"result":"0x3"'*) ok=$((ok + 1)) ;; esac
    done
    b=$(curl -s -m 2 -H 'content-type: application/json' -d '{"jsonrpc":"2.0","id":1,"method":"eth_blockNumber","params":[]}' http://127.0.0.1:8545 || true)
    if [ "$ok" = 4 ] && ! echo "$b" | grep -q '"result":"0x0"' && echo "$b" | grep -q '"result"'; then echo "up: 4 nodes, 3 peers each, $b"; return 0; fi
    sleep 2
  done
  echo "the network did not come up; see $WORK/logs" >&2
  return 1
}

case "${1:-}" in
  up) up ;;
  down) down ;;
  *) echo "usage: native-besu.sh up|down" >&2; exit 2 ;;
esac
