#!/bin/bash
# MCI loan instance v2 (MCI as the active member), Step 5 on a macOS machine (the delivered runs: a MacBook Air, Apple
# M4): v1's driver with v2's data, configs and records. Every process reads data/mci_loan/v2/prepared through
# MCI_LOAN_PREPARED; process A (slot power) is MCI: it creates the run and escrows the pool; p1, p2, p3 are the Bank,
# the E-commerce company and the Credit bureau (scenarios/mci_loan/v2/slots.json).
# The machine's locations come from the environment (README.md, "On-chain runs"):
#   BENCH        a folder for tools, scratch files and Besu's WORK folder (required)
#   NODE_BIN     Node.js's bin folder                          (default $BENCH/opt/node/bin)
#   BESU_HOME    Besu 26.8.1                                   (default $BENCH/opt/besu-26.8.1)
#   JAVA_HOME    a Java runtime for Besu                       (default $BENCH/opt/jre25/Contents/Home)
#   RAPIDSNARK   rapidsnark's prover binary                     (default $BENCH/opt/rapidsnark/bin/prover)
#   GENESIS_SHA256  the SHA-256 the genesis check expects      (default: chain/data/genesis.json's own)
# One stage per call, one heavy job at a time, under caffeinate. Run k: model config
# results/mci_loan/v2/configs/loan2-s<k>-wait10s.json, run seed k, scenario seed 20267005 (one draw for all five runs,
# as v1), the alignment period from the benchmark (results/mci_loan/v2/onchain/plan.json).
#   bash scenarios/mci_loan/v2/onchain/mac-runs-v2.sh test
#   caffeinate -ims bash scenarios/mci_loan/v2/onchain/mac-runs-v2.sh bench            > $BENCH/scratch/mci2-bench.log 2>&1
#   caffeinate -ims bash scenarios/mci_loan/v2/onchain/mac-runs-v2.sh run <k> <blocks>  > $BENCH/scratch/mci2-run-<k>.log 2>&1
#   bash scenarios/mci_loan/v2/onchain/mac-runs-v2.sh replay <k>
# run: native Besu (chain/scripts/native-besu.sh: a fresh chain from chain/data/genesis.json), the genesis check, then
#   contracts/offchain/runner3.mjs --run mci-loan; every 5 s the available memory, the swapout counter and the summed
#   resident memory of the Besu nodes and of the node processes go to <run>/mem.log; then the network goes down.
#   The delivered runs' genesis had SHA-256 abec639c5c0d6653683eeed20bb414e97d01d9608f4c973a2f07cebd1518a9a1
#   (results/mci_loan/v2/onchain/run-<k>-network-check.json); a chain generated anew has its own.
set -uo pipefail
B=${BENCH:?set BENCH to a folder for tools, scratch files and the Besu WORK folder (README.md)}
REPO=$(cd "$(dirname "$0")/../../../.." && pwd)
export PATH=${NODE_BIN:-$B/opt/node/bin}:$PATH TMPDIR=$B/scratch/tmp
export BESU_HOME=${BESU_HOME:-$B/opt/besu-26.8.1} JAVA_HOME=${JAVA_HOME:-$B/opt/jre25/Contents/Home} WORK=$B/besu-native-mci2
export MCI_LOAN_PREPARED=${MCI_LOAN_PREPARED:-$REPO/data/mci_loan/v2/prepared}
export RAPIDSNARK=${RAPIDSNARK:-$B/opt/rapidsnark/bin/prover}
mkdir -p "$TMPDIR"
cd "$REPO"
avail() { # available memory (free + inactive pages) and the swapout counter, from vm_stat
  vm_stat | awk -v ps="$(pagesize)" '/Pages free/ {f=$3} /Pages inactive/ {i=$3} /Swapouts/ {s=$2} END {printf "availMB %d swapouts %d", (f + i) * ps / 1048576, s}'
}
rss_sum() { local s=0 p; for p in "$@"; do s=$((s + $(ps -o rss= -p "$p" 2>/dev/null || echo 0))); done; echo $((s / 1024)); }
sampler() { # <file>: time, available memory, swapouts, Besu RSS MB, node RSS MB
  while sleep 5; do
    besu=$(cat $WORK/pids/*.pid 2>/dev/null)
    nodes=$(pgrep -f "$(basename "$REPO")/(contracts|circuits|scenarios)" 2>/dev/null)
    echo "$(date +%T) $(avail) besuMB $(rss_sum $besu) nodeMB $(rss_sum $nodes)"
  done > "$1" 2>&1
}
RUNS=results/mci_loan/v2/onchain
SCENARIO_SEED=20267005
case "${1:-}" in
  test)
    (cd contracts && node --test offchain/full/test/*.test.mjs) && (cd circuits && node --test test/prover-defaults.test.mjs) \
      && node --test scenarios/mci_loan/tests/*.test.mjs
    ;;
  bench)
    mkdir -p $RUNS
    echo "== alignment benchmark $(date +%T) $(avail)"
    (cd contracts && node offchain/full/bench-psi-full.mjs --reps 2 --scenario mci-loan --out ../$RUNS/psi-bench.json) \
      || { echo "BENCH FAILED"; exit 1; }
    echo "== done $(date +%T) $(avail)"
    ;;
  run)
    K=${2:?run number}; ALIGN=${3:?alignment period in blocks}
    CONFIG=../results/mci_loan/v2/configs/loan2-s$K-wait10s.json
    OUT=$RUNS/run-$K
    if [ -e "$OUT/run.json" ]; then echo "run $K is already recorded in $OUT"; exit 1; fi
    mkdir -p $RUNS
    MEMLOG=$B/scratch/mci2-onchain-run-$K-mem.log
    echo "== network up $(date +%T) $(avail)"
    bash chain/scripts/native-besu.sh up || { echo "NETWORK FAILED"; bash chain/scripts/native-besu.sh down; exit 1; }
    EXPECTED=${GENESIS_SHA256:-$(shasum -a 256 chain/data/genesis.json | cut -d' ' -f1)}
    node chain/scripts/check-network.mjs --genesis chain/data/genesis.json \
      --expected-hash "$EXPECTED" \
      --node-address "$(cat chain/data/power/address)" --rpc http://127.0.0.1:8545 --json > $RUNS/run-$K-network-check.json
    chk=$?
    cat $RUNS/run-$K-network-check.json
    if [ $chk -ne 0 ]; then echo "CHECK FAILED"; bash chain/scripts/native-besu.sh down; exit 1; fi
    sampler $MEMLOG &
    smp=$!
    echo "== run $K (config $CONFIG, run seed $K, scenario seed $SCENARIO_SEED, alignment period $ALIGN blocks) -> $OUT $(date +%T) $(avail)"
    (cd contracts && VFL_MODEL_CONFIG=$CONFIG node offchain/runner3.mjs --run mci-loan --setup mpc-d14 \
      --prover rapidsnark --run-seed $K --scenario-seed $SCENARIO_SEED --alignment-period $ALIGN --out-dir ../$OUT \
      --machine-label "$(sysctl -n hw.model) ($(sysctl -n machdep.cpu.brand_string), $(sysctl -n hw.ncpu) cores, $(( $(sysctl -n hw.memsize) / 1073741824 )) GB), macOS $(sw_vers -productVersion)" \
      --besu-launch "native (chain/scripts/native-besu.sh): Besu on $($JAVA_HOME/bin/java -version 2>&1 | sed -n 2p); docker-compose's node flags, host/ports/paths differ")
    code=$?
    kill $smp 2>/dev/null
    echo "== network down $(date +%T) $(avail)"
    bash chain/scripts/native-besu.sh down
    cp $MEMLOG $OUT/mem.log 2>/dev/null
    echo "RUN $K EXIT $code"
    exit $code
    ;;
  replay)
    K=${2:?run number}
    CONFIG=../results/mci_loan/v2/configs/loan2-s$K-wait10s.json
    echo "== replay run $K (config $CONFIG) $(date +%T) $(avail)"
    (cd contracts && VFL_MODEL_CONFIG=$CONFIG node offchain/full/replay-full.mjs --dir ../$RUNS/run-$K) \
      || { echo "REPLAY FAILED"; exit 1; }
    ;;
  *) echo "stage: test | bench | run <k> <blocks> | replay <k>"; exit 2 ;;
esac
