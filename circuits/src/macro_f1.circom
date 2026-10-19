pragma circom 2.1.9;

// Statement: "the model whose commitment is C_model, evaluated on the
// evaluation set whose commitment is C_eval, has macro-F1 equal to F
// (in 10^-4 units, floored)."
//
// Public inputs : C_model, C_eval, F
// Private inputs: MLP weights (W1, b1, W2, b2); evaluation rows X[N][D], labels Y[N]
//
// Fixed point (see py/s2_model.py): x, W1, W2 are int16 at scale 2^12,
// b1 is int32 at scale 2^24, b2 is int48 at scale 2^36. Every private value is
// range-checked, which bounds |z1| < 2^35 and |z2| < 2^54, so no field
// wraparound can occur and the circuit computes exactly the integer network.
// No rescaling between layers is needed because argmax is scale-invariant.

include "circomlib/circuits/poseidon.circom";
include "circomlib/circuits/bitify.circom";
include "circomlib/circuits/comparators.circom";

function bitsFor(a) {
    var r = 0;
    var p = 1;
    while (p <= a) { p *= 2; r++; }
    return r;
}

// Constrains -2^(b-1) <= in < 2^(b-1); outputs in + 2^(b-1), which lies in [0, 2^b).
template SignedRange(b) {
    signal input in;
    signal output shifted;
    shifted <== in + (1 << (b - 1));
    component n2b = Num2Bits(b);
    n2b.in <== shifted;
}

// out = max(in, 0), given -2^(b-1) <= in < 2^(b-1): the top bit of in + 2^(b-1) is the sign.
template ReLU(b) {
    signal input in;
    signal output out;
    component n2b = Num2Bits(b);
    n2b.in <== in + (1 << (b - 1));
    out <== in * n2b.out[b - 1];
}

// One-hot of the FIRST maximum of z, given |z_k| < 2^(b-1).
// The prover supplies the one-hot as a hint; the circuit checks
//   zmax - z_j - [j < argmax] >= 0 for every j,
// i.e. zmax >= z_j, strictly for indices before the argmax (tie -> smallest index).
template ArgmaxOneHot(K, b) {
    signal input z[K];
    signal output onehot[K];

    var best = 0;
    var bestVal = z[0];
    for (var k = 1; k < K; k++) {
        if (z[k] > bestVal) { best = k; bestVal = z[k]; }
    }
    var s = 0;
    for (var k = 0; k < K; k++) {
        onehot[k] <-- (k == best) ? 1 : 0;
        onehot[k] * (onehot[k] - 1) === 0;
        s += onehot[k];
    }
    s === 1;

    signal sel[K];
    var zmax = 0;
    for (var k = 0; k < K; k++) {
        sel[k] <== onehot[k] * z[k];
        zmax += sel[k];
    }
    component nonneg[K];
    for (var j = 0; j < K; j++) {
        var later = 0;
        for (var k = j + 1; k < K; k++) { later += onehot[k]; }
        nonneg[j] = Num2Bits(b + 1);
        nonneg[j].in <== zmax - z[j] - later;
    }
}

// One-hot of a label y in {0..K-1} (hint + binary + sum + reconstruction).
template LabelOneHot(K) {
    signal input y;
    signal output onehot[K];
    var s = 0;
    var v = 0;
    for (var k = 0; k < K; k++) {
        onehot[k] <-- (y == k) ? 1 : 0;
        onehot[k] * (onehot[k] - 1) === 0;
        s += onehot[k];
        v += k * onehot[k];
    }
    s === 1;
    v === y;
}

// MERKLE = 0: C_eval is a Poseidon hash chain acc_i = H(acc_{i-1}, row_i), acc_0 = N.
// MERKLE = 1: C_eval is a Poseidon Merkle root over row leaves (N a power of two);
//             compiled only to compare constraint counts.
template MacroF1(N, D, H, K, MERKLE) {
    signal input C_model;
    signal input C_eval;
    signal input F;

    signal input W1[H][D];
    signal input b1[H];
    signal input W2[K][H];
    signal input b2[K];
    signal input X[N][D];
    signal input Y[N];

    var F1_SCALE = 10000;
    var XB = 16;  // bits of x, W1, W2
    var B1B = 32;
    var B2B = 48;
    var Z1B = 36; // |z1| < 2^35
    var Z2B = 55; // |z2| < 2^54

    // ---- 1. Model: range checks and commitment ---------------------------------
    component rW1[H][D];
    component rb1[H];
    component rW2[K][H];
    component rb2[K];
    for (var h = 0; h < H; h++) {
        for (var d = 0; d < D; d++) { rW1[h][d] = SignedRange(XB); rW1[h][d].in <== W1[h][d]; }
        rb1[h] = SignedRange(B1B); rb1[h].in <== b1[h];
    }
    for (var k = 0; k < K; k++) {
        for (var h = 0; h < H; h++) { rW2[k][h] = SignedRange(XB); rW2[k][h].in <== W2[k][h]; }
        rb2[k] = SignedRange(B2B); rb2[k].in <== b2[k];
    }

    // Parameters in the order W1 (row-major), b1, W2 (row-major), b2, zero-padded
    // to chunks of 15; chain acc_c = Poseidon(acc_{c-1}, chunk_c), acc_{-1} = tag(D,H,K).
    var P = H * D + H + K * H + K;
    var CH = 15;
    var NC = (P + CH - 1) \ CH;
    signal mflat[NC * CH];
    var idx = 0;
    for (var h = 0; h < H; h++) { for (var d = 0; d < D; d++) { mflat[idx] <== W1[h][d]; idx++; } }
    for (var h = 0; h < H; h++) { mflat[idx] <== b1[h]; idx++; }
    for (var k = 0; k < K; k++) { for (var h = 0; h < H; h++) { mflat[idx] <== W2[k][h]; idx++; } }
    for (var k = 0; k < K; k++) { mflat[idx] <== b2[k]; idx++; }
    for (var i = P; i < NC * CH; i++) { mflat[i] <== 0; }

    component mh[NC];
    for (var c = 0; c < NC; c++) {
        mh[c] = Poseidon(CH + 1);
        if (c == 0) { mh[c].inputs[0] <== D * 65536 + H * 256 + K; }
        else { mh[c].inputs[0] <== mh[c - 1].out; }
        for (var j = 0; j < CH; j++) { mh[c].inputs[j + 1] <== mflat[c * CH + j]; }
    }
    C_model === mh[NC - 1].out;

    // ---- 2. Rows: range checks, packing, forward pass, argmax, labels ----------
    // Row packing: D values (x + 2^15, 16 bits each) and the label form D+1
    // 16-bit slots, 15 slots per field element.
    var NP = (D + 1 + 14) \ 15;

    component rX[N][D];
    component relu[N][H];
    component am[N];
    component lab[N];
    signal prod1[N][H][D];
    signal prod2[N][K][H];
    signal tp[N][K];
    signal packed[N][NP];

    var TPc[K];
    var PCc[K];
    var LCc[K];
    for (var k = 0; k < K; k++) { TPc[k] = 0; PCc[k] = 0; LCc[k] = 0; }

    for (var n = 0; n < N; n++) {
        for (var d = 0; d < D; d++) { rX[n][d] = SignedRange(XB); rX[n][d].in <== X[n][d]; }

        lab[n] = LabelOneHot(K);
        lab[n].y <== Y[n];

        var pk[NP];
        for (var e = 0; e < NP; e++) { pk[e] = 0; }
        for (var d = 0; d < D; d++) { pk[d \ 15] += rX[n][d].shifted * (1 << (16 * (d % 15))); }
        pk[D \ 15] += Y[n] * (1 << (16 * (D % 15)));
        for (var e = 0; e < NP; e++) { packed[n][e] <== pk[e]; }

        for (var h = 0; h < H; h++) {
            var z1 = b1[h];
            for (var d = 0; d < D; d++) {
                prod1[n][h][d] <== W1[h][d] * X[n][d];
                z1 += prod1[n][h][d];
            }
            relu[n][h] = ReLU(Z1B);
            relu[n][h].in <== z1;
        }

        am[n] = ArgmaxOneHot(K, Z2B);
        for (var k = 0; k < K; k++) {
            var z2 = b2[k];
            for (var h = 0; h < H; h++) {
                prod2[n][k][h] <== W2[k][h] * relu[n][h].out;
                z2 += prod2[n][k][h];
            }
            am[n].z[k] <== z2;
        }

        for (var k = 0; k < K; k++) {
            tp[n][k] <== am[n].onehot[k] * lab[n].onehot[k];
            TPc[k] += tp[n][k];
            PCc[k] += am[n].onehot[k];
            LCc[k] += lab[n].onehot[k];
        }
    }

    // ---- 3. Evaluation-set commitment ------------------------------------------
    if (MERKLE == 0) {
        component eh[N];
        for (var n = 0; n < N; n++) {
            eh[n] = Poseidon(NP + 1);
            if (n == 0) { eh[n].inputs[0] <== N; }
            else { eh[n].inputs[0] <== eh[n - 1].out; }
            for (var e = 0; e < NP; e++) { eh[n].inputs[e + 1] <== packed[n][e]; }
        }
        C_eval === eh[N - 1].out;
    } else {
        component leaf[N];
        component node[N - 1];
        signal tree[2 * N];
        tree[0] <== 0;
        for (var n = 0; n < N; n++) {
            leaf[n] = Poseidon(NP);
            for (var e = 0; e < NP; e++) { leaf[n].inputs[e] <== packed[n][e]; }
            tree[N + n] <== leaf[n].out;
        }
        for (var t = 0; t < N - 1; t++) {
            var i = N - 1 - t;
            node[i - 1] = Poseidon(2);
            node[i - 1].inputs[0] <== tree[2 * i];
            node[i - 1].inputs[1] <== tree[2 * i + 1];
            tree[i] <== node[i - 1].out;
        }
        C_eval === tree[1];
    }

    // ---- 4. Macro-F1 --------------------------------------------------------------
    // Per class, 2TP + FP + FN = (#predicted) + (#labelled). A class with zero
    // there is empty: excluded from the mean (its denominator is replaced by 1
    // and its numerator 2TP is 0). With K' non-empty classes,
    //   F1_macro = (1/K') * sum_c 2TP_c / D_c = A / (B * 10^4)
    //   A = 10^4 * sum_c 2TP_c * prod_{j != c} D'_j,   B = K' * prod_c D'_c
    // and F = floor(A / B) is proven as A = B*F + r, 0 <= r < B.
    signal TP[K];
    signal Dc[K];
    signal Dp[K];
    component empty[K];
    var nonEmpty = 0;
    for (var k = 0; k < K; k++) {
        TP[k] <== TPc[k];
        Dc[k] <== PCc[k] + LCc[k];
        empty[k] = IsZero();
        empty[k].in <== Dc[k];
        Dp[k] <== Dc[k] + empty[k].out;
        nonEmpty += 1 - empty[k].out;
    }
    signal pre[K + 1];
    signal suf[K + 1];
    pre[0] <== 1;
    suf[K] <== 1;
    for (var k = 0; k < K; k++) { pre[k + 1] <== pre[k] * Dp[k]; }
    for (var t = 0; t < K; t++) { suf[K - 1 - t] <== suf[K - t] * Dp[K - 1 - t]; }
    signal except[K];
    signal term[K];
    var A = 0;
    for (var k = 0; k < K; k++) {
        except[k] <== pre[k] * suf[k + 1];
        term[k] <== TP[k] * except[k];
        A += 2 * F1_SCALE * term[k];
    }
    signal B;
    B <== nonEmpty * pre[K];

    var NB = bitsFor(K * (2 * N) ** K);
    signal r;
    r <-- A % B;
    signal BF;
    BF <== B * F;
    A === BF + r;
    component fRange = Num2Bits(14);  // F < 2^14 rules out field wraparound
    fRange.in <== F;
    component rRange = Num2Bits(NB);
    rRange.in <== r;
    component rLtB = LessThan(NB);
    rLtB.in[0] <== r;
    rLtB.in[1] <== B;
    rLtB.out === 1;
}
