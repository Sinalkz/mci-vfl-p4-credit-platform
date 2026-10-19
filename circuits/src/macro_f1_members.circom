pragma circom 2.1.9;

// S2b statement: "the fusion head with commitment C_model, on top of the power
// company's bottom model with commitment C_fA, evaluated on the N rows at the
// indices derived from the public seed s, has floor(1e4 * macro-F1) = F. Each
// row is a member of the power company's table (root gammaA: label and
// features) and of every passive party's final-round embedding matrix (roots
// gammaP[p]). Passive parties whose bit in `mask` is 0 are left out
// (zero-padded), so the fused, leave-one-out and baseline models share this
// one circuit."
//
// Public inputs : C_model, C_fA, gammaA, gammaP[3], seed, mask, F
// Private inputs: bottom model W0, b0; head W1, b1, W2, b2; rows XA, Y, HP; Merkle siblings
//
// Index rule (sampling with replacement; duplicates are kept and counted):
//   idx_k = (Poseidon(seed, k) mod 2^128) mod I_SIZE,  k = 0..N-1
// Merkle trees: binary Poseidon(2), depth DEPTH = ceil(log2 I_SIZE), leaves at
// positions >= I_SIZE are 0. Leaves are the packed rows (no leaf hash; the
// fixed depth separates leaves from inner nodes):
//   gammaA leaf = sum_f (x_f + 2^15) 2^(16 f) + y 2^128
//   gammaP leaf = sum_e (h_e + 2^15) 2^(16 e)
// Fixed point: see py/s2b_model.py. Bounds: |z0| < 2^34, e_A < 2^22,
// |z1| < 2^40, |z2| < 2^59, all enforced by range checks.

include "circomlib/circuits/switcher.circom";
include "macro_f1.circom"; // SignedRange, ReLU, ArgmaxOneHot, LabelOneHot, bitsFor, Poseidon, bitify, comparators

// Poseidon chain over P values in chunks of 15 (zero-padded), seeded with TAG.
template ParamChain(P, TAG) {
    signal input p[P];
    signal output out;
    var CH = 15;
    var NC = (P + CH - 1) \ CH;
    component h[NC];
    for (var c = 0; c < NC; c++) {
        h[c] = Poseidon(CH + 1);
        if (c == 0) { h[c].inputs[0] <== TAG; } else { h[c].inputs[0] <== h[c - 1].out; }
        for (var j = 0; j < CH; j++) {
            if (c * CH + j < P) { h[c].inputs[j + 1] <== p[c * CH + j]; } else { h[c].inputs[j + 1] <== 0; }
        }
    }
    out <== h[NC - 1].out;
}

// out = floor(max(in, 0) / 2^SHIFT), given -2^(b-1) <= in < 2^(b-1).
template ReLUShift(b, SHIFT) {
    signal input in;
    signal output out;
    component n2b = Num2Bits(b);
    n2b.in <== in + (1 << (b - 1));
    var t = 0;
    for (var i = SHIFT; i < b - 1; i++) { t += n2b.out[i] * (1 << (i - SHIFT)); }
    out <== t * n2b.out[b - 1];
}

// Binary Poseidon Merkle root; pathBits[i] = 1 when the node at level i is a right child.
template MerkleRoot(DEPTH) {
    signal input leaf;
    signal input pathBits[DEPTH];
    signal input siblings[DEPTH];
    signal output root;
    component sw[DEPTH];
    component h[DEPTH];
    for (var i = 0; i < DEPTH; i++) {
        sw[i] = Switcher();
        sw[i].sel <== pathBits[i];
        if (i == 0) { sw[i].L <== leaf; } else { sw[i].L <== h[i - 1].out; }
        sw[i].R <== siblings[i];
        h[i] = Poseidon(2);
        h[i].inputs[0] <== sw[i].outL;
        h[i].inputs[1] <== sw[i].outR;
    }
    root <== h[DEPTH - 1].out;
}

// idx = (Poseidon(seed, K) mod 2^128) mod I_SIZE; outputs its DEPTH bits, LSB first.
// Num2Bits_strict gives the canonical decomposition, so the prover has no choice
// of index; only the low 128 bits are used, so q*I_SIZE + r cannot wrap the field.
template SampleIndex(I_SIZE, DEPTH, K) {
    signal input seed;
    signal output bits[DEPTH];
    component h = Poseidon(2);
    h.inputs[0] <== seed;
    h.inputs[1] <== K;
    component hb = Num2Bits_strict();
    hb.in <== h.out;
    var low = 0;
    for (var i = 0; i < 128; i++) { low += hb.out[i] * (1 << i); }
    signal q;
    signal r;
    q <-- low \ I_SIZE;
    r <-- low % I_SIZE;
    low === q * I_SIZE + r;
    component qRange = Num2Bits(128);
    qRange.in <== q;
    component rBits = Num2Bits(DEPTH);
    rBits.in <== r;
    component lt = LessThan(DEPTH + 1);
    lt.in[0] <== r;
    lt.in[1] <== I_SIZE;
    lt.out === 1;
    bits <== rBits.out;
}

template MacroF1Members(N, I_SIZE, DEPTH) {
    var FA = 8;
    var E = 4;
    var P = 3;
    var D = 16;
    var H = 8;
    var K = 5;
    var F1_SCALE = 10000;

    signal input C_model;
    signal input C_fA;
    signal input gammaA;
    signal input gammaP[P];
    signal input seed;
    signal input mask;
    signal input F;

    signal input W0[E][FA];
    signal input b0[E];
    signal input W1[H][D];
    signal input b1[H];
    signal input W2[K][H];
    signal input b2[K];

    signal input XA[N][FA];
    signal input Y[N];
    signal input HP[N][P][E];
    signal input sibA[N][DEPTH];
    signal input sibP[N][P][DEPTH];

    // ---- 1. Parameters: range checks and the two model commitments -------------
    component rW0[E][FA];
    component rb0[E];
    component cfa = ParamChain(E * FA + E, FA * 65536 + E * 256);
    for (var e = 0; e < E; e++) {
        for (var f = 0; f < FA; f++) {
            rW0[e][f] = SignedRange(16); rW0[e][f].in <== W0[e][f];
            cfa.p[e * FA + f] <== W0[e][f];
        }
        rb0[e] = SignedRange(32); rb0[e].in <== b0[e];
        cfa.p[E * FA + e] <== b0[e];
    }
    C_fA === cfa.out;

    component rW1[H][D];
    component rb1[H];
    component rW2[K][H];
    component rb2[K];
    component cm = ParamChain(H * D + H + K * H + K, D * 65536 + H * 256 + K);
    for (var h = 0; h < H; h++) {
        for (var d = 0; d < D; d++) {
            rW1[h][d] = SignedRange(16); rW1[h][d].in <== W1[h][d];
            cm.p[h * D + d] <== W1[h][d];
        }
        rb1[h] = SignedRange(32); rb1[h].in <== b1[h];
        cm.p[H * D + h] <== b1[h];
    }
    for (var k = 0; k < K; k++) {
        for (var h = 0; h < H; h++) {
            rW2[k][h] = SignedRange(16); rW2[k][h].in <== W2[k][h];
            cm.p[H * D + H + k * H + h] <== W2[k][h];
        }
        rb2[k] = SignedRange(48); rb2[k].in <== b2[k];
        cm.p[H * D + H + K * H + k] <== b2[k];
    }
    C_model === cm.out;

    // ---- 2. Mask: a left-out passive party's input columns get weight 0 ---------
    component mb = Num2Bits(P);
    mb.in <== mask;
    signal W1m[H][D];
    for (var h = 0; h < H; h++) {
        for (var d = 0; d < E; d++) { W1m[h][d] <== W1[h][d]; }
        for (var p = 0; p < P; p++) {
            for (var e = 0; e < E; e++) { W1m[h][E + E * p + e] <== W1[h][E + E * p + e] * mb.out[p]; }
        }
    }

    // ---- 3. Rows ----------------------------------------------------------------
    component si[N];
    component lab[N];
    component rXA[N][FA];
    component rHP[N][P][E];
    component memA[N];
    component memP[N][P];
    component fa[N][E];
    component relu[N][H];
    component am[N];
    signal prodA[N][E][FA];
    signal prod1[N][H][D];
    signal prod2[N][K][H];
    signal tp[N][K];

    var TPc[K];
    var PCc[K];
    var LCc[K];
    for (var k = 0; k < K; k++) { TPc[k] = 0; PCc[k] = 0; LCc[k] = 0; }

    for (var n = 0; n < N; n++) {
        si[n] = SampleIndex(I_SIZE, DEPTH, n);
        si[n].seed <== seed;

        // Power company's row: label and features, member of gammaA at idx_n.
        lab[n] = LabelOneHot(K);
        lab[n].y <== Y[n];
        var leafA = Y[n] * (1 << (16 * FA));
        for (var f = 0; f < FA; f++) {
            rXA[n][f] = SignedRange(16); rXA[n][f].in <== XA[n][f];
            leafA += rXA[n][f].shifted * (1 << (16 * f));
        }
        memA[n] = MerkleRoot(DEPTH);
        memA[n].leaf <== leafA;
        memA[n].pathBits <== si[n].bits;
        memA[n].siblings <== sibA[n];
        memA[n].root === gammaA;

        // Each passive party's embedding row, member of gammaP[p] at the same idx_n.
        for (var p = 0; p < P; p++) {
            var leafP = 0;
            for (var e = 0; e < E; e++) {
                rHP[n][p][e] = SignedRange(16); rHP[n][p][e].in <== HP[n][p][e];
                leafP += rHP[n][p][e].shifted * (1 << (16 * e));
            }
            memP[n][p] = MerkleRoot(DEPTH);
            memP[n][p].leaf <== leafP;
            memP[n][p].pathBits <== si[n].bits;
            memP[n][p].siblings <== sibP[n][p];
            memP[n][p].root === gammaP[p];
        }

        // Bottom model f_A, computed in-circuit and rescaled to 2^12.
        for (var e = 0; e < E; e++) {
            var z0 = b0[e];
            for (var f = 0; f < FA; f++) {
                prodA[n][e][f] <== W0[e][f] * XA[n][f];
                z0 += prodA[n][e][f];
            }
            fa[n][e] = ReLUShift(35, 12);
            fa[n][e].in <== z0;
        }

        // Fusion head on [e_A, h_1, h_2, h_3] with masked columns.
        for (var h = 0; h < H; h++) {
            var z1 = b1[h];
            for (var d = 0; d < E; d++) {
                prod1[n][h][d] <== W1m[h][d] * fa[n][d].out;
                z1 += prod1[n][h][d];
            }
            for (var p = 0; p < P; p++) {
                for (var e = 0; e < E; e++) {
                    prod1[n][h][E + E * p + e] <== W1m[h][E + E * p + e] * HP[n][p][e];
                    z1 += prod1[n][h][E + E * p + e];
                }
            }
            relu[n][h] = ReLU(41);
            relu[n][h].in <== z1;
        }
        am[n] = ArgmaxOneHot(K, 60);
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

    // ---- 4. Macro-F1, one quotient-and-remainder division (as in macro_f1.circom) -
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
    component fRange = Num2Bits(14);
    fRange.in <== F;
    component rRange = Num2Bits(NB);
    rRange.in <== r;
    component rLtB = LessThan(NB);
    rLtB.in[0] <== r;
    rLtB.in[1] <== B;
    rLtB.out === 1;
}
