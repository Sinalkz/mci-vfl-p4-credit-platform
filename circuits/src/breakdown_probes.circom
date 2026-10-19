pragma circom 2.1.9;

// RESULTS-PROOFCOST item 3: the parts of macro_f1_members_var.circom (the M1b scoring circuit), each compiled on its
// own to attribute the circuit's constraints (circuits/scripts/breakdown.mjs). Each probe copies the corresponding lines
// of MacroF1MembersVar; the scoring circuit itself is not changed. Per-row parts are multiplied by N (and by 3 for the
// passive trees); the difference between the sum and the full circuit is reported as a residual.

include "macro_f1_members_var.circom";

// Per row: idx_n = (Poseidon(seed, n) mod 2^128) mod sizeI and its MAX_DEPTH path bits (section 4, si[n]).
template BDIndex(MAX_DEPTH) {
    signal input seed;
    signal input sizeI;
    signal output bits[MAX_DEPTH];
    component s = SampleIndexVar(MAX_DEPTH, 0);
    s.seed <== seed;
    s.sizeI <== sizeI;
    bits <== s.bits;
}

// Per row and tree: the Merkle path from a leaf to a public root (memA[n], memP[n][p]).
template BDMerklePath(MAX_DEPTH) {
    signal input root;
    signal input leaf;
    signal input bits[MAX_DEPTH];
    signal input sib[MAX_DEPTH];
    component m = MerkleRoot(MAX_DEPTH);
    m.leaf <== leaf;
    m.pathBits <== bits;
    m.siblings <== sib;
    m.root === root;
}

// Per row and tree: the range checks that pack a leaf (rXA[n][f]: 8 values for A; rHP[n][p][e]: 4 per passive).
template BDLeafRanges(M) {
    signal input v[M];
    signal output leaf;
    component r[M];
    var acc = 0;
    for (var i = 0; i < M; i++) {
        r[i] = SignedRange(16); r[i].in <== v[i];
        acc += r[i].shifted * (1 << (16 * i));
    }
    leaf <== acc;
}

// Per row: bottom model f_A, 8 -> 4, ReLU and rescale (prodA, fa[n][e]).
template BDBottomRow() {
    signal input W0[4][8];
    signal input b0[4];
    signal input XA[8];
    signal output ea[4];
    signal prod[4][8];
    component fa[4];
    for (var e = 0; e < 4; e++) {
        var z = b0[e];
        for (var f = 0; f < 8; f++) { prod[e][f] <== W0[e][f] * XA[f]; z += prod[e][f]; }
        fa[e] = ReLUShift(35, 12);
        fa[e].in <== z;
        ea[e] <== fa[e].out;
    }
}

// Per row: fusion head 16 -> 8 -> 5 on [e_A, h_1, h_2, h_3] with the masked weights, and the argmax (prod1, relu, prod2, am).
template BDFusionRow() {
    signal input W1m[8][16];
    signal input b1[8];
    signal input W2[5][8];
    signal input b2[5];
    signal input x[16];
    signal output onehot[5];
    signal prod1[8][16];
    signal prod2[5][8];
    component relu[8];
    for (var h = 0; h < 8; h++) {
        var z1 = b1[h];
        for (var d = 0; d < 16; d++) { prod1[h][d] <== W1m[h][d] * x[d]; z1 += prod1[h][d]; }
        relu[h] = ReLU(41);
        relu[h].in <== z1;
    }
    component am = ArgmaxOneHot(5, 60);
    for (var k = 0; k < 5; k++) {
        var z2 = b2[k];
        for (var h = 0; h < 8; h++) { prod2[k][h] <== W2[k][h] * relu[h].out; z2 += prod2[k][h]; }
        am.z[k] <== z2;
    }
    onehot <== am.onehot;
}

// Per row: the label's one-hot and the true-positive indicators that feed the F1 counts (lab[n], tp[n][k]).
template BDCountRow() {
    signal input y;
    signal input pred[5];
    signal output tp[5];
    component lab = LabelOneHot(5);
    lab.y <== y;
    for (var k = 0; k < 5; k++) { tp[k] <== pred[k] * lab.onehot[k]; }
}

// Once: macro-F1 from the per-class counts, one quotient-and-remainder division (section 5).
template BDMacroF1(N) {
    var K = 5;
    var F1_SCALE = 10000;
    signal input TPc[K];
    signal input PCc[K];
    signal input LCc[K];
    signal input F;
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

// Once: parameter range checks and the model commitments C_fA and C_model (section 1).
template BDParams() {
    signal input C_fA;
    signal input C_model;
    signal input W0[4][8];
    signal input b0[4];
    signal input W1[8][16];
    signal input b1[8];
    signal input W2[5][8];
    signal input b2[5];
    component rW0[4][8];
    component rb0[4];
    component cfa = ParamChain(4 * 8 + 4, 8 * 65536 + 4 * 256);
    for (var e = 0; e < 4; e++) {
        for (var f = 0; f < 8; f++) { rW0[e][f] = SignedRange(16); rW0[e][f].in <== W0[e][f]; cfa.p[e * 8 + f] <== W0[e][f]; }
        rb0[e] = SignedRange(32); rb0[e].in <== b0[e];
        cfa.p[32 + e] <== b0[e];
    }
    C_fA === cfa.out;
    component rW1[8][16];
    component rb1[8];
    component rW2[5][8];
    component rb2[5];
    component cm = ParamChain(8 * 16 + 8 + 5 * 8 + 5, 16 * 65536 + 8 * 256 + 5);
    for (var h = 0; h < 8; h++) {
        for (var d = 0; d < 16; d++) { rW1[h][d] = SignedRange(16); rW1[h][d].in <== W1[h][d]; cm.p[h * 16 + d] <== W1[h][d]; }
        rb1[h] = SignedRange(32); rb1[h].in <== b1[h];
        cm.p[128 + h] <== b1[h];
    }
    for (var k = 0; k < 5; k++) {
        for (var h = 0; h < 8; h++) { rW2[k][h] = SignedRange(16); rW2[k][h].in <== W2[k][h]; cm.p[136 + k * 8 + h] <== W2[k][h]; }
        rb2[k] = SignedRange(48); rb2[k].in <== b2[k];
        cm.p[176 + k] <== b2[k];
    }
    C_model === cm.out;
}

// Once: the mask on the passive columns (section 2) and 1 <= sizeI <= 2^MAX_DEPTH (section 3).
template BDMaskSize(MAX_DEPTH) {
    signal input mask;
    signal input W1[8][16];
    signal input sizeI;
    signal output W1m[8][16];
    component mb = Num2Bits(3);
    mb.in <== mask;
    for (var h = 0; h < 8; h++) {
        for (var d = 0; d < 4; d++) { W1m[h][d] <== W1[h][d]; }
        for (var p = 0; p < 3; p++) {
            for (var e = 0; e < 4; e++) { W1m[h][4 + 4 * p + e] <== W1[h][4 + 4 * p + e] * mb.out[p]; }
        }
    }
    component sizeBits = Num2Bits(MAX_DEPTH + 1);
    sizeBits.in <== sizeI;
    component sizeCap = LessThan(MAX_DEPTH + 2);
    sizeCap.in[0] <== sizeI;
    sizeCap.in[1] <== (1 << MAX_DEPTH) + 1;
    sizeCap.out === 1;
}
