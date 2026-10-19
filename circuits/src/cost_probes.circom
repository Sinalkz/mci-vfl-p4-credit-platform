pragma circom 2.1.9;

// Isolated pieces of macro_f1_members.circom, compiled on their own to measure
// what each design choice costs in constraints (circuits/scripts/cost-probes.mjs).

include "macro_f1_members.circom";

// C_fA variant, per evaluation row: the power company's embedding computed
// in-circuit from its features (8 -> 4, ReLU, rescale to 2^12).
template ProbeBottomModelRow() {
    signal input W0[4][8];
    signal input b0[4];
    signal input XA[8];
    signal output ea[4];
    signal prod[4][8];
    component fa[4];
    for (var e = 0; e < 4; e++) {
        var z = b0[e];
        for (var f = 0; f < 8; f++) {
            prod[e][f] <== W0[e][f] * XA[f];
            z += prod[e][f];
        }
        fa[e] = ReLUShift(35, 12);
        fa[e].in <== z;
        ea[e] <== fa[e].out;
    }
}

// C_fA variant, once per proof: range checks of W0, b0 and the Poseidon chain for C_fA.
template ProbeBottomModelFixed() {
    signal input C_fA;
    signal input W0[4][8];
    signal input b0[4];
    component rW0[4][8];
    component rb0[4];
    component cfa = ParamChain(36, 8 * 65536 + 4 * 256);
    for (var e = 0; e < 4; e++) {
        for (var f = 0; f < 8; f++) {
            rW0[e][f] = SignedRange(16); rW0[e][f].in <== W0[e][f];
            cfa.p[e * 8 + f] <== W0[e][f];
        }
        rb0[e] = SignedRange(32); rb0[e].in <== b0[e];
        cfa.p[32 + e] <== b0[e];
    }
    C_fA === cfa.out;
}

// Alternative (b), per row: a membership proof in one more root of depth D,
// plus the range checks of the 4 embedding values that form its leaf.
template ProbeExtraRoot(D) {
    signal input root;
    signal input h[4];
    signal input bits[D];
    signal input sib[D];
    component r[4];
    var leaf = 0;
    for (var e = 0; e < 4; e++) {
        r[e] = SignedRange(16); r[e].in <== h[e];
        leaf += r[e].shifted * (1 << (16 * e));
    }
    component m = MerkleRoot(D);
    m.leaf <== leaf;
    m.pathBits <== bits;
    m.siblings <== sib;
    m.root === root;
}

// A single Poseidon hash with T-1 inputs (binary tree level vs. 4-ary tree level).
template ProbePoseidon(n) {
    signal input x[n];
    signal output y;
    component h = Poseidon(n);
    h.inputs <== x;
    y <== h.out;
}

// Index derivation per row, for reference.
template ProbeSampleIndex(I_SIZE, D) {
    signal input seed;
    signal output bits[D];
    component s = SampleIndex(I_SIZE, D, 0);
    s.seed <== seed;
    bits <== s.bits;
}
