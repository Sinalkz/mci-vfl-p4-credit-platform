pragma circom 2.1.9;

// Measurement only (not part of any proof): the three parts of macro_f1_members_var.circom whose size depends on the
// power company's input width FA (8 in the circuit), as breakdown_probes.circom's BDBottomRow and BDParams with FA as
// a parameter; the third, the leaf range checks of tree A, is BDLeafRanges(FA). Every other part (index derivation,
// the four Merkle paths, the passives' leaf range checks, the fusion head, the F1 counts and division, mask and sizeI)
// takes A's row only as f_A's 4 outputs or as one packed leaf, so it does not change with FA.
// f_A's ReLUShift keeps b = 35, as in the circuit: |z| <= FA * 2^30 + 2^31 - 1 < 2^34 for every FA <= 14.
// Used by circuits/scripts/fa-cost.mjs.

include "breakdown_probes.circom";

// Per row: bottom model f_A, FA -> 4, ReLU and rescale (BDBottomRow with FA inputs).
template FABottomRow(FA) {
    signal input W0[4][FA];
    signal input b0[4];
    signal input XA[FA];
    signal output ea[4];
    signal prod[4][FA];
    component fa[4];
    for (var e = 0; e < 4; e++) {
        var z = b0[e];
        for (var f = 0; f < FA; f++) { prod[e][f] <== W0[e][f] * XA[f]; z += prod[e][f]; }
        fa[e] = ReLUShift(35, 12);
        fa[e].in <== z;
        ea[e] <== fa[e].out;
    }
}

// Once: the parameters' range checks and the two Poseidon chains C_fA and C_model (BDParams with FA inputs to f_A).
template FAParams(FA) {
    signal input C_fA;
    signal input C_model;
    signal input W0[4][FA];
    signal input b0[4];
    signal input W1[8][16];
    signal input b1[8];
    signal input W2[5][8];
    signal input b2[5];
    component rW0[4][FA];
    component rb0[4];
    component cfa = ParamChain(4 * FA + 4, FA * 65536 + 4 * 256);
    for (var e = 0; e < 4; e++) {
        for (var f = 0; f < FA; f++) { rW0[e][f] = SignedRange(16); rW0[e][f].in <== W0[e][f]; cfa.p[e * FA + f] <== W0[e][f]; }
        rb0[e] = SignedRange(32); rb0[e].in <== b0[e];
        cfa.p[4 * FA + e] <== b0[e];
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
