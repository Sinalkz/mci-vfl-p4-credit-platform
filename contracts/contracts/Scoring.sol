// SPDX-License-Identifier: MIT
pragma solidity ^0.8.28;

import {Reconciliation} from "./Reconciliation.sol";

/// @notice The snarkjs Groth16 verifier of circuits/src/macro_f1_members.circom (9 public inputs).
interface IMembersVerifier {
    function verifyProof(
        uint256[2] calldata a,
        uint256[2][2] calldata b,
        uint256[2] calldata c,
        uint256[9] calldata pubSignals
    ) external view returns (bool);
}

/// @title Scoring figures of one run: fused, each leave-one-out, and the baseline model.
/// @notice All five figures are proven with one circuit and one verification key; a
/// model that leaves passive party p out zero-pads p's inputs (mask bit p = 0).
/// The contract, not the prover, supplies every run-level public input:
///   [C_model, C_fA, gamma_A, gamma_1, gamma_2, gamma_3, seed, mask, F]
/// where C_fA, gamma_A and the seed come from the run, and gamma_i is each passive
/// party's *binding* commitment (its signed root if it equivocated). So all proofs of
/// a run necessarily use the identical C_fA and the same rows; a proof made against
/// any other value fails verification.
contract Scoring {
    uint8 public constant FULL_MASK = 7; // three passive parties

    struct Figure {
        bool set;
        uint256 cModel;
        uint256 value; // F, macro-F1 in units of 1e-4
    }

    Reconciliation public immutable run;
    IMembersVerifier public immutable verifier;
    mapping(uint8 => Figure) public figures;

    event FigureAccepted(uint8 indexed mask, uint256 cModel, uint256 F);

    error WrongMemberCount();
    error NotActiveParty();
    error CommitWindowOpen(uint256 blockNumber, uint256 windowEnd);
    error MaskNotAllowed(uint8 mask);
    error AlreadyScored(uint8 mask);
    error MissingCommitment();
    error InvalidProof();

    constructor(Reconciliation run_, IMembersVerifier verifier_) {
        if (run_.passiveCount() != 3) revert WrongMemberCount();
        run = run_;
        verifier = verifier_;
    }

    /// @notice Fused (7), leave-one-out (6, 5, 3) and baseline (0) are the five figures of a run.
    function maskAllowed(uint8 mask) public pure returns (bool) {
        return mask == FULL_MASK || mask == 0 || mask == 6 || mask == 5 || mask == 3;
    }

    function submitFigure(
        uint8 mask,
        uint256 cModel,
        uint256 F,
        uint256[2] calldata a,
        uint256[2][2] calldata b,
        uint256[2] calldata c
    ) external {
        if (msg.sender != run.activeParty()) revert NotActiveParty();
        uint256 windowEnd = run.commitWindowEnd();
        if (block.number <= windowEnd) revert CommitWindowOpen(block.number, windowEnd);
        if (!maskAllowed(mask)) revert MaskNotAllowed(mask);
        if (figures[mask].set) revert AlreadyScored(mask);

        uint256[9] memory pub;
        pub[0] = cModel;
        pub[1] = uint256(run.bottomModelCommitment());
        pub[2] = uint256(run.tableCommitment());
        for (uint256 i; i < 3; ++i) {
            pub[3 + i] = uint256(run.bindingCommitment(run.passiveParties(i)));
        }
        pub[6] = run.seed();
        pub[7] = mask;
        pub[8] = F;
        for (uint256 i = 1; i < 6; ++i) {
            if (pub[i] == 0) revert MissingCommitment();
        }
        if (!verifier.verifyProof(a, b, c, pub)) revert InvalidProof();

        figures[mask] = Figure({set: true, cModel: cModel, value: F});
        emit FigureAccepted(mask, cModel, F);
    }
}
