// SPDX-License-Identifier: MIT
pragma solidity ^0.8.28;

import {ECDSA} from "@openzeppelin/contracts/utils/cryptography/ECDSA.sol";
import {EIP712} from "@openzeppelin/contracts/utils/cryptography/EIP712.sol";

/// @title Commitments of one VFL run, and equivocation proofs on the final message.
/// @notice One instance per run. Deadlines are block heights (thesis design decision).
///   Phase A (deployment): members, deposits, the scoring seed s and the schedule are fixed.
///   Phase B (block <= tableDeadline): the active party registers gamma_A, the Poseidon
///     Merkle root of its data table (label and features per aligned ID).
///   End of training (commitWindowStart <= block <= commitWindowEnd):
///     each passive party registers gamma_i, the Poseidon Merkle root of its final-round
///     embedding matrix (the same root it signs, EIP-712, when sending the matrix);
///     the active party registers C_fA, the Poseidon commitment of its bottom model.
/// Every commitment must be a BN254 field element, because scoring proofs take them
/// as public inputs (see Scoring.sol).
/// If a registered gamma_i differs from the root the party signed, the active party
/// proves it: the deposit D_p is forfeited and the signed root becomes binding.
/// @dev The signed context ctx_i = (run, party, round) is carried twice: inside the
/// struct (as the brief requires) and in the EIP-712 domain (verifyingContract,
/// chainId). The 0x1901 prefix of EIP-712 means the signed hash can never be the
/// hash of a valid transaction.
contract Reconciliation is EIP712 {
    /// @notice Final-round marker. A sentinel no training-round index can take.
    uint256 public constant FINAL_ROUND = type(uint256).max;
    uint256 public constant SNARK_SCALAR_FIELD =
        21888242871839275222246405745257275088548364400416034343698204186575808495617;

    bytes32 public constant FINAL_MESSAGE_TYPEHASH =
        keccak256("FinalMessage(address run,address party,uint256 round,bytes32 digest)");

    struct Member {
        bool isPassive;
        bool equivocated;
        uint128 deposit; // participation deposit D_p (stub accounting, token units)
    }

    address public immutable activeParty;
    uint256 public immutable seed;
    uint256 public immutable tableDeadline;
    uint256 public immutable commitWindowStart;
    uint256 public immutable commitWindowEnd;

    address[] public passiveParties;
    mapping(address => Member) public members;
    mapping(address => bytes32) public registeredCommitment;
    /// @notice Digest the party signed; set only once an equivocation is proven.
    mapping(address => bytes32) public signedCommitment;
    /// @notice Forfeited deposits; they join the escrow remainder (decision log D61).
    uint256 public forfeitedToEscrow;
    /// @notice gamma_A: root of the active party's table (label, features) over the aligned IDs.
    bytes32 public tableCommitment;
    /// @notice C_fA: commitment of the active party's bottom model, the same for every scoring proof.
    bytes32 public bottomModelCommitment;

    event TableCommitted(bytes32 gammaA);
    event CommitmentRegistered(address indexed party, bytes32 gamma);
    event BottomModelCommitted(bytes32 cfa);
    event EquivocationProven(address indexed party, bytes32 registered, bytes32 signed, uint256 forfeited);

    error InvalidSchedule();
    error NotActiveParty();
    error NotPassiveParty();
    error ZeroCommitment();
    error NotAFieldElement();
    error AlreadyRegistered();
    error OutsideWindow(uint256 blockNumber, uint256 start, uint256 end);
    error NotRegistered();
    error AlreadyProven();
    error NoEquivocation();
    error SignerMismatch(address recovered);

    constructor(
        address active,
        address[] memory passives,
        uint128 participationDeposit,
        uint256 seed_,
        uint256 tableDeadline_,
        uint256 commitWindowStart_,
        uint256 commitWindowEnd_
    ) EIP712("VFL-Reconciliation", "1") {
        if (!(tableDeadline_ < commitWindowStart_ && commitWindowStart_ <= commitWindowEnd_)) revert InvalidSchedule();
        if (seed_ >= SNARK_SCALAR_FIELD) revert NotAFieldElement();
        activeParty = active;
        seed = seed_;
        tableDeadline = tableDeadline_;
        commitWindowStart = commitWindowStart_;
        commitWindowEnd = commitWindowEnd_;
        for (uint256 i; i < passives.length; ++i) {
            passiveParties.push(passives[i]);
            members[passives[i]] = Member({isPassive: true, equivocated: false, deposit: participationDeposit});
        }
    }

    function passiveCount() external view returns (uint256) {
        return passiveParties.length;
    }

    /// @notice Phase B: the active party's table root gamma_A, at or before tableDeadline.
    function registerTableCommitment(bytes32 gammaA) external {
        if (msg.sender != activeParty) revert NotActiveParty();
        if (block.number > tableDeadline) revert OutsideWindow(block.number, 0, tableDeadline);
        if (tableCommitment != bytes32(0)) revert AlreadyRegistered();
        _checkField(gammaA);
        tableCommitment = gammaA;
        emit TableCommitted(gammaA);
    }

    /// @notice End of training: a passive party's gamma_i = root of its final-round embedding matrix.
    function registerCommitment(bytes32 gamma) external {
        if (!members[msg.sender].isPassive) revert NotPassiveParty();
        _inCommitWindow();
        if (registeredCommitment[msg.sender] != bytes32(0)) revert AlreadyRegistered();
        _checkField(gamma);
        registeredCommitment[msg.sender] = gamma;
        emit CommitmentRegistered(msg.sender, gamma);
    }

    /// @notice End of training, same window as gamma_i: the active party's bottom-model commitment C_fA.
    function registerBottomModel(bytes32 cfa) external {
        if (msg.sender != activeParty) revert NotActiveParty();
        _inCommitWindow();
        if (bottomModelCommitment != bytes32(0)) revert AlreadyRegistered();
        _checkField(cfa);
        bottomModelCommitment = cfa;
        emit BottomModelCommitted(cfa);
    }

    /// @notice Called by the active party with the root `party` signed for this run's final round.
    function proveEquivocation(address party, bytes32 signedDigest, bytes calldata sig) external {
        if (msg.sender != activeParty) revert NotActiveParty();
        bytes32 registered = registeredCommitment[party];
        if (registered == bytes32(0)) revert NotRegistered();
        Member storage m = members[party];
        if (m.equivocated) revert AlreadyProven();
        _checkField(signedDigest);
        if (signedDigest == registered) revert NoEquivocation();

        address signer = ECDSA.recoverCalldata(finalMessageHash(party, signedDigest), sig);
        if (signer != party) revert SignerMismatch(signer);

        uint256 forfeited = m.deposit;
        m.equivocated = true;
        m.deposit = 0;
        forfeitedToEscrow += forfeited;
        signedCommitment[party] = signedDigest;
        emit EquivocationProven(party, registered, signedDigest, forfeited);
    }

    /// @notice The commitment the party is bound to: the signed root if it equivocated, else its registration.
    function bindingCommitment(address party) public view returns (bytes32) {
        return members[party].equivocated ? signedCommitment[party] : registeredCommitment[party];
    }

    /// @notice EIP-712 hash a passive party signs for its final message in this run.
    function finalMessageHash(address party, bytes32 digest) public view returns (bytes32) {
        return _hashTypedDataV4(
            keccak256(abi.encode(FINAL_MESSAGE_TYPEHASH, address(this), party, FINAL_ROUND, digest))
        );
    }

    function _inCommitWindow() private view {
        if (block.number < commitWindowStart || block.number > commitWindowEnd) {
            revert OutsideWindow(block.number, commitWindowStart, commitWindowEnd);
        }
    }

    function _checkField(bytes32 v) private pure {
        if (v == bytes32(0)) revert ZeroCommitment();
        if (uint256(v) >= SNARK_SCALAR_FIELD) revert NotAFieldElement();
    }
}
