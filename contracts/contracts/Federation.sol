// SPDX-License-Identifier: MIT
pragma solidity ^0.8.28;

import {IERC20} from "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import {SafeERC20} from "@openzeppelin/contracts/token/ERC20/utils/SafeERC20.sol";
import {ECDSA} from "@openzeppelin/contracts/utils/cryptography/ECDSA.sol";
import {SettlementMath} from "./SettlementMath.sol";

/// @notice The M1b Groth16 verifier (snarkjs export): 10 public inputs
/// [C_model, C_fA, gammaA, gammaP[3], seed, sizeI, mask, F].
interface IScoreVerifier {
    function verifyProof(uint256[2] calldata a, uint256[2][2] calldata b, uint256[2] calldata c, uint256[10] calldata pub)
        external
        view
        returns (bool);
}

/// @notice Terms of one federation run, proposed by the active party A in one transaction
/// (FederationFactory.create) and immutable afterwards. Passives accept them by registering.
struct Proposal {
    // members
    address activeParty; // must equal the caller of factory.create
    bytes activeChannelKey; // A's channel public key, 32 bytes (X25519); A does not register
    address[] passives; // allowlist; this order is member index 1..n
    address arbiter;
    // measurement
    uint8 metricId; // 1 = macro-F1; the only accepted value
    uint256[] classBoundaries; // exactly 4, strictly increasing -> 5 tariff classes
    bytes32 modelConfigHash; // hash of the agreed model configuration
    uint32 T; // training rounds
    uint32 N; // evaluation draws
    uint8 maxDepth; // D106: cap on the aligned set, sizeI <= 2^maxDepth; 1..20
    address verifier; // Groth16 verifier contract (holds the verification key)
    bytes32 vkHash; // hash of the verification key file
    // time, in blocks
    uint64 setupPeriod; // setupDeadline = creation block + setupPeriod
    uint64 alignmentPeriod; // M2: alignmentDeadline = startBlock + alignmentPeriod
    uint64 revealWindow; // reveal window of A, then of the passives
    uint64 deltaR; // M3: reconciliation window after trainingEndBlock, and each relay window
    uint64 deltaA; // M4: the response window after Scored is K * deltaA blocks
    uint64 scoringPeriod; // M3: scoringDeadline = reconciliationEndBlock + scoringPeriod
    uint32 K; // M4: see deltaA
    uint64 finalWait; // M4, D129: blocks A waits for every commitFinal before commitFA; an off-chain rule, stored only
    // payment
    address token;
    uint256 B; // pool
    uint256 rho; // M4: tokens per unit of performance (10^-4 macro-F1); payments are rho * a_i
    uint256 Dp; // participation deposit
    uint256 Dc; // M3: collected from A by fileNonReceipt; M4 (D130): paid to the arbiter when the objection closes
}

/// @notice What a Federation reads from the factory that creates it (M4).
interface ISettlementSource {
    function settlementMath() external view returns (address);
}

/// @title One VFL federation run: setup, alignment, seed reveal (M1, M2), training,
/// reconciliation and scoring (M3).
/// @notice Created by FederationFactory, which moves the pool B into it in the same
/// transaction. Every deadline is a block number (thesis design decision: S1 showed
/// that timestamps jump across a halt). No owner, admin, pause, proxy or setter.
///   Setup      passives register (collects Dp), the arbiter accepts, A and every
///              passive commit to a random number; the last of these starts the run.
///   Alignment  off-chain PSI; A commits gamma_I, |I| and gamma_A by alignmentDeadline,
///              or anyone ends the run with resolveAlignmentTimeout (D107).
///   Revealing  A reveals first, then the passives; the last reveal fixes the seed s.
///   Training   off-chain (D108); each passive registers gamma_i (commitFinal); A ends
///              training with commitFA; a passive without gamma_i is counted as left.
///   Reconciliation (deltaR blocks from trainingEndBlock): equivocation proofs, resend
///              requests, non-receipt objections and the arbiter relay (D110, D114).
///   Scoring    A submits FULL, LOO_j and BASELINE with Groth16 proofs; the contract fills
///              every public input but C_model and F itself. Timeout: D115.
/// Refunds and payouts are credited to `owed` and pulled with `withdraw`.
contract Federation {
    using SafeERC20 for IERC20;

    enum State {
        Setup,
        Alignment,
        Revealing,
        Training, // entered by the reveal that fixes the seed s
        SetupAborted,
        RevealFailed,
        AlignmentFailed,
        Reconciliation,
        Scoring,
        Scored,
        ScoringFailed,
        Settled // M4: after the normal path, an upheld alignment objection or the backup path
    }

    /// @notice Figures of the scoring phase; LOO carries the member index j of the left-out passive.
    /// D139: FULL is F_r (a retrained fusion model, its own C_model); FULL_JOINT is F_j (the fusion model
    /// trained jointly during split training, committed at commitFA as gammaG).
    enum Kind {
        FULL,
        LOO,
        BASELINE,
        FULL_JOINT
    }

    enum Objection {
        None,
        Open,
        Certified,
        TimedOut
    }

    struct Member {
        uint8 index; // 1..n for passives, 0 otherwise (A has an entry with index 0)
        bool registered;
        bool committed;
        bool revealed;
        bytes32 commitment; // keccak256(abi.encode(r, member, address(this)))
        uint256 r; // the revealed number
        bytes channelKey;
        // M3, passives only
        bool finalCommitted; // commitFinal called
        bool left; // counted as left (D109: no gamma at commitFA; D114: relay timeout); share zero
        bool equivocated; // proveEquivocation succeeded; gamma is the signed root
        bool dpForfeited; // Dp moved to forfeitedToPool
        Objection objection;
        uint64 resendRequestBlock; // 0 = no request
        uint64 relayDeadline; // last block of the relay window opened by fileNonReceipt
        bytes32 gamma; // the binding final-round root
        bool acknowledged; // M4: acknowledge() called in the response window
    }

    struct Figure {
        bool accepted;
        uint256 F;
        bytes32 cModel;
    }

    /// @notice Member data as returned by `member`.
    struct MemberView {
        uint8 index;
        bool registered;
        bytes channelKey;
        bool committed;
        bytes32 commitment;
        bool revealed;
    }

    uint256 public constant BN254_R =
        21888242871839275222246405745257275088548364400416034343698204186575808495617;
    uint8 public constant METRIC_MACRO_F1 = 1;
    uint256 public constant MIN_PASSIVES = 3;
    uint256 public constant MAX_PASSIVES = 10;
    uint256 public constant CLASS_BOUNDARIES = 4;
    uint256 public constant KEY_LENGTH = 32; // X25519 public keys
    uint8 public constant MAX_DEPTH_LIMIT = 20;
    /// @notice The scoring circuit (M1b) has exactly 3 passive slots.
    uint256 public constant CIRCUIT_PASSIVES = 3;
    /// @notice EIP-712 type of a passive's final-round message (D75, D83).
    bytes32 public constant FINAL_MESSAGE_TYPEHASH = keccak256("FinalMessage(address sender,bytes32 root,string marker)");
    bytes32 public constant FINAL_ROUND_MARKER_HASH = keccak256(bytes("final-round"));
    /// @notice M4 (D120/D121): A signs, for each passive, the hash of the list I it sent (gamma_I's encoding).
    bytes32 public constant ALIGNED_SET_TYPEHASH = keccak256("AlignedSet(address passive,bytes32 listHash)");
    uint8 public constant FIGURE_KEY_FULL = 0; // LOO_j: j; BASELINE: 255
    uint8 public constant FIGURE_KEY_BASELINE = 255;
    uint8 public constant FIGURE_KEY_FULL_JOINT = 254; // D139: F_j

    /// @dev M4 (D128): the all-zero-leaf Poseidon tree roots at depth 1..20 (circuits/scripts/empty-roots.mjs), depth 1
    /// first, 32 bytes each. Read only by the constructor, so the table stays out of the runtime code.
    bytes private constant EMPTY_ROOTS =
        hex"2098f5fb9e239eab3ceac3f27b81e481dc3124d55ffed523a839ee8446b648641069673dcdb12263df301a6ff584a7ec261a44cb9dc68df067a4774460b1f1e1"
        hex"18f43331537ee2af2e3d758d50f72106467c6eea50371dd528d57eb2b856d23807f9d837cb17b0d36320ffe93ba52345f1b728571a568265caac97559dbc952a"
        hex"2b94cf5e8746b3f5c9631f4c5df32907a699c58c94b2ad4d7b5cec1639183f552dee93c5a666459646ea7d22cca9e1bcfed71e6951b953611d11dda32ea09d78"
        hex"078295e5a22b84e982cf601eb639597b8b0515a88cb5ac7fa8a4aabe3c87349d2fa5e5f18f6027a6501bec864564472a616b2e274a41211a444cbe3a99f3cc61"
        hex"0e884376d0d8fd21ecb780389e941f66e45e7acce3e228ab3e2156a614fcd7471b7201da72494f1e28717ad1a52eb469f95892f957713533de6175e5da190af2"
        hex"1f8d8822725e36385200c0b201249819a6e6e1e4650808b5bebc6bface7d76362c5d82f66c914bafb9701589ba8cfcfb6162b0a12acf88a8d0879a0471b5f85a"
        hex"14c54148a0940bb820957f5adf3fa1134ef5c4aaa113f4646458f270e0bfbfd0190d33b12f986f961e10c0ee44d8b9af11be25588cad89d416118e4bf4ebe80c"
        hex"22f98aa9ce704152ac17354914ad73ed1167ae6596af510aa5b3649325e06c922a7c7c9b6ce5880b9f6f228d72bf6a575a526f29c66ecceef8b753d38bba7323"
        hex"2e8186e558698ec1c67af9c14d463ffc470043c9c2988b954d75dd643f36b9920f57c5571e9a4eab49e2c8cf050dae948aef6ead647392273546249d1c1ff10f"
        hex"1830ee67b5fb554ad5f63d4388800e1cfe78e310697d46e43c9ce36134f72cca2134e76ac5d21aab186c2be1dd8f84ee880a1e46eaf712f9d371b6df22191f3e";

    address public immutable factory;
    uint256 public immutable creationBlock;
    uint256 public immutable setupDeadline;
    // Copies of proposal fields read on every call (the full proposal is in `_p`).
    address private immutable _active;
    address private immutable _arbiter;
    address private immutable _token;
    uint256 private immutable _n;
    uint256 private immutable _B;
    uint256 private immutable _Dp;
    uint256 private immutable _revealWindow;
    uint256 private immutable _maxSizeI; // 2^maxDepth
    uint256 private immutable _alignmentPeriod;
    uint256 private immutable _deltaR;
    uint256 private immutable _scoringPeriod;
    uint256 private immutable _Dc;
    address private immutable _verifier;
    address private immutable _math; // M4: the factory's SettlementMath
    /// @dev EIP-712 domain separator of this run: name "VFLFederation", version "1", this chain, this contract (the value
    /// OpenZeppelin's EIP712 computed until M4; its ERC-5267 view no longer fits under EIP-170).
    bytes32 private immutable _domainSeparator;
    uint256 private immutable _rho; // M4: tokens per performance unit
    uint256 private immutable _settleWindow; // M4: K * deltaA, the response window after Scored
    /// @notice M4 (D128): the scoring public input for a passive who left: the empty-tree root at depth maxDepth.
    bytes32 public immutable emptyRoot;

    Proposal private _p;
    mapping(address => Member) private _members;

    State public state;
    bool public poolHeld;
    bool public arbiterAccepted;
    bytes public arbiterEncKey;
    uint256 public registeredCount; // passives
    uint256 public committedCount; // A and passives
    uint256 public revealedPassives;

    uint256 public startBlock;
    uint256 public alignmentBlock;
    uint256 public aRevealBlock;

    bytes32 public gammaI;
    uint256 public sizeI;
    bytes32 public gammaA;

    /// @notice The shared seed s, set when the state becomes Seeded.
    uint256 public seed;

    mapping(address => uint256) public owed;
    /// @notice Forfeited deposits no one can claim (every passive missed its reveal window).
    uint256 public lockedForever;

    // M3
    bytes32 public gammaFA; // C_fA: Poseidon commitment of A's bottom model (as S2b/M1b)
    /// @notice D139: C_model of the fusion model trained jointly during split training (gamma_g); F_j is proven
    /// for this model (submitJointScore), never for another one.
    bytes32 public gammaG;
    /// @notice D139: U_fed = max(F_r, F_j), set when both FULL figures are accepted (0 before).
    uint256 public uFed;
    uint256 public trainingEndBlock;
    uint256 public reconciliationEndBlock;
    /// @notice Forfeited participation deposits (D61); where they go is decided in M4.
    uint256 public forfeitedToPool;
    /// @notice Dc collected by fileNonReceipt, held until the objection closes; then paid to the arbiter (D130).
    uint256 public dcEscrow;
    uint256 public openObjections;
    uint256 public leftCount;
    uint256 public requiredFigures; // 3 + remaining passives (F_r, F_j, BASELINE, LOO_j), fixed when scoring starts
    uint256 public acceptedFigures;
    // M4
    /// @notice Block of the last accepted figure; responses are accepted until scoredBlock + K * deltaA.
    uint256 public scoredBlock;
    uint256 public acknowledgedCount;
    /// @notice 0 before settlement; SettlementMath.NORMAL (1), OBJECTION (2) or BACKUP (3) after.
    uint8 public settlementPath;
    mapping(uint8 => Figure) private _figures;

    event PoolHeld(uint256 B);
    event Registered(address indexed passive, uint256 index, bytes channelKey);
    event ArbiterAccepted(address indexed arbiter, bytes encKey);
    event SeedCommitted(address indexed member, bytes32 c);
    event Started(uint256 startBlock);
    event SetupAborted(uint256 atBlock);
    event AlignmentCommitted(bytes32 gammaI, uint256 sizeI, bytes32 gammaA, uint256 alignmentBlock);
    event SeedRevealed(address indexed member, uint256 r);
    event SeedFixed(uint256 s, uint256 sizeI);
    event RevealFailed(bool activeMissed, uint256 passivesRevealed, uint256 locked);
    event AlignmentTimedOut(uint256 deadline);
    event FinalCommitted(address indexed passive, bytes32 root);
    event TrainingEnded(bytes32 gammaFA, bytes32 gammaG, uint256 trainingEndBlock, uint256 left);
    event Left(address indexed passive, uint256 forfeited);
    event EquivocationProven(address indexed passive, bytes32 registered, bytes32 signed, uint256 forfeited);
    event ResendRequested(address indexed passive, uint256 atBlock);
    event NonReceiptFiled(address indexed passive, uint256 dc, uint256 relayDeadline);
    event DeliveryCertified(address indexed passive, bytes32 root);
    event RelayTimedOut(address indexed passive);
    event ReconciliationFinalized(uint256 reconciliationEndBlock, uint256 scoringDeadline, uint256 requiredFigures);
    event ScoreAccepted(Kind kind, uint8 memberIndex, uint256 F, bytes32 cModel, uint256 mask);
    event ScoringCompleted(uint256 atBlock);
    event UFedSet(uint256 Fr, uint256 Fj, uint256 uFed);
    event ScoringTimedOut(uint256 deadline);
    event Acknowledged(address indexed passive);
    event AlignmentObjectionUpheld(address indexed passive, bytes32 listHash);
    event Settled(uint8 path, uint256 E, uint256 estate, uint256[] awards);
    event Credited(address indexed to, uint256 amount);
    event Withdrawn(address indexed to, uint256 amount);

    // Proposal validation (constructor)
    error PassiveCount(uint256 n);
    error InvalidPassive(uint256 index);
    error InvalidArbiter();
    error BadKeyLength(uint256 length);
    error ZeroHash();
    error InvalidMaxDepth(uint8 maxDepth);
    error UnsupportedMetric(uint8 metricId);
    error InvalidClassBoundaries();
    error ZeroRoundsOrDraws();
    error NoCode(address account);
    error ZeroPeriod();
    error ZeroK();
    error ZeroAmount();
    // Calls
    error NotFactory();
    error AlreadyActivated();
    error PoolNotFunded(uint256 balance, uint256 required);
    error WrongState(State state);
    error SetupClosed(uint256 deadline);
    error SetupOpen(uint256 deadline);
    error NotActiveParty();
    error NotPassive();
    error NotArbiter();
    error NotMember();
    error NotRegistered();
    error AlreadyRegistered();
    error AlreadyAccepted();
    error AlreadyCommitted();
    error ArbiterMismatch();
    error ZeroCommitment();
    error ZeroSize();
    error SizeAboveCap(uint256 sizeI, uint256 cap);
    error NotAFieldElement();
    error ActiveNotRevealed();
    error AlreadyRevealed();
    error WindowClosed(uint256 lastBlock);
    error WindowOpen(uint256 lastBlock);
    error BadOpening();
    // M3
    error AlreadyProven();
    error NoEquivocation();
    error SignerMismatch(address recovered);
    error MemberLeft();
    error NoFinalCommitment();
    error AlreadyRequested();
    error NoResendRequest();
    error AlreadyFiled();
    error NoOpenObjection();
    error RootMismatch();
    error ObjectionsOpen(uint256 open);
    error BadFigure();
    error FigureAccepted();
    error InvalidProof();
    error UnsupportedPassiveCount(uint256 n);
    // M4
    error AlreadyAcknowledged();
    error NoMisalignment();

    constructor(Proposal memory p) {
        _validate(p);
        factory = msg.sender;
        creationBlock = block.number;
        setupDeadline = block.number + p.setupPeriod;
        _active = p.activeParty;
        _arbiter = p.arbiter;
        _token = p.token;
        _n = p.passives.length;
        _B = p.B;
        _Dp = p.Dp;
        _revealWindow = p.revealWindow;
        _maxSizeI = uint256(1) << p.maxDepth;
        _alignmentPeriod = p.alignmentPeriod;
        _deltaR = p.deltaR;
        _scoringPeriod = p.scoringPeriod;
        _Dc = p.Dc;
        _verifier = p.verifier;
        _rho = p.rho;
        _domainSeparator = keccak256(
            abi.encode(
                keccak256("EIP712Domain(string name,string version,uint256 chainId,address verifyingContract)"),
                keccak256("VFLFederation"),
                keccak256("1"),
                block.chainid,
                address(this)
            )
        );
        _math = ISettlementSource(msg.sender).settlementMath(); // msg.sender is the creating FederationFactory
        _settleWindow = uint256(p.K) * p.deltaA;
        bytes memory roots = EMPTY_ROOTS;
        bytes32 root;
        uint256 offset = 32 * uint256(p.maxDepth); // _validate: 1 <= maxDepth <= 20; skips the length word
        assembly {
            root := mload(add(roots, offset))
        }
        emptyRoot = root;
        _p = p;
        // Member indices; a second occurrence of an address finds its index already set,
        // so uniqueness costs one warm read per passive (linear in n).
        for (uint256 i; i < p.passives.length; ++i) {
            Member storage m = _members[p.passives[i]];
            if (m.index != 0) revert InvalidPassive(i + 1);
            m.index = uint8(i + 1);
        }
    }

    // ---- Creation --------------------------------------------------------------

    /// @notice Called by the factory in the creation transaction, after it moved B here.
    function activate() external {
        if (msg.sender != factory) revert NotFactory();
        if (poolHeld) revert AlreadyActivated();
        uint256 balance = IERC20(_token).balanceOf(address(this));
        if (balance < _B) revert PoolNotFunded(balance, _B);
        poolHeld = true;
        emit PoolHeld(_B);
    }

    // ---- Setup -----------------------------------------------------------------

    /// @notice A listed passive accepts all terms: stores its channel key and pays Dp.
    function register(bytes calldata channelKey, address arbiterEcho) external {
        _inSetup();
        Member storage m = _members[msg.sender];
        if (m.index == 0) revert NotPassive();
        if (m.registered) revert AlreadyRegistered();
        if (channelKey.length != KEY_LENGTH) revert BadKeyLength(channelKey.length);
        if (arbiterEcho != _arbiter) revert ArbiterMismatch();
        m.registered = true;
        m.channelKey = channelKey;
        ++registeredCount;
        emit Registered(msg.sender, m.index, channelKey);
        _tryStart();
        _collect(msg.sender, _Dp);
    }

    /// @notice The arbiter accepts its role and publishes its encryption key.
    function acceptArbiter(bytes calldata encKey) external {
        _inSetup();
        if (msg.sender != _arbiter) revert NotArbiter();
        if (arbiterAccepted) revert AlreadyAccepted();
        if (encKey.length != KEY_LENGTH) revert BadKeyLength(encKey.length);
        arbiterAccepted = true;
        arbiterEncKey = encKey;
        emit ArbiterAccepted(msg.sender, encKey);
        _tryStart();
    }

    /// @notice A or a registered passive commits to its random number r (see `seedCommitment`).
    function commitSeed(bytes32 c) external {
        _inSetup();
        Member storage m = _members[msg.sender];
        if (msg.sender != _active) {
            if (m.index == 0) revert NotMember();
            if (!m.registered) revert NotRegistered();
        }
        if (m.committed) revert AlreadyCommitted();
        if (c == bytes32(0)) revert ZeroCommitment();
        m.committed = true;
        m.commitment = c;
        ++committedCount;
        emit SeedCommitted(msg.sender, c);
        _tryStart();
    }

    /// @notice After the setup deadline, anyone ends a setup that did not complete.
    /// Nothing is forfeited: A is owed B, every registered passive its Dp.
    function abortSetup() external {
        if (state != State.Setup) revert WrongState(state);
        if (block.number <= setupDeadline) revert SetupOpen(setupDeadline);
        state = State.SetupAborted;
        emit SetupAborted(block.number);
        if (poolHeld) _credit(_active, _B);
        address[] storage ps = _p.passives;
        for (uint256 i; i < _n; ++i) {
            if (_members[ps[i]].registered) _credit(ps[i], _Dp);
        }
    }

    // ---- Alignment (M2) --------------------------------------------------------------

    /// @notice Last block at which A can commit the alignment (0 before the start).
    function alignmentDeadline() public view returns (uint256) {
        return startBlock == 0 ? 0 : startBlock + _alignmentPeriod;
    }

    /// @notice A records the alignment result by alignmentDeadline.
    /// gammaI = keccak256(abi.encodePacked(uint64[] I, ascending)), a plain hash (non-zero).
    /// gammaA = Poseidon Merkle root at depth maxDepth over A's rows in the order of I, a
    /// BN254 field element. sizeI = |I|, capped at 2^maxDepth (D106).
    function commitAlignment(bytes32 gammaI_, uint256 sizeI_, bytes32 gammaA_) external {
        if (msg.sender != _active) revert NotActiveParty();
        if (state != State.Alignment) revert WrongState(state);
        uint256 deadline = startBlock + _alignmentPeriod;
        if (block.number > deadline) revert WindowClosed(deadline);
        if (gammaI_ == bytes32(0)) revert ZeroCommitment();
        if (sizeI_ == 0) revert ZeroSize();
        if (sizeI_ > _maxSizeI) revert SizeAboveCap(sizeI_, _maxSizeI);
        if (uint256(gammaA_) >= BN254_R) revert NotAFieldElement();
        gammaI = gammaI_;
        sizeI = sizeI_;
        gammaA = gammaA_;
        alignmentBlock = block.number;
        state = State.Revealing;
        emit AlignmentCommitted(gammaI_, sizeI_, gammaA_, block.number);
    }

    // ---- Seed reveal ---------------------------------------------------------------

    /// @notice A reveals first, within revealWindow blocks of the alignment; each passive
    /// within revealWindow blocks of A's reveal. The last reveal fixes the seed.
    function revealSeed(uint256 r) external {
        if (state != State.Revealing) revert WrongState(state);
        Member storage m = _members[msg.sender];
        if (msg.sender == _active) {
            if (m.revealed) revert AlreadyRevealed();
            uint256 last = alignmentBlock + _revealWindow;
            if (block.number > last) revert WindowClosed(last);
            _open(m, r);
            aRevealBlock = block.number;
            emit SeedRevealed(msg.sender, r);
            return;
        }
        if (m.index == 0) revert NotMember();
        if (!_members[_active].revealed) revert ActiveNotRevealed();
        uint256 lastP = aRevealBlock + _revealWindow;
        if (block.number > lastP) revert WindowClosed(lastP);
        if (m.revealed) revert AlreadyRevealed();
        _open(m, r);
        emit SeedRevealed(msg.sender, r);
        if (++revealedPassives == _n) _fixSeed();
    }

    /// @notice D107: A did not commit the alignment by alignmentDeadline. Anyone ends the run;
    /// the settlement is exactly the D105 rule for a missed reveal by A.
    function resolveAlignmentTimeout() external {
        if (state != State.Alignment) revert WrongState(state);
        uint256 deadline = startBlock + _alignmentPeriod;
        if (block.number <= deadline) revert WindowOpen(deadline);
        state = State.AlignmentFailed;
        _settleActiveMissedReveal();
        emit AlignmentTimedOut(deadline);
    }

    /// @notice After a reveal window expired without every reveal, anyone ends the run.
    function resolveRevealFailure() external {
        if (state != State.Revealing) revert WrongState(state);
        if (!_members[_active].revealed) {
            uint256 last = alignmentBlock + _revealWindow;
            if (block.number <= last) revert WindowOpen(last);
            state = State.RevealFailed;
            _settleActiveMissedReveal();
            emit RevealFailed(true, 0, 0);
        } else {
            uint256 last = aRevealBlock + _revealWindow;
            if (block.number <= last) revert WindowOpen(last);
            state = State.RevealFailed;
            uint256 locked = _settlePassivesMissedReveal();
            emit RevealFailed(false, revealedPassives, locked);
        }
    }

    // ---- Training (M3 §3) -------------------------------------------------------------

    /// @notice A passive registers gamma_i, the Poseidon Merkle root at depth maxDepth of its
    /// final-round embedding matrix h_i (the root it signs, EIP-712, when sending h_i to A).
    /// Once, in Training, before commitFA. The root must be a non-zero BN254 field element,
    /// since the scoring proofs take it as a public input.
    function commitFinal(bytes32 root) external {
        if (state != State.Training) revert WrongState(state);
        Member storage m = _members[msg.sender];
        if (m.index == 0) revert NotPassive();
        if (m.finalCommitted) revert AlreadyCommitted();
        _checkRoot(root);
        m.finalCommitted = true;
        m.gamma = root;
        emit FinalCommitted(msg.sender, root);
    }

    /// @notice D113 + D139: A registers gamma_fA (= C_fA, the Poseidon commitment of its bottom model) and
    /// gammaG (the Poseidon commitment of the fusion model trained jointly during split training, the model
    /// F_j is proven for), which ends training. Every passive without gamma_i is counted as left (D109).
    function commitFA(bytes32 c, bytes32 gammaG_) external {
        if (msg.sender != _active) revert NotActiveParty();
        if (state != State.Training) revert WrongState(state);
        _checkRoot(c);
        _checkRoot(gammaG_);
        gammaFA = c;
        gammaG = gammaG_;
        trainingEndBlock = block.number;
        state = State.Reconciliation;
        address[] storage ps = _p.passives;
        for (uint256 i; i < _n; ++i) {
            Member storage m = _members[ps[i]];
            if (!m.finalCommitted) _leave(ps[i], m);
        }
        emit TrainingEnded(c, gammaG_, block.number, leftCount);
    }

    // ---- Reconciliation (M3 §4) --------------------------------------------------------

    /// @notice Last block of the reconciliation window (0 before training ends).
    function reconciliationDeadline() public view returns (uint256) {
        return trainingEndBlock == 0 ? 0 : trainingEndBlock + _deltaR;
    }

    /// @notice The member signed `signedRoot` for its final round, but registered another gamma.
    /// Its Dp is forfeited, and the signed root replaces gamma and becomes binding; the member
    /// is still scored.
    function proveEquivocation(address member_, bytes32 signedRoot, bytes calldata signature) external {
        _inReconciliationWindow();
        Member storage m = _remainingPassive(member_);
        if (m.equivocated) revert AlreadyProven();
        if (signedRoot == m.gamma) revert NoEquivocation();
        _checkRoot(signedRoot);
        address signer = ECDSA.recoverCalldata(finalMessageHash(member_, signedRoot), signature);
        if (signer != member_) revert SignerMismatch(signer);
        bytes32 registered = m.gamma;
        m.equivocated = true;
        m.gamma = signedRoot;
        uint256 forfeited = _forfeit(m);
        emit EquivocationProven(member_, registered, signedRoot, forfeited);
    }

    /// @notice D110: A records that it asks `member_` to resend its final message.
    function requestResend(address member_) external {
        if (msg.sender != _active) revert NotActiveParty();
        _inReconciliationWindow();
        Member storage m = _remainingPassive(member_);
        if (m.resendRequestBlock != 0) revert AlreadyRequested();
        m.resendRequestBlock = uint64(block.number);
        emit ResendRequested(member_, block.number);
    }

    /// @notice A objects that it still has not received the final message it asked for.
    /// Collects Dc from A into escrow and opens a relay window of deltaR blocks.
    function fileNonReceipt(address member_) external {
        if (msg.sender != _active) revert NotActiveParty();
        _inReconciliationWindow();
        Member storage m = _remainingPassive(member_);
        if (m.resendRequestBlock == 0) revert NoResendRequest();
        if (m.objection != Objection.None) revert AlreadyFiled();
        m.objection = Objection.Open;
        uint256 deadline = block.number + _deltaR;
        m.relayDeadline = uint64(deadline);
        ++openObjections;
        dcEscrow += _Dc;
        emit NonReceiptFiled(member_, _Dc, deadline);
        _collect(msg.sender, _Dc);
    }

    /// @notice The arbiter certifies that it delivered, through the relay, a final message whose
    /// root equals the member's current gamma. Closes the objection.
    function certifyDelivery(address member_, bytes32 root) external {
        if (msg.sender != _arbiter) revert NotArbiter();
        Member storage m = _members[member_];
        if (m.objection != Objection.Open) revert NoOpenObjection();
        if (block.number > m.relayDeadline) revert WindowClosed(m.relayDeadline);
        if (root != m.gamma) revert RootMismatch();
        m.objection = Objection.Certified;
        --openObjections;
        _payDc();
        emit DeliveryCertified(member_, root);
    }

    /// @notice D114: the relay window passed without a certified delivery. The member is
    /// counted as left. Closes the objection.
    function resolveRelayTimeout(address member_) external {
        Member storage m = _members[member_];
        if (m.objection != Objection.Open) revert NoOpenObjection();
        if (block.number <= m.relayDeadline) revert WindowOpen(m.relayDeadline);
        m.objection = Objection.TimedOut;
        --openObjections;
        _payDc();
        _leave(member_, m);
        emit RelayTimedOut(member_);
    }

    /// @notice After the reconciliation window, with no objection open, anyone starts scoring.
    function finalizeReconciliation() external {
        if (state != State.Reconciliation) revert WrongState(state);
        uint256 last = trainingEndBlock + _deltaR;
        if (block.number <= last) revert WindowOpen(last);
        if (openObjections != 0) revert ObjectionsOpen(openObjections);
        reconciliationEndBlock = block.number;
        requiredFigures = 3 + _n - leftCount;
        state = State.Scoring;
        emit ReconciliationFinalized(block.number, block.number + _scoringPeriod, requiredFigures);
    }

    // ---- Scoring (M3 §5) ---------------------------------------------------------------

    /// @notice Last block at which A can submit a figure (0 before scoring starts).
    function scoringDeadline() public view returns (uint256) {
        return reconciliationEndBlock == 0 ? 0 : reconciliationEndBlock + _scoringPeriod;
    }

    /// @notice D139: A submits F_j, the FULL figure of the fusion model committed at commitFA (gammaG);
    /// no C_model is taken. The contract fills every other public input as in submitScore.
    function submitJointScore(uint256 F, uint256[8] calldata proof) external {
        _acceptFigure(uint8(Kind.FULL_JOINT), 0, F, gammaG, proof);
    }

    /// @notice A submits one figure with its proof and the C_model of its retrained fusion model: FULL
    /// (F_r, memberIndex 0), LOO_j (memberIndex j, a remaining passive) or BASELINE (memberIndex 0). F_j goes
    /// through submitJointScore (D139). The contract fills C_fA, gammaA, the current gamma_1..3, seed, sizeI
    /// and the mask (derived from kind) from its own storage.
    /// proof = [a0, a1, b00, b01, b10, b11, c0, c1] as snarkjs exports it for Solidity.
    function submitScore(uint8 kind, uint8 memberIndex, uint256 F, bytes32 cModel, uint256[8] calldata proof)
        external
    {
        if (kind == uint8(Kind.FULL_JOINT)) revert BadFigure();
        _acceptFigure(kind, memberIndex, F, cModel, proof);
    }

    function _acceptFigure(uint8 kind, uint8 memberIndex, uint256 F, bytes32 cModel, uint256[8] calldata proof)
        private
    {
        if (msg.sender != _active) revert NotActiveParty();
        if (state != State.Scoring) revert WrongState(state);
        uint256 deadline = reconciliationEndBlock + _scoringPeriod;
        if (block.number > deadline) revert WindowClosed(deadline);
        if (_n != CIRCUIT_PASSIVES) revert UnsupportedPassiveCount(_n);
        (uint8 key, uint256 mask) = _figureKeyAndMask(kind, memberIndex);
        Figure storage fig = _figures[key];
        if (fig.accepted) revert FigureAccepted();
        uint256[10] memory pub = publicInputs(mask, F, cModel);
        bool ok = IScoreVerifier(_verifier).verifyProof(
            [proof[0], proof[1]], [[proof[2], proof[3]], [proof[4], proof[5]]], [proof[6], proof[7]], pub
        );
        if (!ok) revert InvalidProof();
        fig.accepted = true;
        fig.F = F;
        fig.cModel = cModel;
        emit ScoreAccepted(Kind(kind), memberIndex, F, cModel, mask);
        if (kind == uint8(Kind.FULL) || kind == uint8(Kind.FULL_JOINT)) {
            Figure storage fr = _figures[FIGURE_KEY_FULL];
            Figure storage fj = _figures[FIGURE_KEY_FULL_JOINT];
            if (fr.accepted && fj.accepted) {
                uFed = fr.F > fj.F ? fr.F : fj.F; // D139: the joint model is a floor under the retrained one
                emit UFedSet(fr.F, fj.F, uFed);
            }
        }
        if (++acceptedFigures == requiredFigures) {
            state = State.Scored;
            scoredBlock = block.number;
            emit ScoringCompleted(block.number);
        }
    }

    /// @notice D115: A did not get every required figure accepted by scoringDeadline. Each
    /// remaining passive is credited its Dp back (unless forfeited) plus an equal share of the
    /// whole pool B, the remainder to the first remaining passive; A gets nothing from B.
    function resolveScoringTimeout() external {
        if (state != State.Scoring) revert WrongState(state);
        uint256 deadline = reconciliationEndBlock + _scoringPeriod;
        if (block.number <= deadline) revert WindowOpen(deadline);
        state = State.ScoringFailed;
        // D115 with the M4 Q3 rule for forfeits, D141 when no passive remains (SettlementMath.timeout)
        _payout(SettlementMath(_math).timeout(_settlementInput(0)));
        emit ScoringTimedOut(deadline);
    }

    // ---- Settlement (M4) -----------------------------------------------------------------

    /// @notice M4: the division after Scored, in proposal order (SettlementMath.division): E = max(0, U_fed - U_A)
    /// with U_A the BASELINE figure; l_i = max(0, U_fed - F_LOO_i), 0 for a passive who left; E' = min(E, floor(B/rho));
    /// a = Talmud(E'; l), floored in performance units (payment rho * a_i).
    /// The return data is SettlementMath's, passed through unchanged.
    function settlementDivision()
        external
        view
        returns (uint256, uint256, uint256[] memory, uint256[] memory) // E, E', l, a
    {
        if (state != State.Scored) revert WrongState(state);
        SettlementMath.Input memory x = _settlementInput(0);
        (bool ok, bytes memory ret) =
            _math.staticcall(abi.encodeCall(SettlementMath.division, (x.B, x.rho, x.uFed, x.uA, x.loo, x.left)));
        assembly ("memory-safe") {
            if iszero(ok) { revert(add(ret, 0x20), mload(ret)) }
            return(add(ret, 0x20), mload(ret))
        }
    }

    /// @notice Last block of the response window (acknowledge, objectAlignment); 0 before Scored.
    function settlementDeadline() public view returns (uint256) {
        return scoredBlock == 0 ? 0 : scoredBlock + _settleWindow;
    }

    /// @notice A remaining passive has no objection. The last acknowledgement settles the run (normal path).
    function acknowledge() external {
        Member storage m = _respondingPassive();
        m.acknowledged = true;
        emit Acknowledged(msg.sender);
        if (++acknowledgedCount == _n - leftCount) _settle(1); // SettlementMath.NORMAL
    }

    /// @notice The 10 public inputs the contract uses for a figure with this mask, F and C_model.
    function publicInputs(uint256 mask, uint256 F, bytes32 cModel) public view returns (uint256[10] memory pub) {
        address[] storage ps = _p.passives;
        pub[0] = uint256(cModel);
        pub[1] = uint256(gammaFA);
        pub[2] = uint256(gammaA);
        for (uint256 i; i < CIRCUIT_PASSIVES; ++i) {
            Member storage m = _members[ps[i]];
            pub[3 + i] = uint256(m.left ? emptyRoot : m.gamma); // D128: a passive who left, committed or not
        }
        pub[6] = seed;
        pub[7] = sizeI;
        pub[8] = mask;
        pub[9] = F;
    }

    /// @notice Bit i-1 set for every passive i that has not left.
    function remainingMask() public view returns (uint256 mask) {
        address[] storage ps = _p.passives;
        for (uint256 i; i < _n; ++i) {
            if (!_members[ps[i]].left) mask |= uint256(1) << i;
        }
    }

    function figure(Kind kind, uint8 memberIndex) external view returns (Figure memory) {
        (uint8 key,) = _figureKeyAndMask(uint8(kind), memberIndex);
        return _figures[key];
    }

    /// @notice EIP-712 digest of a passive's final-round message
    /// FinalMessage(address sender, bytes32 root, string marker = "final-round") for this run.
    function finalMessageHash(address sender, bytes32 root) public view returns (bytes32) {
        return _hashTypedDataV4(keccak256(abi.encode(FINAL_MESSAGE_TYPEHASH, sender, root, FINAL_ROUND_MARKER_HASH)));
    }

    // ---- Withdrawals -------------------------------------------------------------

    /// @notice Pays everything owed to the caller (pull pattern; zeroed before the transfer).
    function withdraw() external {
        uint256 amount = owed[msg.sender];
        if (amount == 0) return;
        owed[msg.sender] = 0;
        emit Withdrawn(msg.sender, amount);
        _pay(msg.sender, amount);
    }

    // ---- Views -------------------------------------------------------------------

    function proposal() external view returns (Proposal memory) {
        return _p;
    }

    function passiveCount() external view returns (uint256) {
        return _n;
    }

    function member(address a) external view returns (MemberView memory) {
        Member storage m = _members[a];
        return MemberView(m.index, m.registered, m.channelKey, m.committed, m.commitment, m.revealed);
    }

    /// @notice A passive's M3 status.
    function finalStatus(address a)
        external
        view
        returns (
            bool finalCommitted,
            bytes32 gamma,
            bool left,
            bool equivocated,
            bool dpForfeited,
            Objection objection,
            uint64 resendRequestBlock,
            uint64 relayDeadline
        )
    {
        Member storage m = _members[a];
        return (
            m.finalCommitted, m.gamma, m.left, m.equivocated, m.dpForfeited, m.objection, m.resendRequestBlock,
            m.relayDeadline
        );
    }

    /// @notice The seed commitment a member computes off-chain: keccak256(abi.encode(r, member, this)).
    function seedCommitment(uint256 r, address member_) public view returns (bytes32) {
        return keccak256(abi.encode(r, member_, address(this)));
    }

    // ---- Internal ----------------------------------------------------------------

    /// @dev Starts the run when the last setup condition is met (no separate start transaction).
    /// The validator floor is checked off-chain (chain/scripts/check-network.mjs).
    function _tryStart() private {
        if (registeredCount == _n && arbiterAccepted && poolHeld && committedCount == _n + 1) {
            state = State.Alignment;
            startBlock = block.number;
            emit Started(block.number);
        }
    }

    function _inSetup() private view {
        if (state != State.Setup) revert WrongState(state);
        if (block.number > setupDeadline) revert SetupClosed(setupDeadline);
    }

    function _open(Member storage m, uint256 r) private {
        if (keccak256(abi.encode(r, msg.sender, address(this))) != m.commitment) revert BadOpening();
        m.revealed = true;
        m.r = r;
    }

    /// @dev s = keccak256(abi.encodePacked(r_A, r_1, ..., r_n)) mod BN254_R, passives in proposal order.
    function _fixSeed() private {
        uint256[] memory rs = new uint256[](_n + 1);
        rs[0] = _members[_active].r;
        address[] storage ps = _p.passives;
        for (uint256 i; i < _n; ++i) rs[i + 1] = _members[ps[i]].r;
        uint256 s = uint256(keccak256(abi.encodePacked(rs))) % BN254_R;
        seed = s;
        state = State.Training;
        emit SeedFixed(s, sizeI);
    }

    /// @dev Case (a), A missed its window (D105, confirmed), and the alignment timeout
    /// (D107, same rule). Each passive gets Dp back plus compensation c from
    /// the pool: c = Dp if B >= n*Dp, else c = floor(B/n) with the integer remainder of
    /// B/n to passive index 1. A is credited what is left of B after the passives are
    /// paid: B - n*c when B >= n*Dp, and 0 otherwise (see QUESTIONS-M1.md, Q2).
    function _settleActiveMissedReveal() private {
        address[] storage ps = _p.passives;
        uint256 c;
        uint256 rem;
        if (_B >= _n * _Dp) {
            c = _Dp;
        } else {
            c = _B / _n;
            rem = _B % _n;
        }
        for (uint256 i; i < _n; ++i) {
            _credit(ps[i], _Dp + c + (i == 0 ? rem : 0));
        }
        _credit(_active, _B - _n * c - rem);
    }

    /// @dev Case (b), A revealed and one or more passives missed their window. A is owed B;
    /// each revealer gets its Dp back plus an equal share of the non-revealers' forfeited
    /// deposits (remainder to the first revealer in proposal order). With no revealer the
    /// forfeited deposits stay locked for good. Returns the locked amount.
    function _settlePassivesMissedReveal() private returns (uint256 locked) {
        _credit(_active, _B);
        uint256 revealers = revealedPassives;
        uint256 forfeited = _Dp * (_n - revealers);
        if (revealers == 0) {
            lockedForever = forfeited;
            return forfeited;
        }
        uint256 share = forfeited / revealers;
        uint256 rem = forfeited % revealers;
        address[] storage ps = _p.passives;
        for (uint256 i; i < _n; ++i) {
            if (!_members[ps[i]].revealed) continue;
            _credit(ps[i], _Dp + share + rem);
            rem = 0;
        }
    }

    /// @notice M4 (D120/D121, no bond): a remaining passive shows A's AlignedSet signature over the hash of the list A sent
    /// it. Upheld iff the signature recovers to A for (caller, listHash) and listHash != gamma_I: the run settles at once
    /// (SettlementMath.OBJECTION). Anything else reverts.
    function objectAlignment(bytes calldata sigA, bytes32 listHash) external {
        _respondingPassive();
        address signer = ECDSA.recoverCalldata(alignedSetHash(msg.sender, listHash), sigA);
        if (signer != _active) revert SignerMismatch(signer);
        if (listHash == gammaI) revert NoMisalignment();
        emit AlignmentObjectionUpheld(msg.sender, listHash);
        _settle(2); // SettlementMath.OBJECTION
    }

    /// @notice M4 backup path: after K * deltaA blocks without every acknowledgement or an upheld objection, anyone ends
    /// the run. D142 (supersedes M4 Q6): it pays exactly as the normal path; silence after the window counts as
    /// acceptance (with no passive remaining this is D141: B back to A, the forfeits locked). Recorded as path 3.
    function settleBackup() external {
        if (state != State.Scored) revert WrongState(state);
        uint256 deadline = scoredBlock + _settleWindow;
        if (block.number <= deadline) revert WindowOpen(deadline);
        _settle(3); // SettlementMath.BACKUP
    }

    /// @notice EIP-712 digest of A's AlignedSet(passive, listHash) for this run.
    function alignedSetHash(address passive, bytes32 listHash) public view returns (bytes32) {
        return _hashTypedDataV4(keccak256(abi.encode(ALIGNED_SET_TYPEHASH, passive, listHash)));
    }

    /// @dev The caller, a remaining passive that has not acknowledged, in the response window after Scored.
    function _respondingPassive() private view returns (Member storage m) {
        if (state != State.Scored) revert WrongState(state);
        uint256 deadline = scoredBlock + _settleWindow;
        if (block.number > deadline) revert WindowClosed(deadline);
        m = _members[msg.sender];
        if (m.index == 0) revert NotPassive();
        if (m.left) revert MemberLeft();
        if (m.acknowledged) revert AlreadyAcknowledged();
    }

    /// @dev The run's figures and flags in proposal order, for SettlementMath.
    function _settlementInput(uint8 path) private view returns (SettlementMath.Input memory x) {
        x.path = path;
        x.B = _B;
        x.rho = _rho;
        x.Dp = _Dp;
        x.uFed = uFed;
        x.uA = _figures[FIGURE_KEY_BASELINE].F;
        x.forfeited = forfeitedToPool;
        x.loo = new uint256[](_n);
        address[] storage ps = _p.passives;
        for (uint256 i; i < _n; ++i) {
            Member storage m = _members[ps[i]];
            if (m.left) x.left |= 1 << i;
            else x.loo[i] = _figures[uint8(i + 1)].F;
            if (!m.left && m.dpForfeited) x.eqv |= 1 << i;
        }
    }

    /// @dev Applies a SettlementMath result: the forfeited deposits are distributed or locked by it.
    function _payout(SettlementMath.Output memory o) private {
        forfeitedToPool = 0;
        address[] storage ps = _p.passives;
        for (uint256 i; i < _n; ++i) _credit(ps[i], o.credits[i]);
        _credit(_active, o.creditA);
        lockedForever += o.locked;
    }

    /// @dev Settles along `path` (SettlementMath.NORMAL / OBJECTION / BACKUP): credits, locks, and ends the run.
    function _settle(uint8 path) private {
        SettlementMath.Output memory o = SettlementMath(_math).settle(_settlementInput(path));
        state = State.Settled;
        settlementPath = path;
        _payout(o);
        emit Settled(path, o.E, o.estate, o.awards);
    }

    /// @dev The EIP-712 digest of a struct hash in this run's domain.
    function _hashTypedDataV4(bytes32 structHash) private view returns (bytes32) {
        return keccak256(abi.encodePacked(hex"1901", _domainSeparator, structHash));
    }

    function _checkRoot(bytes32 v) private pure {
        if (v == bytes32(0)) revert ZeroCommitment();
        if (uint256(v) >= BN254_R) revert NotAFieldElement();
    }

    function _inReconciliationWindow() private view {
        if (state != State.Reconciliation) revert WrongState(state);
        uint256 last = trainingEndBlock + _deltaR;
        if (block.number > last) revert WindowClosed(last);
    }

    /// @dev A passive with a final commitment that has not left.
    function _remainingPassive(address a) private view returns (Member storage m) {
        m = _members[a];
        if (m.index == 0) revert NotPassive();
        if (m.left) revert MemberLeft();
        if (!m.finalCommitted) revert NoFinalCommitment();
    }

    /// @dev M4 (D130): the arbiter is paid D_c when the relay objection closes, certified or timed out; never returned.
    function _payDc() private {
        dcEscrow -= _Dc;
        _credit(_arbiter, _Dc);
    }

    /// @dev Counted as left: share zero, Dp forfeited (D61, D109, D114).
    function _leave(address a, Member storage m) private {
        m.left = true;
        ++leftCount;
        emit Left(a, _forfeit(m));
    }

    function _forfeit(Member storage m) private returns (uint256) {
        if (m.dpForfeited) return 0;
        m.dpForfeited = true;
        forfeitedToPool += _Dp;
        return _Dp;
    }

    function _figureKeyAndMask(uint8 kind, uint8 memberIndex) private view returns (uint8 key, uint256 mask) {
        if (kind == uint8(Kind.FULL)) {
            if (memberIndex != 0) revert BadFigure();
            return (FIGURE_KEY_FULL, remainingMask());
        }
        if (kind == uint8(Kind.BASELINE)) {
            if (memberIndex != 0) revert BadFigure();
            return (FIGURE_KEY_BASELINE, 0);
        }
        if (kind == uint8(Kind.FULL_JOINT)) {
            if (memberIndex != 0) revert BadFigure();
            return (FIGURE_KEY_FULL_JOINT, remainingMask());
        }
        if (kind != uint8(Kind.LOO) || memberIndex == 0 || memberIndex > _n) revert BadFigure();
        if (_members[_p.passives[memberIndex - 1]].left) revert MemberLeft();
        return (memberIndex, remainingMask() & ~(uint256(1) << (memberIndex - 1)));
    }

    function _credit(address to, uint256 amount) private {
        if (amount == 0) return;
        owed[to] += amount;
        emit Credited(to, amount);
    }

    /// @dev Every incoming token movement of this contract. The token type may change later.
    function _collect(address from, uint256 amount) private {
        IERC20(_token).safeTransferFrom(from, address(this), amount);
    }

    /// @dev Every outgoing token movement of this contract.
    function _pay(address to, uint256 amount) private {
        IERC20(_token).safeTransfer(to, amount);
    }

    function _validate(Proposal memory p) private view {
        uint256 n = p.passives.length;
        if (n < MIN_PASSIVES || n > MAX_PASSIVES) revert PassiveCount(n);
        if (p.arbiter == address(0) || p.arbiter == p.activeParty) revert InvalidArbiter();
        for (uint256 i; i < n; ++i) {
            address a = p.passives[i];
            if (a == address(0) || a == p.activeParty || a == p.arbiter) revert InvalidPassive(i + 1);
        } // duplicates: checked where the constructor assigns member indices
        if (p.activeChannelKey.length != KEY_LENGTH) revert BadKeyLength(p.activeChannelKey.length);
        if (p.metricId != METRIC_MACRO_F1) revert UnsupportedMetric(p.metricId);
        if (p.modelConfigHash == bytes32(0) || p.vkHash == bytes32(0)) revert ZeroHash();
        if (p.classBoundaries.length != CLASS_BOUNDARIES) revert InvalidClassBoundaries();
        for (uint256 i = 1; i < CLASS_BOUNDARIES; ++i) {
            if (p.classBoundaries[i] <= p.classBoundaries[i - 1]) revert InvalidClassBoundaries();
        }
        if (p.T == 0 || p.N == 0) revert ZeroRoundsOrDraws();
        if (p.maxDepth < 1 || p.maxDepth > MAX_DEPTH_LIMIT) revert InvalidMaxDepth(p.maxDepth);
        if (p.verifier.code.length == 0) revert NoCode(p.verifier);
        if (p.token.code.length == 0) revert NoCode(p.token);
        if (
            p.setupPeriod == 0 || p.alignmentPeriod == 0 || p.revealWindow == 0 || p.deltaR == 0 || p.deltaA == 0
                || p.scoringPeriod == 0
        ) {
            revert ZeroPeriod();
        }
        if (p.K < 1) revert ZeroK();
        if (p.B == 0 || p.rho == 0 || p.Dp == 0 || p.Dc == 0) revert ZeroAmount();
    }
}
