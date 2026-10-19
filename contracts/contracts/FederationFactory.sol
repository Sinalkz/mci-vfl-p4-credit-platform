// SPDX-License-Identifier: MIT
pragma solidity ^0.8.28;

import {IERC20} from "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import {SafeERC20} from "@openzeppelin/contracts/token/ERC20/utils/SafeERC20.sol";

import {Federation, Proposal} from "./Federation.sol";
import {SettlementMath} from "./SettlementMath.sol";

/// @title Creates a federation run and takes the pool B in the same transaction.
/// @notice A approves this factory for B beforehand. `create` deploys the run (whose
/// constructor validates the proposal), moves B from A into it, and calls `activate`,
/// which checks the run's token balance. All in one transaction, so a run created
/// here always holds its pool.
/// @dev Since M3, Federation's creation code (about 26 KB) no longer fits in a factory's
/// runtime code (EIP-170, 24,576 bytes). The constructor therefore writes it into two
/// code-only contracts (each runtime = a STOP byte + one half), and `create` copies both
/// halves back with EXTCODECOPY. The code comes from this contract's own compilation
/// (type(Federation).creationCode), so nothing outside the factory is trusted;
/// `federationCodeHash` is keccak256 of that creation code, for audit.
contract FederationFactory {
    using SafeERC20 for IERC20;

    address private immutable _half0;
    address private immutable _half1;
    bytes32 public immutable federationCodeHash;
    /// @notice M4: the settlement arithmetic every Federation created here calls (created in this constructor, from
    /// this contract's own compilation, like Federation's code).
    address public immutable settlementMath;

    event FederationCreated(address indexed federation, address indexed activeParty, uint256 B);

    error NotActiveParty();
    error StoreFailed();

    constructor() {
        bytes memory code = type(Federation).creationCode;
        federationCodeHash = keccak256(code);
        uint256 half = code.length / 2;
        _half0 = _store(code, 0, half);
        _half1 = _store(code, half, code.length - half);
        settlementMath = address(new SettlementMath());
    }

    function create(Proposal calldata p) external returns (Federation f) {
        if (p.activeParty != msg.sender) revert NotActiveParty();
        bytes memory init = bytes.concat(_load(_half0), _load(_half1), abi.encode(p));
        address a;
        assembly ("memory-safe") {
            a := create(0, add(init, 0x20), mload(init))
            if iszero(a) {
                // Bubble up the constructor's revert (e.g. Federation's proposal validation).
                returndatacopy(0, 0, returndatasize())
                revert(0, returndatasize())
            }
        }
        f = Federation(a);
        _collect(msg.sender, a, p.B, p.token);
        f.activate();
        emit FederationCreated(a, msg.sender, p.B);
    }

    /// @dev The factory's only token movement: the pool, from A to the new run.
    function _collect(address from, address to, uint256 amount, address token) private {
        IERC20(token).safeTransferFrom(from, to, amount);
    }

    /// @dev Deploys a contract whose runtime code is 0x00 followed by code[from, from + len).
    /// Init code: PUSH2 len+1, DUP1, PUSH1 12, PUSH1 0, CODECOPY, PUSH1 0, RETURN (12 bytes).
    function _store(bytes memory code, uint256 from, uint256 len) private returns (address a) {
        bytes memory chunk = new bytes(len);
        assembly ("memory-safe") {
            mcopy(add(chunk, 0x20), add(add(code, 0x20), from), len)
        }
        bytes memory init = bytes.concat(hex"61", bytes2(uint16(len + 1)), hex"80600c6000396000f300", chunk);
        assembly ("memory-safe") {
            a := create(0, add(init, 0x20), mload(init))
        }
        if (a == address(0)) revert StoreFailed();
    }

    /// @dev The stored half, without its leading STOP byte.
    function _load(address a) private view returns (bytes memory out) {
        uint256 size = a.code.length - 1;
        out = new bytes(size);
        assembly ("memory-safe") {
            extcodecopy(a, add(out, 0x20), 1, size)
        }
    }
}
