// SPDX-License-Identifier: MIT
pragma solidity ^0.8.28;

/// @notice Test double for the M1b verifier (M3 S05-S07): accepts a proof iff its first word
/// is non-zero, whatever the public inputs. Lets tests exercise the figure bookkeeping, masks
/// and settlement without proving. Never used outside tests.
contract MockScoreVerifier {
    function verifyProof(uint256[2] calldata a, uint256[2][2] calldata, uint256[2] calldata, uint256[10] calldata)
        external
        pure
        returns (bool)
    {
        return a[0] != 0;
    }
}
