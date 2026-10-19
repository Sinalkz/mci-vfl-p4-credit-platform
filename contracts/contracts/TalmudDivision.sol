// SPDX-License-Identifier: MIT
pragma solidity ^0.8.28;

/// @title Talmud division rule (Aumann & Maschler, 1985) in exact integer arithmetic.
/// @notice With C = sum(claims):
///   estate >= C     : award_i = claim_i
///   2*estate <= C   : award_i = min(claim_i/2, lambda)        (CEA on half-claims)
///   otherwise       : award_i = claim_i - min(claim_i/2, mu)  (half-claims + CEL)
/// lambda and mu are found exactly in one pass over the claims, never by numeric
/// iteration: both are "CEA levels" on the half-claims, for amount = estate and
/// amount = C - estate respectively. A level is kept as the fraction
/// 2*level = num/den, so no halves or decimals are ever materialised. Each award
/// is floored.
/// Claims must be passed in non-decreasing order. The order is checked in O(n)
/// instead of sorted on-chain, so the whole division is O(n).
library TalmudLib {
    error ClaimsNotSorted(uint256 index);

    /// @param claims in non-decreasing order; reverts with ClaimsNotSorted(i) at the first i with claims[i] < claims[i-1].
    /// @return awards floored Talmud awards, awards[i] for claims[i].
    function divide(uint256 estate, uint256[] memory claims) internal pure returns (uint256[] memory awards) {
        uint256 n = claims.length;
        awards = new uint256[](n);
        uint256 total;
        for (uint256 i; i < n; ++i) {
            if (i > 0 && claims[i] < claims[i - 1]) revert ClaimsNotSorted(i);
            total += claims[i];
        }

        if (estate >= total) {
            for (uint256 i; i < n; ++i) awards[i] = claims[i];
            return awards;
        }

        bool lowerHalf = 2 * estate <= total;
        (uint256 num, uint256 den) = _ceaLevel(lowerHalf ? estate : total - estate, claims);
        uint256 twoDen = 2 * den;
        for (uint256 i; i < n; ++i) {
            uint256 c = claims[i];
            // min(c/2, level) is c/2 exactly when c <= 2*level = num/den.
            bool halfClaimBinds = c * den <= num;
            if (lowerHalf) {
                // floor(min(c/2, num/(2 den)))
                awards[i] = halfClaimBinds ? c / 2 : num / twoDen;
            } else {
                // floor(c - min(c/2, num/(2 den))) = c - ceil(min(c/2, num/(2 den)))
                awards[i] = c - (halfClaimBinds ? (c + 1) / 2 : (num + twoDen - 1) / twoDen);
            }
        }
    }

    /// @notice Budget-capped division in token units (decision log D56/D65):
    /// estate = min(rho * estatePerf, pool), claim_i = rho * claimsPerf[i].
    /// claimsPerf must be non-decreasing (multiplying by rho keeps the order).
    function divideCapped(uint256 rho, uint256 pool, uint256 estatePerf, uint256[] memory claimsPerf)
        internal
        pure
        returns (uint256[] memory awards, uint256 estate)
    {
        estate = rho * estatePerf;
        if (estate > pool) estate = pool;
        uint256 n = claimsPerf.length;
        uint256[] memory claims = new uint256[](n);
        for (uint256 i; i < n; ++i) claims[i] = rho * claimsPerf[i];
        awards = divide(estate, claims);
    }

    /// @dev CEA level on half-claims: the level L with sum_i min(c_i/2, L) == amount,
    /// returned as 2L = num/den. `s` is sorted (checked by divide). Precondition
    /// 2*amount <= sum(claims), which both callers guarantee, so the loop always
    /// returns by k = n-1.
    function _ceaLevel(uint256 amount, uint256[] memory s) private pure returns (uint256 num, uint256 den) {
        uint256 n = s.length;
        uint256 rest = 2 * amount; // doubled amount not yet given to the k smallest claims
        for (uint256 k; k < n; ++k) {
            den = n - k;
            // Equal split of the rest over the n-k largest claims stays at or
            // below the smallest of their half-claims: that split is the level.
            if (rest <= den * s[k]) return (rest, den);
            rest -= s[k]; // claim k is paid its full half-claim
        }
        assert(false);
    }
}

/// @notice Stateless wrapper exposing the library for tests and off-chain calls.
contract TalmudDivision {
    function divide(uint256 estate, uint256[] calldata claims) external pure returns (uint256[] memory) {
        return TalmudLib.divide(estate, claims);
    }

    function divideCapped(uint256 rho, uint256 pool, uint256 estatePerf, uint256[] calldata claimsPerf)
        external
        pure
        returns (uint256[] memory awards, uint256 estate)
    {
        return TalmudLib.divideCapped(rho, pool, estatePerf, claimsPerf);
    }
}
