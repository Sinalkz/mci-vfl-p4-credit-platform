// SPDX-License-Identifier: MIT
pragma solidity ^0.8.28;

import {TalmudLib} from "./TalmudDivision.sol";

/// @title M4 settlement arithmetic (claude-code-M4-settlement.md), pure; the reference is py/settlement_reference.py.
/// @notice FederationFactory creates one instance in its constructor and every Federation it creates calls it, so this
/// code sits outside Federation's runtime (EIP-170) with the same origin as Federation's own code. Nothing here holds
/// funds or state: Federation passes the run's figures and flags in proposal order and applies the returned credits.
///
/// Division: E = max(0, U_fed - U_A); l_i = max(0, U_fed - F_LOO_i), 0 for a passive who left; E' = min(E, floor(B/rho));
/// a = Talmud(E'; l), floored in performance units; payment rho * a_i.
/// Paths: Normal (1), Objection (2, upheld), Backup (3, D142: pays as Normal); see settle. An equivocator (remaining, D_p forfeited) never gets
/// D_p back and is excluded from every split of forfeited deposits; every equal split gives its integer remainder to the
/// first eligible passive in proposal order, and with no eligible passive the amount is locked.
contract SettlementMath {
    uint8 public constant NORMAL = 1;
    uint8 public constant OBJECTION = 2;
    uint8 public constant BACKUP = 3;

    struct Input {
        uint8 path;
        uint256 B;
        uint256 rho;
        uint256 Dp;
        uint256 uFed;
        uint256 uA;
        uint256 forfeited; // forfeitedToPool: D_p of every passive who left or equivocated
        uint256[] loo; // F_LOO_i (ignored for a passive who left)
        uint256 left; // bit i: passive i+1 left
        uint256 eqv; // bit i: passive i+1 remains with its D_p forfeited (equivocated)
    }

    struct Output {
        uint256 E;
        uint256 estate; // E' (Normal, Objection) or the estate the backup awards were divided from
        uint256[] awards; // performance units, proposal order
        uint256[] credits; // tokens per passive
        uint256 creditA;
        uint256 locked;
    }

    error BadPath(uint8 path);

    function division(uint256 B, uint256 rho, uint256 uFed, uint256 uA, uint256[] memory loo, uint256 left)
        public
        pure
        returns (uint256 E, uint256 Eprime, uint256[] memory l, uint256[] memory a)
    {
        E = uFed > uA ? uFed - uA : 0;
        uint256 n = loo.length;
        l = new uint256[](n);
        for (uint256 i; i < n; ++i) {
            if (!_bit(left, i) && uFed > loo[i]) l[i] = uFed - loo[i];
        }
        uint256 cap = B / rho;
        Eprime = E < cap ? E : cap;
        a = talmudInOrder(Eprime, l);
    }

    /// @notice Normal: each remaining passive gets rho*a_i + D_p (not an equivocator); the forfeited deposits go to the
    /// remaining non-equivocators; A gets B - rho*sum(a). Objection: as Normal, but B - rho*sum(a) is split among all
    /// remaining passives and A gets 0. Backup (D142, supersedes M4 Q6): exactly as Normal; silence after the response
    /// window counts as acceptance, so no one forfeits for not responding and nothing is locked for it.
    function settle(Input memory x) external pure returns (Output memory o) {
        uint256 n = x.loo.length;
        o.credits = new uint256[](n);
        bool[] memory eligible = new bool[](n);
        uint256 paid;
        if (x.path == NORMAL || x.path == OBJECTION || x.path == BACKUP) {
            uint256[] memory l;
            (o.E, o.estate, l, o.awards) = division(x.B, x.rho, x.uFed, x.uA, x.loo, x.left);
            bool[] memory remaining = new bool[](n);
            for (uint256 i; i < n; ++i) {
                if (_bit(x.left, i)) continue;
                remaining[i] = true;
                eligible[i] = !_bit(x.eqv, i);
                uint256 award = x.rho * o.awards[i];
                paid += award;
                o.credits[i] = award + (eligible[i] ? x.Dp : 0);
            }
            o.locked = _split(o.credits, x.forfeited, eligible);
            if (x.path == OBJECTION) o.locked += _split(o.credits, x.B - paid, remaining);
            else o.creditA = x.B - paid;
        } else {
            revert BadPath(x.path);
        }
    }

    /// @notice Talmud awards in the claims' own order: TalmudLib takes non-decreasing claims, so they are sorted here
    /// (insertion sort, n <= 10). A zero claim gets a zero award and does not change the others.
    function talmudInOrder(uint256 estate, uint256[] memory claims) public pure returns (uint256[] memory awards) {
        uint256 n = claims.length;
        uint256[] memory order = new uint256[](n);
        uint256[] memory sorted = new uint256[](n);
        for (uint256 i; i < n; ++i) {
            uint256 j = i;
            while (j > 0 && sorted[j - 1] > claims[i]) {
                sorted[j] = sorted[j - 1];
                order[j] = order[j - 1];
                --j;
            }
            sorted[j] = claims[i];
            order[j] = i;
        }
        uint256[] memory s = TalmudLib.divide(estate, sorted);
        awards = new uint256[](n);
        for (uint256 k; k < n; ++k) awards[order[k]] = s[k];
    }

    function _bit(uint256 mask, uint256 i) private pure returns (bool) {
        return (mask >> i) & 1 == 1;
    }

    /// @dev Adds equal shares of `amount` to credits[i] for every eligible i (remainder to the first); returns what
    /// could not be given (all of it when no one is eligible).
    /// @notice Scoring timeout (D115): each remaining passive gets an equal share of B and its D_p back unless it
    /// equivocated; the forfeited deposits go to the remaining non-equivocators, locked if there is none (M4 Q3 rule).
    /// D141: with no remaining passive, B goes to A and the forfeited deposits are locked. Uses B, Dp, forfeited,
    /// left, eqv and the length of loo.
    function timeout(Input memory x) external pure returns (Output memory o) {
        uint256 n = x.loo.length;
        o.credits = new uint256[](n);
        bool[] memory remaining = new bool[](n);
        bool[] memory eligible = new bool[](n);
        bool any;
        for (uint256 i; i < n; ++i) {
            remaining[i] = !_bit(x.left, i);
            eligible[i] = remaining[i] && !_bit(x.eqv, i);
            if (eligible[i]) o.credits[i] = x.Dp;
            if (remaining[i]) any = true;
        }
        if (!any) {
            o.creditA = x.B;
            o.locked = x.forfeited;
            return o;
        }
        _split(o.credits, x.B, remaining);
        o.locked = _split(o.credits, x.forfeited, eligible);
    }

    function _split(uint256[] memory credits, uint256 amount, bool[] memory eligible) private pure returns (uint256) {
        if (amount == 0) return 0;
        uint256 count;
        for (uint256 i; i < eligible.length; ++i) if (eligible[i]) ++count;
        if (count == 0) return amount;
        uint256 share = amount / count;
        uint256 rem = amount % count;
        for (uint256 i; i < eligible.length; ++i) {
            if (!eligible[i]) continue;
            credits[i] += share + rem;
            rem = 0;
        }
        return 0;
    }
}
