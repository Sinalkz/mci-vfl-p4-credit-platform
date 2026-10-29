"""M4 settlement rules (M4 settlement spec), an independent reference for Federation.sol's settlement.

Division after Scored: E = max(0, U_fed - U_A); l_i = max(0, U_fed - F_LOO_i), 0 for a passive who left;
E' = min(E, floor(B / rho)); a = talmud(E', l), floored in performance units; payment rho * a_i.

Paths (n passives in proposal order; left[i]: counted as left; eqv[i]: equivocated, D_p forfeited but still scored):
- normal: every remaining passive gets rho*a_i + D_p (none for an equivocator); the forfeited deposits
  (D_p of every left or equivocating passive) are split among remaining non-equivocators; A gets B - rho*sum(a).
- objection (upheld): as normal, but B - rho*sum(a) is split among all remaining passives; A gets 0.
- backup (D142, supersedes M4 Q6): settleBackup after the response window settles exactly as normal; silence counts
  as acceptance (with no remaining passive this is D141: B back to A, the forfeited deposits locked).
- backup-q6: the superseded M4 Q6 rule, kept only to recheck E3's first run on Besu, which settled under it:
  responders = remaining passives who acknowledged; each is paid rho*l_i if sum(rho*l_i) <= B, else rho*a_i with
  a = talmud(floor(B/rho), l of responders); responders get D_p back (not equivocators); the forfeited deposits plus
  every non-responder's D_p (not equivocators: already forfeited) are split among non-equivocating responders; the
  rest of B is locked.
Every equal split gives its integer remainder to the first eligible passive in proposal order; with no eligible
passive the amount is locked.
"""
from __future__ import annotations

from talmud_reference import talmud


def division(B, rho, u_fed, u_a, loo, left):
    E = max(0, u_fed - u_a)
    l = [0 if left[i] else max(0, u_fed - loo[i]) for i in range(len(loo))]
    e_prime = min(E, B // rho)
    return E, e_prime, l, talmud(e_prime, l)


def split(amount, eligible):
    """(shares in proposal order, locked amount)."""
    idx = [i for i, e in enumerate(eligible) if e]
    out = [0] * len(eligible)
    if not idx:
        return out, amount
    share, rem = divmod(amount, len(idx))
    for k, i in enumerate(idx):
        out[i] = share + (rem if k == 0 else 0)
    return out, 0


def settle(path, *, B, rho, Dp, u_fed, u_a, loo, left, eqv, acks=None):
    """Owed amounts {"A", "P": [...]}, the locked amount, and the division, for one settlement path."""
    n = len(loo)
    if path == "backup":
        path = "normal"  # D142 (and D141 when no passive remains)
    E, e_prime, l, a = division(B, rho, u_fed, u_a, loo, left)
    remaining = [not x for x in left]
    forfeited = Dp * sum(1 for i in range(n) if left[i] or eqv[i])
    owed_a, owed_p, locked = 0, [0] * n, 0

    def add(shares):
        for i, s in enumerate(shares):
            owed_p[i] += s

    if path in ("normal", "objection"):
        paid = 0
        for i in range(n):
            if remaining[i]:
                owed_p[i] += rho * a[i] + (0 if eqv[i] else Dp)
                paid += rho * a[i]
        shares, lk = split(forfeited, [remaining[i] and not eqv[i] for i in range(n)])
        add(shares)
        locked += lk
        rest = B - paid
        if path == "normal":
            owed_a += rest
        else:
            shares, lk = split(rest, remaining)
            add(shares)
            locked += lk
        awards = a
    elif path == "backup-q6":
        resp = [remaining[i] and acks[i] for i in range(n)]
        claims = [l[i] if resp[i] else 0 for i in range(n)]
        if rho * sum(claims) <= B:
            awards = claims
        else:
            awards = talmud(B // rho, claims)
        pay = [rho * x for x in awards]
        add(pay)
        for i in range(n):
            if resp[i] and not eqv[i]:
                owed_p[i] += Dp
        pool = forfeited + Dp * sum(1 for i in range(n) if remaining[i] and not resp[i] and not eqv[i])
        shares, lk = split(pool, [resp[i] and not eqv[i] for i in range(n)])
        add(shares)
        locked += lk + (B - sum(pay))
    else:
        raise ValueError(path)
    return {"A": owed_a, "P": owed_p, "locked": locked, "E": E, "Eprime": e_prime, "l": l, "a": a, "awards": awards}
