// DATA1: A <-> passive Sessions in one process over loopback TCP: the M2 transport, secure channel
// and retransmission layer, with keys from fresh X25519 pairs. The brief allows in-process sessions
// for E1/E2 (no chain, no proofs).
import { Channel, channelKeyPair, deriveChannelKeys } from "../channel.mjs";
import { connect, listen } from "../link.mjs";
import { Session } from "../session.mjs";

/** n A<->passive pairs: [{ a: Session (A's side), p: Session (passive's side) }] and close(). */
export async function sessionPairs(n, { chainId, contract, activeParty, passives }) {
  const accepted = [];
  const server = await listen(0, (link) => accepted.push(link));
  const pairs = [];
  for (let i = 0; i < n; i++) {
    const a = channelKeyPair();
    const p = channelKeyPair();
    const pLink = await connect(server.address().port);
    while (accepted.length < i + 1) await new Promise((ok) => setImmediate(ok));
    const ctx = { contract, chainId, activeParty, passive: passives[i] };
    pairs.push({
      a: new Session(accepted[i], new Channel("A", deriveChannelKeys({ selfSk: a.sk, peerPk: p.pk, ...ctx }))),
      p: new Session(pLink, new Channel("p", deriveChannelKeys({ selfSk: p.sk, peerPk: a.pk, ...ctx }))),
    });
  }
  const close = () => {
    for (const x of pairs) { x.a.link.destroy(); x.p.link.destroy(); }
    server.close();
  };
  return { pairs, close };
}
