// M4 final review: the parties' settlement decisions (party-m3.mjs). An upheld objection settles in the block after
// Scored, so a party may first see Settled; a passive who left must not respond (acknowledge reverts MemberLeft).
import { expect } from "chai";

// @ts-ignore -- plain ESM module
import { settlementResponse, settlementStarted } from "../offchain/party-m3.mjs";

const SCORED = 9, SCORING_FAILED = 10, SETTLED = 11;
const aligned = { matches: true, alignSig: "0x", listHash: "0x" };

describe("party settlement decisions (M4)", function () {
  it("W01 the wait after scoring ends at Scored or Settled (A), and also at ScoringFailed (passives)", function () {
    expect(settlementStarted(SCORED)).to.equal(true);
    expect(settlementStarted(SETTLED)).to.equal(true);
    expect(settlementStarted(8)).to.equal(false);
    expect(settlementStarted(SCORING_FAILED)).to.equal(false);
    expect(settlementStarted(SCORING_FAILED, { orFailed: true })).to.equal(true);
    expect(settlementStarted(SETTLED, { orFailed: true })).to.equal(true);
  });

  it("W02 a passive responds only while the run is Scored and it has not left", function () {
    expect(settlementResponse({ state: SCORED, left: false, alignment: aligned })).to.equal(aligned);
    expect(settlementResponse({ state: SCORED, left: false, alignment: undefined })).to.deep.equal({ matches: true });
    expect(settlementResponse({ state: SCORED, left: true, alignment: aligned })).to.equal(null);
    expect(settlementResponse({ state: SETTLED, left: false, alignment: aligned })).to.equal(null);
  });
});
