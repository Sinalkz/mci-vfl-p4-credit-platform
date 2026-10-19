// The thesis's electricity data (RECS) and the configs made from it are not in every checkout of this code. A case
// that reads them skips when they are absent, and its title names the missing files.
import { existsSync } from "node:fs";
import { relative, resolve } from "node:path";

/** The prepared RECS data (data/recs/prep.py). Paths are taken from contracts/, where the suite runs. */
export const RECS_MANIFEST = resolve("..", "data", "recs", "prepared", "manifest.json");

/** Skips the running case when any of `files` is absent, with the reason in its title. */
export function skipWithoutRecs(ctx: Mocha.Context, ...files: string[]): void {
  const missing = files.filter((f) => !existsSync(f)).map((f) => relative(resolve(".."), f).replaceAll("\\", "/"));
  if (missing.length === 0) return;
  ctx.test!.title += ` [skipped: the thesis's RECS data is absent; missing ${missing.join(", ")}]`;
  ctx.skip();
}
