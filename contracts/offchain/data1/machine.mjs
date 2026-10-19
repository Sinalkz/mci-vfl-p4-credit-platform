// DATA1: the machine block recorded with every measurement (brief rule 4: machine and free RAM).
import { cpus, freemem, hostname, platform, release, totalmem } from "node:os";

export function machine() {
  return {
    cpu: cpus()[0]?.model ?? "unknown", cores: cpus().length, totalRamMB: Math.round(totalmem() / 2 ** 20),
    freeRamMB: Math.round(freemem() / 2 ** 20), node: process.version, platform: `${platform()} ${release()}`, hostname: hostname(),
  };
}

export const peakRssMB = () => Math.round(process.resourceUsage().maxRSS / 1024);
