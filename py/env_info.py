"""Print hardware and tool versions as a markdown table.

Every report in results/ embeds the output of this script so that measured
numbers can be tied to the machine and toolchain that produced them.

Usage:  .venv/Scripts/python py/env_info.py
"""
from __future__ import annotations

import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str], cwd: Path | None = None, timeout: int = 30) -> str:
    exe = shutil.which(cmd[0])
    if exe is None:
        return "not found"
    try:
        out = subprocess.run([exe, *cmd[1:]], cwd=cwd, capture_output=True,
                             text=True, timeout=timeout)
        text = (out.stdout or out.stderr).strip()
        return text.splitlines()[0] if text else f"exit {out.returncode}"
    except (subprocess.TimeoutExpired, OSError) as exc:
        return f"error: {exc.__class__.__name__}"


def npm_pkg_version(folder: str, pkg: str) -> str:
    p = ROOT / folder / "node_modules" / pkg / "package.json"
    if not p.exists():
        return "not installed"
    return json.loads(p.read_text(encoding="utf-8"))["version"]


def hardware() -> dict[str, str]:
    info = {"OS": f"{platform.system()} {platform.release()} ({platform.version()})"}
    if platform.system() == "Windows":
        ps = ("$c=Get-CimInstance Win32_Processor;"
              "$m=Get-CimInstance Win32_ComputerSystem;"
              "\"$($c.Name.Trim())|$($c.NumberOfCores)|$($c.NumberOfLogicalProcessors)|"
              "$([math]::Round($m.TotalPhysicalMemory/1GB,1))\"")
        raw = run(["powershell", "-NoProfile", "-Command", ps])
        try:
            cpu, cores, threads, ram = raw.split("|")
            info["CPU"] = f"{cpu} ({cores} cores / {threads} threads)"
            info["RAM"] = f"{ram} GB"
        except ValueError:
            info["CPU"] = platform.processor()
    else:
        info["CPU"] = platform.processor()
    return info


def tools() -> dict[str, str]:
    return {
        "Python": sys.version.split()[0],
        "Node.js": run(["node", "--version"]),
        "npm": run(["npm", "--version"]),
        "circom": run(["circom", "--version"]),
        "snarkjs": npm_pkg_version("circuits", "snarkjs"),
        "circomlib": npm_pkg_version("circuits", "circomlib"),
        "Hardhat": npm_pkg_version("contracts", "hardhat"),
        "ethers": npm_pkg_version("contracts", "ethers"),
        "Docker": run(["docker", "--version"]),
        "Docker Compose": run(["docker", "compose", "version"]),
    }


def main() -> None:
    print("| Item | Value |")
    print("|---|---|")
    for k, v in {**hardware(), **tools()}.items():
        print(f"| {k} | {v} |")


if __name__ == "__main__":
    main()
