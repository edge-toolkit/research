#!/usr/bin/env python3
"""Derive the figures and tables reported for the GPU fingerprint from registry.yaml.

Nothing here lists the fingerprinting mechanisms: they are read from the registry, so one added to the page shows
up in the report without a change to this script.

A numeric interface is one whose signatures are filed by mode ("<interface>-split", "<interface>-combined") and
hold answers to the probes in probes.yaml. For each, the report gives how far each signature is from the correctly
rounded answer, how many distinct values each probe took, the number of probes on which each pair of signatures
differs, the smallest sets of probes that tell every signature apart, and the probes on which the two modes
disagree. The per-interface figures use the split-mode signature where there is one and the combined-mode one
otherwise. WebGL 2 and WebGPU are also compared on each GPU measured through both.

Every other interface -- a canvas fingerprint, or any mechanism whose results are not probe answers -- is counted
and its signatures listed, as are the interfaces that browsers were recorded as withholding. Standard library only.

Examples:
    python3 analyse_registry.py            # the text report
    python3 analyse_registry.py --latex    # the same, followed by LaTeX table rows
"""

import argparse
import itertools
import os
import re

from server import read_registry

HERE = os.path.dirname(os.path.abspath(__file__))

# The two modes a numeric signature is taken in; the registry files each under "<interface>-<mode>".
MODES = ("split", "combined")
HEX_WORD = re.compile(r"^[0-9a-f]{8}$")

# How the paper names things. None of the tables below decides what is analysed: an interface without a title is
# reported under its registry name, and a signature without a column name under one of the form "?1".
TITLES = {"webgl2": "WebGL 2", "webgpu": "WebGPU"}

# A short column name for each signature, picked out by a phrase that is unique, within one interface and mode, to
# its registry label. A letter is one GPU; a suffix separates the signatures one GPU gave through different backends
# or browsers.
COLUMNS = {
    "webgl2": [
        ("A", "Apple M5 Pro GPU through Metal on macOS -- WebGL 2"),
        ("B", "Intel Iris Xe (Alder Lake GT2)"),
        ("C", "Mali-G76"),
        ("D", "Intel UHD Graphics 630"),
        ("E,G", "Qualcomm Adreno on Android"),
        ("F", "AMD Radeon Graphics (0x15E7)"),
        ("H", "Adreno 830"),
        ("I", "Intel Xe-LPG through Direct3D 11"),
        ("J.gl", "NVIDIA GB10/PCIe through OpenGL"),
        ("J.vk", "NVIDIA GB10 through Vulkan on Linux -- WebGL 2"),
        ("K", "PowerVR"),
        ("M", "HD Graphics 400"),
        ("N", "llvmpipe"),
    ],
    "webgpu": [
        ("A.cm", "macOS -- WebGPU on Chromium"),
        ("A.sa", "WebGPU on Safari"),
        ("A.fx", "macOS -- WebGPU on Firefox"),
        ("C", "Mali-G76"),
        ("F", "RDNA 2"),
        ("G", "Adreno 730 (Samsung Galaxy S22 Ultra) through Vulkan"),
        ("H", "Adreno 830"),
        ("I", "Intel Xe-LPG on Windows -- WebGPU"),
        ("J", "NVIDIA GB10 through Vulkan on Linux -- WebGPU"),
        ("L", "RDNA 1"),
        ("P", "SwiftShader"),
    ],
}

# The WebGL 2 and WebGPU signatures of one GPU, by column name, for the comparison of the two interfaces.
BOTH_INTERFACES = [
    ("A", "A", "A.cm"),
    ("C", "C", "C"),
    ("F", "F", "F"),
    ("G", "E,G", "G"),
    ("H", "H", "H"),
    ("I", "I", "I"),
    ("J through OpenGL", "J.gl", "J"),
    ("J through Vulkan", "J.vk", "J"),
]

LATEX_LABEL = {
    "multiply_add": "multiply-add",
    "fma": "\\code{fma}",
    "swamped_sum": "swamped sum",
    "divide": "divide",
    "subnormal_product": "subnormal",
    "sqrt": "\\code{sqrt}",
    "inverse_sqrt": "\\code{inverseSqrt}",
    "sin": "\\code{sin}",
    "cos": "\\code{cos}",
    "tan": "\\code{tan}",
    "exp": "\\code{exp}",
    "log": "\\code{log}",
    "pow": "\\code{pow}",
    "atan2": "\\code{atan2}",
    "tanh": "\\code{tanh}",
    "dot": "dot product",
}


def read_probes():
    """Read probes.yaml: each probe's correctly rounded answer, by name, and the names of the choice probes.

    A choice probe has two legal exact answers, so its result is a choice and no distance from the reference is
    reported for it.
    """
    with open(os.path.join(HERE, "probes.yaml"), encoding="utf-8") as handle:
        text = handle.read()
    names = re.findall(r"^  - name: (\w+)$", text, flags=re.MULTILINE)
    block = re.search(r"^reference: \[(.*?)\]", text, flags=re.MULTILINE | re.DOTALL).group(1)
    words = re.findall(r'"([0-9a-f]{8})"', block)
    if len(names) != len(words):
        raise SystemExit(f"probes.yaml names {len(names)} probes but gives {len(words)} reference answers")
    choices = re.search(r"^choices:\n((?:[ #].*\n?|\n)*)", text, flags=re.MULTILINE)
    choice = set(re.findall(r"^  (\w+):$", choices.group(1), flags=re.MULTILINE)) if choices else set()
    return {name: words[index] for index, name in enumerate(names)}, choice


def is_private(entry):
    """Whether a registry row was taken in a private window; a row that does not say was taken in a normal one."""
    return entry.get("window") == "private"


def split_interface(name):
    """Split a registry interface name into (interface, mode); the mode is None for a name that has none."""
    interface, _, mode = name.rpartition("-")
    return (interface, mode) if interface and mode in MODES else (name, None)


def moded_rows(entries, interface):
    """The normal-window signature rows filed for one interface under either mode.

    Private-window rows are left out; compare_windows compares them with these.
    """
    rows = []
    for entry in entries:
        base, mode = split_interface(entry["interface"])
        if base == interface and mode and entry["results"] and not is_private(entry):
            rows.append(entry)
    return rows


def numeric_interfaces(entries, reference):
    """The interfaces whose signatures are filed by mode and hold nothing but answers to the probes.

    Those the paper names come first, in its order, and any others after them by name.
    """
    bases = {split_interface(entry["interface"])[0] for entry in entries if split_interface(entry["interface"])[1]}
    numeric = [b for b in bases if all(set(row["results"]) <= set(reference) for row in moded_rows(entries, b))]
    numeric = [b for b in numeric if moded_rows(entries, b)]
    return [b for b in COLUMNS if b in numeric] + sorted(b for b in numeric if b not in COLUMNS)


def answered_probes(entries, interface, reference):
    """The probes every signature of one interface answers, in probes.yaml order.

    A probe is left out, with a note, when any signature lacks an answer to it: an interface that cannot express
    the probe, or signatures recorded before the probe was added.
    """
    rows = moded_rows(entries, interface)
    probes = []
    for probe in reference:
        missing = sum(not HEX_WORD.match(row["results"].get(probe, "")) for row in rows)
        if missing:
            print(f"note: {interface} leaves out {probe}: {missing} of its {len(rows)} signatures do not answer it")
        else:
            probes.append(probe)
    return probes


def compare_windows(entries):
    """Print, for each private-window row, whether a normal window gave the same result under the same interface."""
    print("\n===== private windows against normal ones =====")
    private = [e for e in entries if is_private(e)]
    if not private:
        print("no private-window rows")
    width = max((len(row["interface"]) for row in private), default=0)
    for row in private:
        if row.get("blocked"):
            print(f"{row['interface']:{width}} withheld: {row.get('reason', '')} ({row['label']})")
            continue
        same = [
            e["label"]
            for e in entries
            if not is_private(e) and e["interface"] == row["interface"] and e["results"] == row["results"]
        ]
        verdict = f"same as a normal window ({same[0]})" if same else "matches no normal-window row"
        print(f"{row['interface']:{width}} {verdict}: {row['label']}")


def summarise(entries, numeric):
    """Print a count for every interface in the registry, then what is not covered by the numeric analysis.

    That is the signatures of each interface that is not numeric, and every row recording an interface withheld.
    """
    print("\n===== every interface in the registry =====")
    names = list(dict.fromkeys(entry["interface"] for entry in entries))
    width = max((len(name) for name in names), default=0)
    for name in names:
        rows = [e for e in entries if e["interface"] == name]
        signed = [e for e in rows if not e.get("blocked")]
        distinct = len({tuple(sorted(e["results"].items())) for e in signed})
        counts = f"{len(signed):3} signature rows ({distinct} distinct), {len(rows) - len(signed)} withheld"
        print(f"{name:{width}} {counts}, {sum(is_private(e) for e in rows)} from private windows")
    for name in names:
        interface, mode = split_interface(name)
        signed = [e for e in entries if e["interface"] == name and not e.get("blocked")]
        if signed and not (mode and interface in numeric):
            print(f"\n===== {name}: signatures, not analysed numerically =====")
            for entry in signed:
                where = "[private window] " if is_private(entry) else ""
                print(f"{'-'.join(entry['results'].values())}  {where}{entry['label']}")
    print("\n===== interfaces recorded as withheld =====")
    for entry in (e for e in entries if e.get("blocked")):
        where = " [private window]" if is_private(entry) else ""
        print(f"{entry['interface']:{width}} {entry['label']}{where}: {entry.get('reason', '')}")


def named_rows(entries, interface, mode):
    """Return {column name: results} for one interface and mode.

    A row whose label matches no phrase in COLUMNS gets a name of the form "?1", numbered by label across both
    modes, so that one device's split and combined signatures share a name when they share a label.
    """
    phrases = COLUMNS.get(interface, [])
    every = moded_rows(entries, interface)
    unnamed = list(dict.fromkeys(e["label"] for e in every if not any(phrase in e["label"] for _, phrase in phrases)))
    rows = [e for e in every if e["interface"] == f"{interface}-{mode}"]
    named = {}
    used = set()
    for name, phrase in phrases:
        hits = [e for e in rows if phrase in e["label"]]
        if len(hits) > 1:
            raise SystemExit(f"{name}: the phrase {phrase!r} matches {len(hits)} {interface}-{mode} labels")
        if hits:
            named[name] = hits[0]["results"]
            used.add(id(hits[0]))
    for entry in (e for e in rows if id(e) not in used):
        name = f"?{unnamed.index(entry['label']) + 1}"
        while name in named:
            name += "'"
        print(f"note: no column name for {interface}-{mode} signature {entry['label']!r}; calling it {name}")
        named[name] = entry["results"]
    return named


def signatures(entries, interface):
    """Return the (column name, results) pairs for one interface: the split-mode signature, else the combined one."""
    split, combined = (named_rows(entries, interface, mode) for mode in MODES)
    known = [name for name, _ in COLUMNS.get(interface, [])]
    order = known + [n for n in list(split) + list(combined) if n.startswith("?")]
    chosen = []
    for name in dict.fromkeys(order):
        if name in split:
            chosen.append((name, split[name]))
        elif name in combined:
            print(f"note: {interface} {name} has no split-mode signature; using its combined-mode one")
            chosen.append((name, combined[name]))
    return chosen


def compare_modes(entries, interface, probes):
    """Print, for each signature taken in both modes, the probes on which split and combined disagree."""
    print(f"\n===== split versus combined, {interface} =====")
    split, combined = (named_rows(entries, interface, mode) for mode in MODES)
    for name in (name for name in split if name in combined):
        differing = [
            f"{p} (split {split[name][p]}, combined {combined[name][p]})"
            for p in probes
            if split[name][p] != combined[name][p]
        ]
        print(f"{name:5} " + (f"differ on {len(differing)}: " + "; ".join(differing) if differing else "identical"))
    for mode, rows, other in (("split", split, combined), ("combined", combined, split)):
        only = [name for name in rows if name not in other]
        print(f"taken in {mode} mode only: {' '.join(only) if only else 'none'}")


def ulp(results, reference, probe):
    """The distance of a result from the reference in units in the last place, as a difference of bit patterns."""
    return int(results[probe], 16) - int(reference[probe], 16)


def compare_pairs(sigs, probes):
    """Print the number of probes on which each pair of signatures differs, and the smallest separating probe sets."""
    print("-- probes on which each pair of signatures differs")
    counts = {}
    for (name1, res1), (name2, res2) in itertools.combinations(sigs, 2):
        counts[(name1, name2)] = sum(res1[p] != res2[p] for p in probes)
    values = list(counts.values())
    print(f"pairs={len(values)} min={min(values)} max={max(values)} mean={sum(values) / len(values):.2f}")
    print("closest pairs:", [pair for pair, count in counts.items() if count == min(values)])
    print("furthest pairs:", [pair for pair, count in counts.items() if count == max(values)])
    names = [name for name, _ in sigs]
    print("      " + " ".join(f"{name:>5}" for name in names[1:]))
    for row, name1 in enumerate(names[:-1]):
        cells = [f"{counts[(name1, name2)]:>5}" if col > row else "     " for col, name2 in enumerate(names) if col]
        print(f"{name1:5} " + " ".join(cells))

    print("-- smallest sets of probes that tell every signature apart")
    for size in range(1, len(probes) + 1):
        sets = [
            chosen
            for chosen in itertools.combinations(probes, size)
            if len({tuple(res[p] for p in chosen) for _, res in sigs}) == len(sigs)
        ]
        if sets:
            print(f"size {size}: {len(sets)} sets, such as {sets[:6]}")
            break
    else:
        print("no set separates them: two signatures are identical")


def report(sigs, probes, reference, choice, title):
    """Print the per-signature, per-probe and pairwise figures for one interface."""
    print(f"\n===== {title}: {len(sigs)} signatures over {len(probes)} probes =====")
    for name, res in sigs:
        deviations = sum(res[p] != reference[p] for p in probes)
        choices = " ".join(f"{p}={res[p]}" for p in probes if p in choice)
        distances = {p: ulp(res, reference, p) for p in probes if p not in choice and res[p] != reference[p]}
        print(f"{name:5} deviations={deviations:2} of {len(probes)}  {choices}  ulp: {distances}")

    print("-- distinct values per probe")
    same = []
    for probe in probes:
        values = sorted({res[probe] for _, res in sigs})
        if len(values) == 1:
            same.append(probe)
        print(f"   {probe:18} {len(values)} {values if len(values) > 1 else ''}")
    print("identical in every signature:", same)

    if len(sigs) > 1:
        compare_pairs(sigs, probes)
    else:
        print("-- a single signature, so there are no pairs to compare")

    functions = [(abs(ulp(res, reference, p)), name, p) for name, res in sigs for p in probes if p not in choice]
    if functions:
        print("largest deviation of a function (ulp, signature, probe):", max(functions))


def latex(sigs, probes, reference, choice, title):
    """Print the rows of the bit-pattern table and of the distance table for one interface."""
    labels = {probe: LATEX_LABEL.get(probe, "\\code{" + probe.replace("_", "\\_") + "}") for probe in probes}
    print(f"\n----- LaTeX bit-pattern rows: {title} ({' '.join(name for name, _ in sigs)}) -----")
    for probe in probes:
        cells = [f"\\code{{{reference[probe]}}}"]
        for _, res in sigs:
            macro = "code" if res[probe] == reference[probe] else "dv"
            cells.append(f"\\{macro}{{{res[probe]}}}")
        print(f"{labels[probe]} & " + " & ".join(cells) + " \\tabularnewline")
    print(f"----- LaTeX distance rows: {title} -----")
    for probe in probes:
        if probe in choice:
            continue
        distances = [ulp(res, reference, probe) for _, res in sigs]
        cells = ["---" if distance == 0 else f"${distance:+d}$" for distance in distances]
        print(f"{labels[probe]} & " + " & ".join(cells) + " \\tabularnewline")
    counts = [str(sum(res[p] != reference[p] for p in probes)) for _, res in sigs]
    print("Not correctly rounded & " + " & ".join(counts) + " \\tabularnewline")


def compare_interfaces(webgl2, webgpu, probes):
    """Print the probes on which a GPU's WebGL 2 and WebGPU signatures differ, for each GPU measured through both."""
    print(f"\n===== the two interfaces on one GPU ({len(probes)} shared probes) =====")
    gl, gpu = dict(webgl2), dict(webgpu)
    for gpu_name, gl_column, gpu_column in BOTH_INTERFACES:
        if gl_column in gl and gpu_column in gpu:
            differing = [p for p in probes if gl[gl_column][p] != gpu[gpu_column][p]]
            print(f"{gpu_name:17} differ on {len(differing)}: {differing}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--latex", action="store_true", help="also print the rows of the LaTeX tables")
    args = parser.parse_args()

    reference, choice = read_probes()
    entries = read_registry()
    numeric = numeric_interfaces(entries, reference)
    probes = {interface: answered_probes(entries, interface, reference) for interface in numeric}
    sigs = {interface: signatures(entries, interface) for interface in numeric}
    titles = {interface: TITLES.get(interface, interface) for interface in numeric}

    for interface in numeric:
        report(sigs[interface], probes[interface], reference, choice, titles[interface])
    if "webgl2" in numeric and "webgpu" in numeric:
        shared = [p for p in probes["webgl2"] if p in probes["webgpu"]]
        compare_interfaces(sigs["webgl2"], sigs["webgpu"], shared)
    for interface in numeric:
        compare_modes(entries, interface, probes[interface])
    summarise(entries, numeric)
    compare_windows(entries)
    if args.latex:
        for interface in numeric:
            latex(sigs[interface], probes[interface], reference, choice, titles[interface])


if __name__ == "__main__":
    main()
