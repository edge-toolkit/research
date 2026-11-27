"""Create source or result archives for the scaling benchmark."""

import argparse
import tarfile

from common import ROOT, controller_source, sources, support_source, tool_identity


def main():
    """Keep each source package version separate and exclude environments and credentials."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", action="store_true")
    args = parser.parse_args()
    prefix = "harbour-fl-scaling-" + tool_identity()[:16]
    suffix = "-results" if args.results else ""
    destination = ROOT / "packages" / (prefix + suffix + ".tar.gz")
    destination.parent.mkdir(exist_ok=True)
    if args.results:
        files = [
            p
            for p in (ROOT / "results").rglob("*")
            if p.is_file() and not p.is_symlink() and p.name not in ("inputs.json", "manifests.json", "pods-final.json")
        ]
        if not files:
            raise RuntimeError("No results to package")
        files += [
            ROOT / "README.md",
            ROOT / "settings.json",
            ROOT / "settings-per-client.json",
        ]
    else:
        files = sources()
    with tarfile.open(destination, "w:gz") as archive:
        for path in files:
            archive.add(path, arcname=prefix + "/" + str(path.relative_to(ROOT)), recursive=False)
        if not args.results:
            if support_source() not in files:
                archive.add(support_source(), arcname=prefix + "/study_support.py", recursive=False)
            source = controller_source(ROOT)
            for path in sorted(source.rglob("*.py")):
                archive.add(path, arcname=prefix + "/controller/src/" + str(path.relative_to(source)), recursive=False)
    print(destination)
    print("Extracted directory: " + prefix)


if __name__ == "__main__":
    main()
