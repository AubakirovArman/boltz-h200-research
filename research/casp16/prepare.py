"""Freeze the public CASP16 affinity inputs used by Boltz issue #636."""

import argparse
import concurrent.futures
import csv
import hashlib
import io
import json
import re
import tarfile
import time
from pathlib import Path

import requests
import yaml
from rdkit import Chem
from rdkit.Chem import Descriptors


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
CASP = "https://predictioncenter.org"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch(url, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        reply = requests.get(url, timeout=120)
        reply.raise_for_status()
        tmp = path.with_name(path.name + ".partial")
        tmp.write_bytes(reply.content)
        tmp.rename(path)
    return {"url": url, "file": str(path.relative_to(ROOT)),
            "bytes": path.stat().st_size, "sha256": digest(path)}


def prepare():
    raw = HERE / "raw"
    jobs = []
    for target, name in [("L1000", "chymase"), ("L3000", "autotaxin")]:
        for suffix in [".SMILES.tar.gz", "_exper_affinity.csv", f"_{name}_README.txt"]:
            filename = target + suffix
            jobs.append((f"{CASP}/download_area/CASP16/targets/pharma_ligands/{filename}", raw / filename))
        jobs.append((f"{CASP}/casp16/target.cgi?target={target}&view=sequence", raw / (target + ".fasta")))
    jobs += [
        (f"{CASP}/download_area/CASP16/targets/pharma_ligands/L1000_exper_affinity_obsolete.csv",
         raw / "L1000_exper_affinity_obsolete.csv"),
        ("https://jeremywohlwend.com/assets/boltz2.pdf", raw / "boltz2.pdf"),
    ]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        receipts = list(pool.map(lambda job: fetch(*job), jobs))
    (HERE / "downloads.json").write_text(json.dumps(receipts, indent=2))
    manifest = []
    for target in ["L1000", "L3000"]:
        seq = "".join(line.strip() for line in (raw / (target + ".fasta")).read_text().splitlines()
                      if not line.startswith(">"))
        assert re.fullmatch("[A-Z]+", seq), "Not a FASTA sequence"
        labels = list(csv.DictReader((raw / (target + "_exper_affinity.csv")).read_text(encoding="utf-8-sig").splitlines()))
        tsv = {}
        with tarfile.open(raw / (target + ".SMILES.tar.gz")) as archive:
            for item in archive.getmembers():
                if item.isfile() and item.name.endswith(".tsv"):
                    tsv[Path(item.name).stem] = list(csv.DictReader(
                        io.StringIO(archive.extractfile(item).read().decode()), delimiter="\t"))
        msa = ROOT / "data/msa" / (target + ".a3m")
        for row in labels:
            row = dict(row)
            # The two official CSVs use different column names; retain the raw files.
            if target == "L1000":
                row["Structure"] = row["ligand_smiles"]
                row["IC50"] = row["CHYMASE_IC50_COLI_NH"]
            name = row["Target ID"]
            molecule = Chem.MolFromSmiles(row["Structure"])
            assert molecule is not None and "A" in row["Task"], row
            assert float(row["IC50"]) > 0
            companions = tsv[name]
            labeled = [item for item in companions if "A" in item["Task"]]
            label_canonical = Chem.MolToSmiles(molecule)
            # Record any curation discrepancies, never silently normalize the input.
            matches = [Chem.MolToSmiles(Chem.MolFromSmiles(item["SMILES"])) == label_canonical
                       for item in labeled]
            payload = {
                "version": 1,
                "sequences": [
                    {"protein": {"id": "A", "sequence": seq, "msa": str(msa)}},
                    {"ligand": {"id": "B", "smiles": row["Structure"]}},
                ],
                "properties": [{"affinity": {"binder": "B"}}],
            }
            folder = HERE / "inputs" / target
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / (name + ".yaml")
            path.write_text(yaml.safe_dump(payload, sort_keys=False))
            manifest.append({
                **row, "target": target, "protein_residues": len(seq),
                "molecular_weight": Descriptors.MolWt(molecule),
                "heavy_atoms": molecule.GetNumHeavyAtoms(),
                "csv_smiles_matches_affinity_tsv": any(matches),
                "official_tsv_rows": companions,
                "input_scope": "Single full-length protein plus the CSV affinity compound, as in issue #636; no extra cofactors.",
                "input": str(path.relative_to(ROOT)), "input_sha256": digest(path),
            })
    assert len(manifest) == 140 and len({item["Target ID"] for item in manifest}) == 140
    assert sum(item["target"] == "L1000" for item in manifest) == 17
    assert sum(item["target"] == "L3000" for item in manifest) == 123
    (HERE / "dataset.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({"examples": len(manifest), "sequence_lengths": {
        target: next(item["protein_residues"] for item in manifest if item["target"] == target)
        for target in ["L1000", "L3000"]},
        "tsv_mismatches": [item["Target ID"] for item in manifest if not item["csv_smiles_matches_affinity_tsv"]],
        "multirow_inputs": [item["Target ID"] for item in manifest if len(item["official_tsv_rows"]) > 1]}, indent=2))


def msa(target):
    path = ROOT / "data/msa" / (target + ".a3m")
    if path.exists():
        print(json.dumps({"msa_retained": str(path), "sha256": digest(path)}), flush=True)
        return
    sequence = "".join(line.strip() for line in (HERE / "raw" / (target + ".fasta")).read_text().splitlines()
                       if not line.startswith(">"))
    receipt_path = HERE / (target + "-msa-job.json")
    session = requests.Session()
    session.headers["User-Agent"] = "boltz"
    host = "https://api.colabfold.com"
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        job = receipt["status_history"][-1]
    else:
        reply = session.post(host + "/ticket/msa", data={"q": ">101\n" + sequence + "\n", "mode": "env"}, timeout=60)
        reply.raise_for_status()
        job = reply.json()
        receipt = {"host": host, "mode": "env", "sequence": sequence,
                   "sequence_sha256": hashlib.sha256(sequence.encode()).hexdigest(), "status_history": [job]}
        receipt_path.write_text(json.dumps(receipt, indent=2))
    started = time.monotonic()
    while job["status"] in {"PENDING", "RUNNING", "UNKNOWN"}:
        print(json.dumps({"msa": target, "job": job}), flush=True)
        if time.monotonic() - started > 1800:
            raise TimeoutError("MSA queue exceeded 30 minutes; saved job can be resumed")
        time.sleep(10)
        reply = session.get(host + "/ticket/" + job["id"], timeout=60)
        reply.raise_for_status()
        job = reply.json()
        receipt["status_history"].append(job)
        receipt_path.write_text(json.dumps(receipt, indent=2))
    assert job["status"] == "COMPLETE", job
    reply = session.get(host + "/result/download/" + job["id"], timeout=120)
    reply.raise_for_status()
    archive = HERE / (target + "-colabfold.tar.gz")
    archive.write_bytes(reply.content)
    with tarfile.open(archive) as tar:
        text = "".join(tar.extractfile(name).read().decode().replace("\x00", "")
                       for name in ["uniref.a3m", "bfd.mgnify30.metaeuk30.smag30.a3m"])
    assert text.splitlines()[1] == sequence
    path.write_text(text)
    receipt.update({"msa_sha256": digest(path), "archive_sha256": digest(archive),
                    "rows": sum(line.startswith(">") for line in text.splitlines())})
    receipt_path.write_text(json.dumps(receipt, indent=2))
    print(json.dumps({"msa_complete": target, "rows": receipt["rows"], "sha256": receipt["msa_sha256"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["prepare", "msa"])
    parser.add_argument("--target", choices=["L1000", "L3000"], default="L1000")
    args = parser.parse_args()
    prepare() if args.action == "prepare" else msa(args.target)
