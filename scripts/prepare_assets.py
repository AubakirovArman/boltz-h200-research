"""Prepare public Boltz model assets and the fixed CASP16 input outside timing."""

import argparse
import concurrent.futures
import csv
import hashlib
import io
import json
import tarfile
import time
import urllib.request
from pathlib import Path

import requests
import yaml


ROOT = Path(__file__).resolve().parents[1]
CASP = "https://predictioncenter.org"


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            result.update(chunk)
    return result.hexdigest()


def model_assets():
    api = "https://huggingface.co/api/models/boltz-community/boltz-2"
    revision_file = ROOT / "manifests/model-revision.json"
    revision = (json.loads(revision_file.read_text())["revision"] if revision_file.exists()
                else requests.get(api, timeout=60).json()["sha"])
    (ROOT / "manifests/model-revision.json").write_text(
        json.dumps({"repo": "boltz-community/boltz-2", "revision": revision}, indent=2)
    )

    def fetch(name):
        url = f"https://huggingface.co/boltz-community/boltz-2/resolve/{revision}/{name}"
        target = ROOT / "cache" / name
        started = time.monotonic()
        if not target.exists():
            partial = target.with_suffix(target.suffix + ".partial")
            with urllib.request.urlopen(url, timeout=120) as response, partial.open("wb") as out:
                while chunk := response.read(8 * 1024**2):
                    out.write(chunk)
            partial.rename(target)
        row = {
            "file": str(target.relative_to(ROOT)), "url": url,
            "bytes": target.stat().st_size, "sha256": digest(target),
            "preparation_s": time.monotonic() - started,
        }
        recorded = ROOT / "manifests/model_checksums.json"
        if recorded.exists():
            expected = next((item for item in json.loads(recorded.read_text())
                             if item["file"] == row["file"]), None)
            assert expected is None or expected["sha256"] == row["sha256"], name
        print(json.dumps({"asset_ready": row}), flush=True)
        return row

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        rows = list(executor.map(fetch, ["boltz2_conf.ckpt", "boltz2_aff.ckpt", "mols.tar"]))
    (ROOT / "manifests/model_checksums.json").write_text(json.dumps(rows, indent=2))
    if not (ROOT / "cache/mols").exists():
        with tarfile.open(ROOT / "cache/mols.tar") as archive:
            archive.extractall(ROOT / "cache", filter="data")
    print("All checkpoint and CCD assets prepared.", flush=True)


def inputs():
    # Raw inputs are downloadable from the official CASP source; no uploaded
    # sequences or hand-written ligand structures are needed to reproduce this.
    receipt_file = ROOT / "manifests/downloads-casp16.json"
    recorded = json.loads(receipt_file.read_text()) if receipt_file.exists() else []
    for name in ["L3000.SMILES.tar.gz", "L3000_autotaxin_README.txt", "L3000_exper_affinity.csv"]:
        target = ROOT / "data/raw" / name
        url = CASP + "/download_area/CASP16/targets/pharma_ligands/" + name
        if not target.exists():
            target.write_bytes(urllib.request.urlopen(url, timeout=60).read())
        expected = next((item for item in recorded if Path(item["file"]).name == name), None)
        if expected:
            assert expected["sha256"] == digest(target), "Public input changed: " + name
        else:
            recorded.append({"url": url, "file": str(target.relative_to(ROOT)),
                             "bytes": target.stat().st_size, "sha256": digest(target)})
    receipt_file.write_text(json.dumps(recorded, indent=2))
    sequence_url = CASP + "/casp16/target.cgi?target=L3000&view=sequence"
    fasta = ROOT / "data/raw/L3000.fasta"
    if not fasta.exists():
        fasta.write_bytes(urllib.request.urlopen(sequence_url, timeout=60).read())
    sequence = "".join(s.strip() for s in fasta.read_text().splitlines() if not s.startswith(">"))
    assert len(sequence) == 846
    receipts = []
    with tarfile.open(ROOT / "data/raw/L3000.SMILES.tar.gz") as archive:
        candidates = []
        for member in sorted(archive.getmembers(), key=lambda item: item.name):
            if not member.isfile() or not member.name.endswith(".tsv"):
                continue
            raw = archive.extractfile(member).read()
            rows = list(csv.DictReader(io.StringIO(raw.decode()), delimiter="\t"))
            if len(rows) == 1 and "A" in rows[0]["Task"]:
                candidates.append((Path(member.name).stem, raw))
        # Fixed, deterministic pilot subset of complete single-ligand affinity inputs.
        # Inputs containing incidental ligands are not simplified or partly selected.
        assert candidates[0][0] == "L3001"
        for name, raw in candidates[:4]:
            (ROOT / "data/raw" / f"{name}.tsv").write_bytes(raw)
            rows = list(csv.DictReader(io.StringIO(raw.decode()), delimiter="\t"))
            assert len(rows) == 1 and "A" in rows[0]["Task"], (name, rows)
            payload = {
                "version": 1,
                "sequences": [
                    {"protein": {"id": "A", "sequence": sequence,
                                 "msa": str(ROOT / "data/msa/L3000.a3m")}},
                    {"ligand": {"id": "B", "smiles": rows[0]["SMILES"]}},
                ],
                "properties": [{"affinity": {"binder": "B"}}],
            }
            target = ROOT / "data/inputs" / f"{name}.yaml"
            target.write_text(yaml.safe_dump(payload, sort_keys=False))
            receipts.append({
                "id": name, "source_sequence": sequence_url,
                "source_ligand": CASP + "/download_area/CASP16/targets/pharma_ligands/L3000.SMILES.tar.gz",
                "protein_residues": len(sequence), "ligand_rows": len(rows),
                "yaml_sha256": digest(target), "tsv_sha256": hashlib.sha256(raw).hexdigest(),
                "historical_msa_available": False,
                "scope": "Fresh full-sequence CASP16 input; historical issue MSA/config unavailable.",
            })
    official = ROOT / "upstream/examples/prot_custom_msa.yaml"
    smoke = yaml.safe_load(official.read_text())
    smoke["sequences"][0]["protein"]["msa"] = str(ROOT / "upstream/examples/msa/seq2.a3m")
    (ROOT / "data/inputs/official_smoke.yaml").write_text(yaml.safe_dump(smoke, sort_keys=False))
    official_affinity = yaml.safe_load((ROOT / "upstream/examples/affinity.yaml").read_text())
    official_affinity["sequences"][0]["protein"]["msa"] = str(ROOT / "upstream/examples/msa/seq1.a3m")
    (ROOT / "data/inputs/official_affinity.yaml").write_text(yaml.safe_dump(official_affinity, sort_keys=False))
    fixed_set = ROOT / "data/inputs/fixed_set"
    fixed_set.mkdir(exist_ok=True)
    for name in [*[item["id"] for item in receipts], "official_affinity"]:
        (fixed_set / (name + ".yaml")).write_bytes((ROOT / "data/inputs" / (name + ".yaml")).read_bytes())
    (ROOT / "manifests/input_checksums.json").write_text(json.dumps(receipts, indent=2))
    print(json.dumps({"inputs_ready": [row["id"] for row in receipts]}), flush=True)


def msa():
    started = time.monotonic()
    seq = "".join(s.strip() for s in (ROOT / "data/raw/L3000.fasta").read_text().splitlines()
                  if not s.startswith(">"))
    target = ROOT / "data/msa/L3000.a3m"
    receipt_path = ROOT / "manifests/msa-preparation.json"
    host = "https://api.colabfold.com"
    if target.exists():
        print("Existing frozen L3000 MSA retained.", flush=True)
        return
    session = requests.Session()
    session.headers["User-Agent"] = "boltz"
    response = session.post(host + "/ticket/msa", data={"q": ">101\n" + seq + "\n", "mode": "env"}, timeout=60)
    response.raise_for_status()
    job = response.json()
    receipt = {"host": host, "mode": "env", "sequence_sha256": hashlib.sha256(seq.encode()).hexdigest(),
               "status_history": [job], "started_unix": time.time(), "historical_msa": False}
    receipt_path.write_text(json.dumps(receipt, indent=2))
    print(json.dumps({"msa_job": job}), flush=True)
    while job["status"] in {"PENDING", "RUNNING", "UNKNOWN"}:
        if time.monotonic() - started > 1800:
            raise TimeoutError("MSA queue exceeded 30 minutes; receipt retains the job ID")
        time.sleep(10)
        reply = session.get(host + "/ticket/" + job["id"], timeout=60)
        reply.raise_for_status()
        job = reply.json()
        receipt["status_history"].append(job)
        receipt_path.write_text(json.dumps(receipt, indent=2))
        print(json.dumps({"msa_job": job}), flush=True)
    if job["status"] != "COMPLETE":
        raise RuntimeError("MSA service did not complete: " + json.dumps(job))
    archive_path = ROOT / "data/msa/L3000_colabfold.tar.gz"
    reply = session.get(host + "/result/download/" + job["id"], timeout=120)
    reply.raise_for_status()
    archive_path.write_bytes(reply.content)
    # Match Boltz's own unpaired env-mode assembly, including its repeated query rows.
    assembled = []
    with tarfile.open(archive_path) as archive:
        for name in ["uniref.a3m", "bfd.mgnify30.metaeuk30.smag30.a3m"]:
            block = archive.extractfile(name).read().decode().replace("\x00", "")
            assembled.append(block)
    target.write_text("".join(assembled))
    assert target.read_text().splitlines()[1] == seq
    receipt.update({"completed_s": time.monotonic() - started, "msa_sha256": digest(target),
                    "archive_sha256": digest(archive_path), "rows": sum(s.startswith(">") for s in target.read_text().splitlines())})
    receipt_path.write_text(json.dumps(receipt, indent=2))
    print(json.dumps({"msa_ready": receipt}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["models", "inputs", "msa"])
    action = parser.parse_args().action
    {"models": model_assets, "inputs": inputs, "msa": msa}[action]()
