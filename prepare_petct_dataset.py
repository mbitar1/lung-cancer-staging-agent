import os
import csv
import re
import glob
import subprocess
import xml.etree.ElementTree as ET  # optional, for later inspection
from pathlib import Path

import pydicom

# ---------- PATHS ----------
BASE_DIR = Path(__file__).resolve().parent

# where the DICOMs live
INPUT_DICOM_ROOT = (
    BASE_DIR
    / "Data"
    / "Lung-PET-CT-Dx"
    / "manifest-1608669183333"
    / "Lung-PET-CT-Dx"
)

# where the XML annotation folders live
XML_DIR = (
    BASE_DIR
    / "Data"
    / "Lung-PET-CT-Dx-Annotations-XML-Files-rev12222020"
    / "Annotation"
)

OUTPUT_NII_DIR = BASE_DIR / "petct_nii"         
CSV_OUT        = BASE_DIR / "lung_petct_dx.csv"  
DCM2NIIX_EXE = r"C:\School\MS\CIS_583\Term_Project_2\dcm2niix.exe"

# -----------------------------------------------------

OUTPUT_NII_DIR.mkdir(parents=True, exist_ok=True)

print("INPUT_DICOM_ROOT:", INPUT_DICOM_ROOT, "exists?", INPUT_DICOM_ROOT.is_dir())
print("XML_DIR         :", XML_DIR, "exists?", XML_DIR.is_dir())
print("OUTPUT_NII_DIR  :", OUTPUT_NII_DIR)


def norm_case_id(name: str) -> str:
    """
    Extract a clean case id. For 'Lung_Dx-A0001', this returns 'A0001'.
    Used just for short filenames.
    """
    m = re.search(r"([A-Za-z]*\d+)", name)
    return m.group(1) if m else name


def run_dcm2niix(series_dir: str | Path, out_dir: Path, out_name: str) -> str:
    """
    Convert a DICOM series folder to NIfTI. Returns path to the .nii.gz.
    """
    series_dir = str(series_dir)
    out_dir = str(out_dir)

    cmd = [
        DCM2NIIX_EXE,
        "-z", "y",             # gzip
        "-f", out_name,        # filename pattern
        "-o", out_dir,         # output dir
        series_dir,
    ]
    subprocess.run(cmd, check=True)

    candidates = glob.glob(os.path.join(out_dir, f"{out_name}*.nii.gz"))
    if not candidates:
        raise FileNotFoundError(f"No NIfTI produced for {series_dir}")
    candidates.sort(key=len)
    return candidates[0]


def find_ct_pet_series(case_dir: Path):
    """
    Walk all subfolders of a case (Lung_Dx-A0001) and look for series dirs.
    Identify modality via DICOM header:
        CT  -> Modality == 'CT'
        PET -> Modality in {'PT', 'PET'}
    """
    ct_dir = None
    pet_dir = None

    for root, dirs, files in os.walk(case_dir):
        dcm_files = [f for f in files if f.lower().endswith(".dcm")]
        if not dcm_files:
            continue

        sample_file = Path(root) / dcm_files[0]
        try:
            ds = pydicom.dcmread(sample_file, stop_before_pixels=True)
            modality = getattr(ds, "Modality", "").upper()
        except Exception:
            continue

        if modality == "CT" and ct_dir is None:
            ct_dir = Path(root)
        elif modality in ("PT", "PET") and pet_dir is None:
            pet_dir = Path(root)

        if ct_dir is not None and pet_dir is not None:
            break

    return ct_dir, pet_dir


def get_case_xml_and_label(case_id: str):
    """
    Given a case_id like 'A0001', look in XML_DIR / case_id.
    If there are any .xml files, treat this as a positive (label=1)
    and return the first xml file path.

    Returns: (xml_path: str, label: int, xml_found: bool)
    """
    case_xml_dir = XML_DIR / case_id  # e.g. .../Annotation/A0001
    if not case_xml_dir.is_dir():
        return "", 0, False

    xml_files = sorted(case_xml_dir.glob("*.xml"))
    if not xml_files:
        return "", 0, False

    return str(xml_files[0].resolve()), 1, True


def main():
    rows = []

    cases = [
        d for d in os.listdir(INPUT_DICOM_ROOT)
        if (INPUT_DICOM_ROOT / d).is_dir()
    ]
    cases.sort()

    print(f"Found {len(cases)} case folders under {INPUT_DICOM_ROOT}")
    for case in cases:
        case_dir = INPUT_DICOM_ROOT / case
        subject_id = case                      # 'Lung_Dx-A0001'
        case_id = norm_case_id(case)           # 'A0001'

        ct_dir, pet_dir = find_ct_pet_series(case_dir)
        if ct_dir is None:
            print(f"  [skip] {case} — no CT series found")
            continue

        if pet_dir is None:
            print(f"  [info] {case} — using CT only (no PET series detected)")

        # --- Convert CT (and PET if present) to NIfTI ---
        try:
            ct_out_name = f"{case_id}_ct"
            ct_nii = run_dcm2niix(ct_dir, OUTPUT_NII_DIR, ct_out_name)
        except subprocess.CalledProcessError as e:
            print(f"  [error] dcm2niix failed for CT in {case}: {e}")
            continue

        pet_nii = ""
        if pet_dir is not None:
            try:
                pet_out_name = f"{case_id}_pet"
                pet_nii = run_dcm2niix(pet_dir, OUTPUT_NII_DIR, pet_out_name)
            except subprocess.CalledProcessError as e:
                print(f"  [warn] dcm2niix failed for PET in {case}: {e}")
                pet_nii = ""

        # --- XML label based on Annotation/<case_id> folder ---
        xml_path, label, xml_found = get_case_xml_and_label(case_id)

        rows.append({
            "subject_id": subject_id,
            "ct_path": str(Path(ct_nii).resolve()),
            "pet_path": str(Path(pet_nii).resolve()) if pet_nii else "",
            "xml_path": xml_path,
            "label": label,
        })

        print(
            f"  [ok] {case} → ct={Path(ct_nii).name}, "
            f"pet={Path(pet_nii).name if pet_nii else 'None'}, "
            f"label={label}, xml_found={xml_found}"
        )

    if not rows:
        print("No rows generated — check your folder structure and paths.")
        return

    with open(CSV_OUT, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["subject_id", "ct_path", "pet_path", "xml_path", "label"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} rows to: {CSV_OUT}")
    print("Example row:")
    print(rows[0])


if __name__ == "__main__":
    main()