# Deep Learning for Lung Cancer Staging (CT + Clinical Metadata)

This repository contains the code for our CIS 583 term project on **early vs. advanced lung cancer staging** using thoracic CT scans and basic clinical metadata.

We compare two architectures, trained on the **same patients, same splits, and same features**:

- A **CNN baseline** (`cnn_ct_classifier.py`)
- A **U-Net–based classifier** (`unet_ct_classifier.py`)

Both models are trained on CT slices from the **Lung-PET-CT-Dx** collection from **The Cancer Imaging Archive (TCIA)**, with labels derived from TNM staging fields in the clinical spreadsheet.

---

## Repository Structure

```text
.
├── Lung-PET-CT-Dx-NBIA-Manifest-122220.tcia   # TCIA manifest used with NBIA Data Retriever
├── NBIA Data Retriever-4.4.msi                # NBIA Data Retriever installer (Windows)
├── dcm2niix.exe                               # DICOM → NIfTI converter used by the pipeline
├── prepare_petct_dataset.py                   # Data preparation script (DICOM → NIfTI + CSV)
├── cnn_ct_classifier.py                       # CNN baseline classifier (CT + metadata)
├── unet_ct_classifier.py                      # U-Net classifier (CT + metadata)
├── requiremets.txt                            # Python dependencies (typo: 'requirements')
└── README.md


## Notes
Load TCIA manifest to NBIA Data Retriever to install the dataset 
