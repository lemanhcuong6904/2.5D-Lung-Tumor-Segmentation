# NSCLC dataset preprocessing

Both datasets are kept separate under `data` so that a split, PNG pair, or
NIfTI mask can never silently be read from the other cohort.

```
data/
  nsclc-radiomics/
    processed/<LUNG-id>/{images,labels}/
    nifti/<LUNG-id>/{image,mask}.nii.gz
    config/{train,val,test}.txt
    processing_report.csv
    nsclc_radiogenomics/
      processed/<R-id>/{images,labels}/
      nifti/<R-id>/{image,mask}.nii.gz
      intermediate/totalsegmentator/<R-id>/ct_hu_clipped.nii.gz
      intermediate/totalsegmentator/<R-id>/totalsegmentator/*.nii.gz
      config/{train,val,test}.txt
      processing_report.csv
```

Radiogenomics processing clips CT to `[-700, 500]` HU, saves that HU-valued
NIfTI, then runs the `total` TotalSegmentator task. Its predicted lung-lobe
masks define the crop after small connected components are removed. The tumour
label is read only from the native DICOM SEG (or RTSTRUCT fallback), never from
TotalSegmentator. In this collection, SEG labels `Heart`, `Tissue`, and
`Segmentation` are all treated as tumour contours. The final image uses the
same uint8 normalization and `256 x 256` XY resize as Radiomics.

The top-level preprocessing config passes exactly the five lung-lobe ROIs as
`--roi_subset`, so inference is limited to the detailed organs part (Task
291). TotalSegmentator 2.11 still downloads the complete `total` task weight
list before applying that subset; this does not prevent its initial download
of Task 292 and other `total` weights.

Install the required model package once:

```powershell
python -m pip install -r requirements.txt
```

Set `CASE_IDS = ["R01-001"]` and `TOTALSEGMENTATOR_DEVICE = "gpu"` near the
top of `scripts/preprocess_nsclc_radiogenomics.py`, then run one case first
(the first invocation downloads TotalSegmentator model weights):

```powershell
python scripts/preprocess_nsclc_radiogenomics.py
```

If the downloaded data is still in the downloader's `-Fast` directory, set
`RAW_ROOT = Path(r"D:\NSCLC-Radiogenomics-Fast")` in that same config block.

Then restore `CASE_IDS = None`, process every case, and generate its
independent patient-level split:

```powershell
python scripts/preprocess_nsclc_radiogenomics.py
python data/nsclc-radiomics/config/create_stratified_split.py --dataset radiogenomics
```

Radiomics remains an independent run and never invokes TotalSegmentator:

```powershell
python scripts/preprocess_nsclc_radiomics.py
python data/nsclc-radiomics/config/create_stratified_split.py --dataset radiomics
```

To train or infer on Radiogenomics, change the complete root group in the
relevant `CFG` (`PROCESSED_ROOT`, `NIFTI_ROOT`, and split paths) from
`data/nsclc-radiomics` to `data/nsclc-radiomics/nsclc_radiogenomics`; use a
model checkpoint trained on the same dataset.
