from pathlib import Path
import shutil
import sys
import time

import pandas as pd
import pydicom
from idc_index import IDCClient

# ============================================================
# CONFIG
# ============================================================

COLLECTION = "nsclc_radiogenomics"

OUTPUT_ROOT = Path(r"D:\NSCLC-Radiogenomics-Fast")

PAIRS_CSV = OUTPUT_ROOT / "pairs.csv"

EXPECTED_PATIENTS = 144

# ------------------------------------------------------------
# Download strategy
# ------------------------------------------------------------

# GCS trước vì AWS của bạn vừa gặp:
# "connection was forcibly closed by remote host"
#
# Nếu GCS vẫn lỗi, script tự chuyển sang AWS.
DOWNLOAD_PROVIDERS = [
    "gcs",
    "aws",
]

# Không nên để 144 series trong một batch.
# 8 tương đối cân bằng giữa tốc độ và ổn định.
BATCH_SIZE = 8

# Mỗi provider retry tối đa 3 vòng.
MAX_RETRIES_PER_PROVIDER = 3

# Nghỉ giữa các vòng retry.
RETRY_WAIT_SECONDS = 5

# Hiển thị progress.
SHOW_PROGRESS = True

# ------------------------------------------------------------
# Nếu muốn test một bệnh nhân:
#
# TEST_PATIENT = "R01-001"
#
# Chạy toàn bộ:
# ------------------------------------------------------------

TEST_PATIENT = None

# Nếu pairs.csv đã có 144/144 thành công thì dùng lại ngay,
# không parse SEG lại từ đầu.
REUSE_VALID_PAIRS_CSV = True


# ============================================================
# UTILITIES
# ============================================================


def chunks(items, size):
    """Split list into small batches."""
    for i in range(0, len(items), size):
        yield items[i : i + size]


def safe_int(value, default=0):
    try:
        return int(float(value))
    except Exception:
        return default


def safe_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return default


def free_disk_gb(path):
    usage = shutil.disk_usage(path)
    return usage.free / (1024**3)


# ============================================================
# INITIALIZE IDC
# ============================================================

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)

print("=" * 80)
print("Initializing IDC index...")
print("=" * 80)

client = IDCClient.client()

print(f"IDC data version: " f"{client.get_idc_version()}")

print(f"Output directory: {OUTPUT_ROOT}")

print(f"Free disk space : " f"{free_disk_gb(OUTPUT_ROOT):.2f} GB")


# ============================================================
# LOCAL SEG CHECK
# ============================================================


def find_local_seg_uid(patient_id, expected_uid):
    """
    Return SEG file path if the expected SEG SeriesInstanceUID
    already exists locally.
    """

    seg_dir = OUTPUT_ROOT / patient_id / "SEG"

    if not seg_dir.exists():
        return None

    for path in seg_dir.glob("*.dcm"):

        try:
            ds = pydicom.dcmread(
                path,
                stop_before_pixels=True,
                force=True,
            )
        except Exception:
            continue

        if getattr(ds, "Modality", "") != "SEG":
            continue

        uid = str(
            getattr(
                ds,
                "SeriesInstanceUID",
                "",
            )
        )

        if uid == expected_uid:
            return path

    return None


# ============================================================
# LOCAL CT CHECK
# ============================================================


def ct_complete(record):
    """
    Check whether a patient's selected CT appears complete.

    Conditions:
    - CT directory exists
    - number of local .dcm files >= expected instanceCount
    - one sample DICOM belongs to expected CT SeriesInstanceUID
    """

    patient_id = str(record["PatientID"])
    ct_uid = str(record["CTSeriesUID"])

    expected_count = safe_int(record["CTImages"])

    ct_dir = OUTPUT_ROOT / patient_id / "CT"

    if not ct_dir.exists():
        return False

    files = list(ct_dir.glob("*.dcm"))

    if expected_count > 0:
        if len(files) < expected_count:
            return False

    if len(files) == 0:
        return False

    # Verify that downloaded files belong to the expected series.
    try:
        ds = pydicom.dcmread(
            files[0],
            stop_before_pixels=True,
            force=True,
        )

        local_uid = str(
            getattr(
                ds,
                "SeriesInstanceUID",
                "",
            )
        )

        if local_uid != ct_uid:
            return False

    except Exception:
        return False

    return True


# ============================================================
# DOWNLOAD WRAPPER
# ============================================================


def download_series(
    series_uids,
    modality_name,
    validator,
):
    """
    Robust download:
      - skips already complete series
      - batches UIDs
      - GCS first
      - retry
      - AWS fallback
      - validates after every batch

    validator(uid) -> True/False
    """

    # Remove duplicates while preserving order.
    series_uids = list(dict.fromkeys(str(uid) for uid in series_uids))

    pending = [uid for uid in series_uids if not validator(uid)]

    already_complete = len(series_uids) - len(pending)

    print("\n" + "=" * 80)
    print(f"{modality_name} DOWNLOAD STATUS")
    print("=" * 80)

    print(f"Total series    : " f"{len(series_uids)}")

    print(f"Already complete: " f"{already_complete}")

    print(f"Need download   : " f"{len(pending)}")

    if not pending:
        print(f"All {modality_name} series " f"are already complete.")
        return []

    # --------------------------------------------------------
    # Try providers in order
    # --------------------------------------------------------

    for provider in DOWNLOAD_PROVIDERS:

        if not pending:
            break

        print("\n" + "#" * 80)
        print(f"Using provider: " f"{provider.upper()}")
        print("#" * 80)

        for attempt in range(
            1,
            MAX_RETRIES_PER_PROVIDER + 1,
        ):

            if not pending:
                break

            print("\n" + "-" * 80)
            print(f"{modality_name} retry " f"{attempt}/" f"{MAX_RETRIES_PER_PROVIDER}")
            print(f"Pending series: " f"{len(pending)}")
            print("-" * 80)

            next_pending = []

            batches = list(
                chunks(
                    pending,
                    BATCH_SIZE,
                )
            )

            for batch_index, batch in enumerate(
                batches,
                start=1,
            ):

                print(
                    f"\n[{modality_name}] "
                    f"Batch "
                    f"{batch_index}/"
                    f"{len(batches)} "
                    f"- {len(batch)} series"
                )

                try:

                    client.download_from_selection(
                        downloadDir=str(OUTPUT_ROOT),
                        seriesInstanceUID=batch,
                        # Final folder layout:
                        #
                        # R01-001/
                        #   CT/
                        #   SEG/
                        #
                        dirTemplate=("%PatientID/%Modality"),
                        source_bucket_location=(provider),
                        # Important:
                        # resume partially downloaded
                        # content instead of restarting.
                        use_s5cmd_sync=True,
                        quiet=False,
                        show_progress_bar=(SHOW_PROGRESS),
                    )

                except Exception as e:

                    # Do not immediately fail.
                    # Validation below determines
                    # what is actually missing.
                    print("\nDOWNLOAD EXCEPTION:")
                    print(e)

                # --------------------------------------------
                # Verify batch after download
                # --------------------------------------------

                incomplete = []

                for uid in batch:

                    if not validator(uid):
                        incomplete.append(uid)

                successful = len(batch) - len(incomplete)

                print(f"\nBatch result: " f"{successful}/" f"{len(batch)} complete")

                if incomplete:

                    print(f"Incomplete: " f"{len(incomplete)}")

                    next_pending.extend(incomplete)

            # Remove duplicate pending UID.
            pending = list(dict.fromkeys(next_pending))

            if not pending:

                print(
                    f"\nAll {modality_name} "
                    f"series completed using "
                    f"{provider.upper()}."
                )

                return []

            print(f"\nStill incomplete after " f"attempt {attempt}: " f"{len(pending)}")

            if attempt < MAX_RETRIES_PER_PROVIDER:

                wait = RETRY_WAIT_SECONDS * attempt

                print(f"Waiting {wait}s " f"before retry...")

                time.sleep(wait)

        # End attempts.
        if pending:

            print(
                f"\n{len(pending)} "
                f"{modality_name} series "
                f"still incomplete with "
                f"{provider.upper()}."
            )

            print("Trying next provider...")

    # --------------------------------------------------------
    # Failed after all providers
    # --------------------------------------------------------

    if pending:

        print("\n" + "!" * 80)
        print(f"FAILED TO COMPLETE " f"{len(pending)} " f"{modality_name} SERIES")
        print("!" * 80)

        return pending

    return []


# ============================================================
# DICOM SEG REFERENCE PARSER
# ============================================================


def find_referenced_series(ds):
    """
    Recursively find all SeriesInstanceUIDs inside
    ReferencedSeriesSequence.

    DICOM SEG references the source CT series here.
    """

    refs = set()

    def walk(dataset):

        for elem in dataset:

            if elem.keyword == "ReferencedSeriesSequence":

                for item in elem.value:

                    uid = getattr(
                        item,
                        "SeriesInstanceUID",
                        None,
                    )

                    if uid:
                        refs.add(str(uid))

            if elem.VR == "SQ":

                for item in elem.value:
                    walk(item)

    walk(ds)

    return sorted(refs)


# ============================================================
# PAIRS.CSV VALIDATION
# ============================================================

REQUIRED_PAIR_COLUMNS = {
    "PatientID",
    "SEGSeriesUID",
    "CTSeriesUID",
    "CTSeriesDescription",
    "CTImages",
    "CTSizeMB",
    "Method",
    "Status",
}


def load_existing_pairs():
    """
    Reuse the 144/144 mapping generated during the previous run
    if it is valid.
    """

    if not REUSE_VALID_PAIRS_CSV:
        return None

    if not PAIRS_CSV.exists():
        return None

    try:
        df = pd.read_csv(
            PAIRS_CSV,
            dtype={
                "PatientID": str,
                "SEGSeriesUID": str,
                "CTSeriesUID": str,
            },
        )
    except Exception:
        return None

    if not REQUIRED_PAIR_COLUMNS.issubset(set(df.columns)):
        return None

    if TEST_PATIENT is not None:

        df = df[df["PatientID"] == TEST_PATIENT].copy()

        if len(df) != 1:
            return None

    else:

        if df["PatientID"].nunique() != EXPECTED_PATIENTS:
            return None

    if not (df["Status"] == "OK").all():
        return None

    if df["CTSeriesUID"].isna().any():
        return None

    if df["SEGSeriesUID"].isna().any():
        return None

    return df.reset_index(drop=True)


# ============================================================
# BUILD PAIRS FROM SCRATCH
# ============================================================


def build_pairs():
    """
    Used only if pairs.csv does not exist or is invalid.

    Workflow:
      IDC metadata
       -> download SEG
       -> parse referenced CT UID
       -> save pairs.csv
    """

    # --------------------------------------------------------
    # Query SEG
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("STEP 1 - Query original SEG series")
    print("=" * 80)

    seg_df = client.sql_query(f"""
        SELECT
            PatientID,
            StudyInstanceUID,
            SeriesInstanceUID,
            SeriesDescription,
            series_size_MB,
            instanceCount
        FROM index
        WHERE collection_id = '{COLLECTION}'
          AND Modality = 'SEG'
          AND analysis_result_id IS NULL
        ORDER BY PatientID
        """)

    seg_df = seg_df.drop_duplicates(subset=["SeriesInstanceUID"]).reset_index(drop=True)

    if TEST_PATIENT is not None:

        seg_df = seg_df[seg_df["PatientID"] == TEST_PATIENT].reset_index(drop=True)

    print(f"SEG series: " f"{len(seg_df)}")

    print(f"Patients  : " f"{seg_df['PatientID'].nunique()}")

    if TEST_PATIENT is None:

        if seg_df["PatientID"].nunique() != EXPECTED_PATIENTS:

            raise RuntimeError(
                "Expected 144 patients "
                "with original SEG, but "
                f"found "
                f"{seg_df['PatientID'].nunique()}."
            )

    # --------------------------------------------------------
    # SEG UID -> patient mapping
    # --------------------------------------------------------

    seg_uid_to_patient = {
        str(row["SeriesInstanceUID"]): str(row["PatientID"])
        for _, row in seg_df.iterrows()
    }

    def seg_validator(uid):

        patient = seg_uid_to_patient[uid]

        return (
            find_local_seg_uid(
                patient,
                uid,
            )
            is not None
        )

    # --------------------------------------------------------
    # Download only missing SEG
    # --------------------------------------------------------

    seg_uids = seg_df["SeriesInstanceUID"].astype(str).tolist()

    failed_seg = download_series(
        seg_uids,
        "SEG",
        seg_validator,
    )

    if failed_seg:

        raise RuntimeError(f"{len(failed_seg)} SEG " f"series could not be downloaded.")

    # --------------------------------------------------------
    # Query original CT metadata
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("STEP 2 - Query original CT metadata")
    print("=" * 80)

    ct_df = client.sql_query(f"""
        SELECT
            PatientID,
            StudyInstanceUID,
            SeriesInstanceUID,
            SeriesDescription,
            series_size_MB,
            instanceCount
        FROM index
        WHERE collection_id = '{COLLECTION}'
          AND Modality = 'CT'
          AND analysis_result_id IS NULL
        """)

    ct_df = ct_df.drop_duplicates(subset=["SeriesInstanceUID"]).reset_index(drop=True)

    print(f"Original CT series available: " f"{len(ct_df)}")

    ct_uid_set = set(ct_df["SeriesInstanceUID"].astype(str))

    # --------------------------------------------------------
    # Resolve each SEG -> CT
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("STEP 3 - Resolve SEG -> CT")
    print("=" * 80)

    pairs = []

    for index, seg_row in seg_df.iterrows():

        patient_id = str(seg_row["PatientID"])

        seg_uid = str(seg_row["SeriesInstanceUID"])

        study_uid = str(seg_row["StudyInstanceUID"])

        seg_path = find_local_seg_uid(
            patient_id,
            seg_uid,
        )

        print(
            f"[{index + 1}/" f"{len(seg_df)}] " f"{patient_id}",
            end="",
        )

        if seg_path is None:

            print(" -> SEG NOT FOUND")

            pairs.append(
                {
                    "PatientID": patient_id,
                    "SEGSeriesUID": seg_uid,
                    "CTSeriesUID": "",
                    "CTSeriesDescription": "",
                    "CTImages": "",
                    "CTSizeMB": "",
                    "Method": "",
                    "Status": "SEG_NOT_FOUND",
                }
            )

            continue

        ds = pydicom.dcmread(
            seg_path,
            stop_before_pixels=True,
            force=True,
        )

        refs = find_referenced_series(ds)

        ct_refs = [uid for uid in refs if uid in ct_uid_set]

        selected = None
        method = ""

        # ----------------------------------------------------
        # Preferred:
        # explicit DICOM SEG reference
        # ----------------------------------------------------

        if len(ct_refs) == 1:

            matches = ct_df[ct_df["SeriesInstanceUID"].astype(str) == ct_refs[0]]

            if len(matches) == 1:

                selected = matches.iloc[0]

                method = "ReferencedSeriesSequence"

        # ----------------------------------------------------
        # Fallback:
        # same patient + same study
        # only when exactly one CT exists.
        # ----------------------------------------------------

        elif len(ct_refs) == 0:

            candidates = ct_df[
                (ct_df["PatientID"].astype(str) == patient_id)
                & (ct_df["StudyInstanceUID"].astype(str) == study_uid)
            ]

            if len(candidates) == 1:

                selected = candidates.iloc[0]

                method = "Unique_CT_in_same_study"

        if selected is None:

            print(" -> UNRESOLVED")

            pairs.append(
                {
                    "PatientID": patient_id,
                    "SEGSeriesUID": seg_uid,
                    "CTSeriesUID": "",
                    "CTSeriesDescription": "",
                    "CTImages": "",
                    "CTSizeMB": "",
                    "Method": "",
                    "Status": "UNRESOLVED",
                }
            )

            continue

        ct_uid = str(selected["SeriesInstanceUID"])

        description = str(selected["SeriesDescription"])

        image_count = safe_int(selected["instanceCount"])

        size_mb = safe_float(selected["series_size_MB"])

        print(f" -> {description} " f"({image_count} images)")

        pairs.append(
            {
                "PatientID": patient_id,
                "SEGSeriesUID": seg_uid,
                "CTSeriesUID": ct_uid,
                "CTSeriesDescription": description,
                "CTImages": image_count,
                "CTSizeMB": size_mb,
                "Method": method,
                "Status": "OK",
            }
        )

    pairs_df = pd.DataFrame(pairs)

    pairs_df.to_csv(
        PAIRS_CSV,
        index=False,
    )

    print(f"\nSaved:\n" f"{PAIRS_CSV}")

    failed = pairs_df[pairs_df["Status"] != "OK"]

    print(f"\nSuccess: " f"{len(pairs_df) - len(failed)}")

    print(f"Failed : " f"{len(failed)}")

    if len(failed) > 0:

        print(
            failed[
                [
                    "PatientID",
                    "Status",
                ]
            ].to_string(index=False)
        )

        raise RuntimeError("Some SEG -> CT pairs " "could not be resolved.")

    return pairs_df


# ============================================================
# LOAD OR BUILD PAIRS
# ============================================================

print("\n" + "=" * 80)
print("PAIRING")
print("=" * 80)

pairs_df = load_existing_pairs()

if pairs_df is not None:

    print("Valid pairs.csv found.")

    print("Reusing existing " "144/144 SEG -> CT mapping.")

    print("SEG parsing will NOT be " "performed again.")

else:

    print("No valid pairs.csv found.")

    print("Building SEG -> CT mapping...")

    pairs_df = build_pairs()


# ============================================================
# FILTER TEST PATIENT IF REQUESTED
# ============================================================

if TEST_PATIENT is not None:

    pairs_df = pairs_df[pairs_df["PatientID"] == TEST_PATIENT].reset_index(drop=True)


# ============================================================
# ENSURE SEG IS PRESENT
# ============================================================

print("\n" + "=" * 80)
print("CHECK SEG FILES")
print("=" * 80)

seg_uid_to_patient = {
    str(row["SEGSeriesUID"]): str(row["PatientID"]) for _, row in pairs_df.iterrows()
}


def seg_validator_from_pairs(uid):

    patient = seg_uid_to_patient[uid]

    return (
        find_local_seg_uid(
            patient,
            uid,
        )
        is not None
    )


seg_uids = list(seg_uid_to_patient.keys())

failed_seg = download_series(
    seg_uids,
    "SEG",
    seg_validator_from_pairs,
)

if failed_seg:

    print("\nSEG download incomplete.")

    sys.exit(1)


# ============================================================
# PREPARE CT RECORDS
# ============================================================

print("\n" + "=" * 80)
print("CT COHORT")
print("=" * 80)

ct_records = {}

for _, row in pairs_df.iterrows():

    uid = str(row["CTSeriesUID"])

    ct_records[uid] = row


def ct_validator(uid):

    return ct_complete(ct_records[uid])


ct_uids = list(ct_records.keys())

total_ct_size_gb = (
    pd.to_numeric(
        pairs_df["CTSizeMB"],
        errors="coerce",
    )
    .fillna(0)
    .sum()
    / 1024
)

total_images = (
    pd.to_numeric(
        pairs_df["CTImages"],
        errors="coerce",
    )
    .fillna(0)
    .sum()
)

already_complete_ct = [uid for uid in ct_uids if ct_validator(uid)]

pending_ct = [uid for uid in ct_uids if not ct_validator(uid)]

pending_size_gb = (
    sum(safe_float(ct_records[uid]["CTSizeMB"]) for uid in pending_ct) / 1024
)


print(f"Patients       : " f"{pairs_df['PatientID'].nunique()}")

print(f"SEG series     : " f"{pairs_df['SEGSeriesUID'].nunique()}")

print(f"CT series      : " f"{len(ct_uids)}")

print(f"CT DICOM files : " f"{int(total_images):,}")

print(f"Total CT size  : " f"{total_ct_size_gb:.2f} GB")

print(f"CT complete    : " f"{len(already_complete_ct)}")

print(f"CT incomplete  : " f"{len(pending_ct)}")

print(f"Remaining size : " f"~{pending_size_gb:.2f} GB")

print(f"Free disk      : " f"{free_disk_gb(OUTPUT_ROOT):.2f} GB")


# ============================================================
# SAVE SELECTED CT UID LIST
# ============================================================

uid_file = OUTPUT_ROOT / "selected_CT_SeriesInstanceUID.txt"

uid_file.write_text(
    "\n".join(ct_uids),
    encoding="utf-8",
)

print(f"\nCT UID list:\n" f"{uid_file}")


# ============================================================
# DOWNLOAD ONLY INCOMPLETE CT SERIES
# ============================================================

failed_ct = download_series(
    ct_uids,
    "CT",
    ct_validator,
)


# ============================================================
# FINAL VALIDATION TABLE
# ============================================================

print("\n" + "=" * 80)
print("FINAL VALIDATION")
print("=" * 80)

validation_rows = []

for _, row in pairs_df.iterrows():

    patient_id = str(row["PatientID"])

    seg_uid = str(row["SEGSeriesUID"])

    ct_uid = str(row["CTSeriesUID"])

    seg_ok = (
        find_local_seg_uid(
            patient_id,
            seg_uid,
        )
        is not None
    )

    ct_ok = ct_complete(row)

    ct_dir = OUTPUT_ROOT / patient_id / "CT"

    local_ct_count = 0

    if ct_dir.exists():

        local_ct_count = len(list(ct_dir.glob("*.dcm")))

    expected_ct_count = safe_int(row["CTImages"])

    validation_rows.append(
        {
            "PatientID": patient_id,
            "SEG_OK": seg_ok,
            "CT_OK": ct_ok,
            "Expected_CT_Files": expected_ct_count,
            "Local_CT_Files": local_ct_count,
            "CTSeriesUID": ct_uid,
        }
    )


validation_df = pd.DataFrame(validation_rows)

validation_csv = OUTPUT_ROOT / "validation.csv"

validation_df.to_csv(
    validation_csv,
    index=False,
)


complete = validation_df[validation_df["SEG_OK"] & validation_df["CT_OK"]]

incomplete = validation_df[~(validation_df["SEG_OK"] & validation_df["CT_OK"])]


print(f"Complete patients : " f"{len(complete)}")

print(f"Incomplete        : " f"{len(incomplete)}")

print(f"\nValidation saved:\n" f"{validation_csv}")


# ============================================================
# FINAL RESULT
# ============================================================

if len(incomplete) == 0:

    print("\n" + "=" * 80)
    print("SUCCESS")
    print("=" * 80)

    print(f"All " f"{pairs_df['PatientID'].nunique()} " f"patients have:")

    print("  - correct original SEG")

    print("  - correct referenced CT")

    print("  - complete DICOM CT series")

    print(f"\nDataset:\n" f"{OUTPUT_ROOT}")

else:

    print("\n" + "=" * 80)
    print("DOWNLOAD NOT YET COMPLETE")
    print("=" * 80)

    print(
        incomplete[
            [
                "PatientID",
                "SEG_OK",
                "CT_OK",
                "Expected_CT_Files",
                "Local_CT_Files",
            ]
        ].to_string(index=False)
    )

    print("\nDo NOT delete the dataset.")

    print("Run this script again.")

    print("It will skip completed data " "and retry only missing CT files.")

    sys.exit(1)
