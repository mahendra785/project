"""Catalogue of real medical imaging datasets PUMS can use.

WHY THIS EXISTS
---------------
PUMS previously calibrated on procedural phantoms, which meant every derived
figure had to be labelled `synthetic` and could not support a research claim.
Registering a real annotated dataset here turns λ̂, ECE, Dice, IoU, AURC and the
uncertainty-error correlation into genuine measurements on genuine anatomy.

LICENSING IS PART OF THE RECORD
-------------------------------
No dataset is vendored into this repository. Each entry carries its licence and
citation, both of which are printed by the download script and surfaced through
the API, because a research project that cannot say where its data came from
cannot defend its results.

Datasets requiring a signed data-use agreement (BraTS via Synapse, Stanford
MURA, most Kaggle competition data) are deliberately NOT auto-downloadable —
they are listed with `requires_registration=True` and a URL, so the choice to
accept those terms stays with the user.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Modality(str, Enum):
    MRI = "MRI"
    CT = "CT"
    XRAY = "X-ray"
    ENDOSCOPY = "Endoscopy"
    DERMOSCOPY = "Dermoscopy"


class DatasetFormat(str, Enum):
    #: NIfTI volumes (.nii.gz) inside a tar — Medical Segmentation Decathlon.
    MSD_NIFTI = "msd_nifti"
    #: Paired image/mask PNG or JPG directories inside a zip.
    IMAGE_MASK_ZIP = "image_mask_zip"
    #: Images plus a COCO JSON of segmentation polygons inside a zip (FracAtlas).
    COCO_POLYGON_ZIP = "coco_polygon_zip"


@dataclass(frozen=True)
class DatasetSpec:
    """Everything needed to fetch, load, cite and legally use a dataset."""

    id: str
    name: str
    modality: Modality
    #: What the segmentation mask actually delineates.
    target: str
    #: Clinical use case this supports, in plain language.
    use_case: str
    fmt: DatasetFormat
    url: str | None
    approx_bytes: int
    licence: str
    licence_url: str
    citation: str
    homepage: str
    #: True when a data-use agreement must be accepted manually first.
    requires_registration: bool = False
    #: Label values that count as foreground for the binary reduction the
    #: calibration layer needs. Empty means "every non-zero label".
    foreground_labels: list[int] = field(default_factory=list)
    #: Sub-directory inside the archive holding images / labels.
    images_dir: str = "imagesTr"
    labels_dir: str = "labelsTr"
    notes: str = ""

    @property
    def approx_mb(self) -> float:
        return round(self.approx_bytes / (1024 * 1024), 1)

    @property
    def downloadable(self) -> bool:
        return self.url is not None and not self.requires_registration


_MSD_LICENCE = "CC BY-SA 4.0"
_MSD_LICENCE_URL = "https://creativecommons.org/licenses/by-sa/4.0/"
_MSD_CITATION = (
    "Antonelli, M., Reinke, A., Bakas, S. et al. "
    "'The Medical Segmentation Decathlon.' Nature Communications 13, 4128 (2022)."
)
_MSD_HOME = "http://medicaldecathlon.com/"


DATASETS: dict[str, DatasetSpec] = {
    # ------------------------------------------------------------------ MRI --
    "msd-hippocampus": DatasetSpec(
        id="msd-hippocampus",
        name="MSD Task04 — Hippocampus (MRI)",
        modality=Modality.MRI,
        target="Hippocampus head (anterior) and body (posterior)",
        use_case=(
            "Neurodegenerative disease assessment — hippocampal volumetry is a "
            "standard biomarker in Alzheimer's and epilepsy workups."
        ),
        fmt=DatasetFormat.MSD_NIFTI,
        url="https://msd-for-monai.s3-us-west-2.amazonaws.com/Task04_Hippocampus.tar",
        approx_bytes=28_425_216,
        licence=_MSD_LICENCE,
        licence_url=_MSD_LICENCE_URL,
        citation=_MSD_CITATION,
        homepage=_MSD_HOME,
        foreground_labels=[1, 2],
        notes=(
            "260 mono-modal MRI volumes, tightly cropped around the hippocampus "
            "(~35x50 px slices). Small and fast to download, which makes it the "
            "default. Slices are low-resolution by nature of the crop."
        ),
    ),
    "msd-heart": DatasetSpec(
        id="msd-heart",
        name="MSD Task02 — Left Atrium (Cardiac MRI)",
        modality=Modality.MRI,
        target="Left atrium blood pool",
        use_case=(
            "Atrial fibrillation planning — left-atrial geometry guides ablation "
            "and stroke-risk assessment."
        ),
        fmt=DatasetFormat.MSD_NIFTI,
        url="https://msd-for-monai.s3-us-west-2.amazonaws.com/Task02_Heart.tar",
        approx_bytes=456_000_000,
        licence=_MSD_LICENCE,
        licence_url=_MSD_LICENCE_URL,
        citation=_MSD_CITATION,
        homepage=_MSD_HOME,
        foreground_labels=[1],
        notes=(
            "20 mono-modal cardiac MRI volumes at 320x320 per slice. Full-field-"
            "of-view anatomy, so it looks like a real clinical MRI in the viewer "
            "and forms a genuine cross-protocol shift pair with Hippocampus."
        ),
    ),
    "msd-brain-tumour": DatasetSpec(
        id="msd-brain-tumour",
        name="MSD Task01 — Brain Tumour (multimodal MRI)",
        modality=Modality.MRI,
        target="Glioma: oedema, enhancing and non-enhancing tumour",
        use_case=(
            "Neuro-oncology — tumour delineation for surgical planning and "
            "treatment-response monitoring. The flagship medical UQ benchmark."
        ),
        fmt=DatasetFormat.MSD_NIFTI,
        url="https://msd-for-monai.s3-us-west-2.amazonaws.com/Task01_BrainTumour.tar",
        approx_bytes=7_700_000_000,
        licence=_MSD_LICENCE,
        licence_url=_MSD_LICENCE_URL,
        citation=_MSD_CITATION,
        homepage=_MSD_HOME,
        foreground_labels=[1, 2, 3],
        notes=(
            "484 four-channel (FLAIR/T1/T1c/T2) volumes at 240x240. The BraTS "
            "subset redistributed under CC BY-SA, so no Synapse registration is "
            "needed. ~7.7 GB — download deliberately, not by default."
        ),
    ),
    # ------------------------------------------------------------------- CT --
    "msd-spleen": DatasetSpec(
        id="msd-spleen",
        name="MSD Task09 — Spleen (CT)",
        modality=Modality.CT,
        target="Spleen",
        use_case="Abdominal organ volumetry; a clean CT counterpart to the MRI tasks.",
        fmt=DatasetFormat.MSD_NIFTI,
        url="https://msd-for-monai.s3-us-west-2.amazonaws.com/Task09_Spleen.tar",
        approx_bytes=1_600_000_000,
        licence=_MSD_LICENCE,
        licence_url=_MSD_LICENCE_URL,
        citation=_MSD_CITATION,
        homepage=_MSD_HOME,
        foreground_labels=[1],
        notes=(
            "Useful as a true cross-MODALITY shift partner: calibrate on MRI, "
            "test on CT, and the exchangeability assumption visibly breaks."
        ),
    ),
    "msd-hepatic-vessel": DatasetSpec(
        id="msd-hepatic-vessel",
        name="MSD Task08 — Hepatic Vessels & Tumours (CT)",
        modality=Modality.CT,
        target="Liver tumours (label 2); hepatic vessels (label 1) ignored",
        use_case=(
            "Liver metastasis and HCC assessment — frequently MULTIPLE tumours "
            "per liver, which makes this the one openly-licensed dataset in reach "
            "where lesion-level and pixel-level risk genuinely diverge."
        ),
        fmt=DatasetFormat.MSD_NIFTI,
        url="https://msd-for-monai.s3-us-west-2.amazonaws.com/Task08_HepaticVessel.tar",
        approx_bytes=8_700_000_000,
        licence=_MSD_LICENCE,
        licence_url=_MSD_LICENCE_URL,
        citation=_MSD_CITATION,
        homepage=_MSD_HOME,
        foreground_labels=[2],
        notes=(
            "303 training CT volumes. Foreground is the TUMOUR label only, so "
            "connected components are lesions rather than vasculature. ~8.7 GB. "
            "Requires a volume-level calibration unit (26-connectivity) — the "
            "2-D slice sampler misrepresents lesion counts here."
        ),
    ),
    # ---------------------------------------------------------------- X-ray --
    "fracatlas": DatasetSpec(
        id="fracatlas",
        name="FracAtlas — Musculoskeletal Fracture X-ray",
        modality=Modality.XRAY,
        target="Fracture site (segmentation masks + boxes for the fractured subset)",
        use_case=(
            "Emergency-department fracture triage — the use case where a "
            "calibrated 'I am not sure, escalate this' signal has the clearest "
            "clinical value."
        ),
        fmt=DatasetFormat.COCO_POLYGON_ZIP,
        # Resolved from the figshare article API (v7); the article-page
        # "download" link returns 202/empty until figshare has staged the file.
        url="https://ndownloader.figshare.com/files/65518038",
        approx_bytes=338_400_000,
        licence="CC BY 4.0",
        licence_url="https://creativecommons.org/licenses/by/4.0/",
        citation=(
            "Abedeen, I., Rahman, M.A., Prottyasha, F.Z. et al. 'FracAtlas: A "
            "Dataset for Fracture Classification, Localization and Segmentation "
            "of Musculoskeletal Radiographs.' Scientific Data 10, 521 (2023)."
        ),
        homepage="https://figshare.com/articles/dataset/The_dataset/22363012",
        notes=(
            "4083 radiographs; 717 are fractured and carry localisation "
            "annotations. Directly answers the fracture use case. Only the "
            "fractured subset has masks, so the calibration split is built from "
            "those."
        ),
    ),
    # ------------------------------------------------------------ Endoscopy --
    "kvasir-seg": DatasetSpec(
        id="kvasir-seg",
        name="Kvasir-SEG — Polyp Segmentation",
        modality=Modality.ENDOSCOPY,
        target="Colorectal polyp",
        use_case="Colonoscopy screening — missed-polyp rate is the classic FNR problem.",
        fmt=DatasetFormat.IMAGE_MASK_ZIP,
        url="https://datasets.simula.no/downloads/kvasir-seg.zip",
        approx_bytes=46_000_000,
        licence="CC BY 4.0",
        licence_url="https://creativecommons.org/licenses/by/4.0/",
        citation=(
            "Jha, D., Smedsrud, P.H., Riegler, M.A. et al. 'Kvasir-SEG: A "
            "Segmented Polyp Dataset.' MMM 2020."
        ),
        homepage="https://datasets.simula.no/kvasir-seg/",
        notes=(
            "1000 RGB frames with expert masks. Small, fast, and the false-"
            "negative-rate loss used by Conformal Risk Control is exactly the "
            "clinically meaningful quantity here."
        ),
    ),
    # ------------------------------------------- registration-gated entries --
    "brats": DatasetSpec(
        id="brats",
        name="BraTS — Brain Tumour Segmentation Challenge",
        modality=Modality.MRI,
        target="Glioma sub-regions",
        use_case="The reference benchmark for medical segmentation uncertainty.",
        fmt=DatasetFormat.MSD_NIFTI,
        url=None,
        approx_bytes=0,
        licence="Challenge data-use agreement",
        licence_url="https://www.synapse.org/brats",
        citation="Menze, B.H. et al. IEEE TMI 34(10), 1993-2024 (2015).",
        homepage="https://www.synapse.org/brats",
        requires_registration=True,
        notes=(
            "Requires a Synapse account and acceptance of the challenge DUA. Use "
            "msd-brain-tumour for an openly-licensed subset of the same data."
        ),
    ),
    "mura": DatasetSpec(
        id="mura",
        name="MURA — Musculoskeletal Radiographs (Stanford)",
        modality=Modality.XRAY,
        target="Study-level abnormality label (classification, no masks)",
        use_case="Fracture and abnormality screening across seven body regions.",
        fmt=DatasetFormat.IMAGE_MASK_ZIP,
        url=None,
        approx_bytes=0,
        licence="Stanford ML Group research-use agreement",
        licence_url="https://stanfordmlgroup.github.io/competitions/mura/",
        citation="Rajpurkar, P. et al. 'MURA: Large Dataset for Abnormality Detection.' MIDL 2018.",
        homepage="https://stanfordmlgroup.github.io/competitions/mura/",
        requires_registration=True,
        notes=(
            "Classification only — no segmentation masks, so it cannot feed the "
            "dense conformal calibration path. Listed for the fracture use case; "
            "FracAtlas is the segmentable alternative."
        ),
    ),
}

#: Downloaded by default: real MRI, small, and together they form a genuine
#: cross-protocol pair for measuring distribution shift on real data.
DEFAULT_DATASET_ID = "msd-hippocampus"


def get_spec(dataset_id: str) -> DatasetSpec:
    if dataset_id not in DATASETS:
        raise KeyError(
            f"Unknown dataset '{dataset_id}'. Available: {', '.join(sorted(DATASETS))}"
        )
    return DATASETS[dataset_id]


def list_specs() -> list[DatasetSpec]:
    return list(DATASETS.values())
