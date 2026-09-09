'use client';

import { BookOpen, Database, ExternalLink, GitBranch, Target } from 'lucide-react';

import { PageContainer, PageHeader } from '@/components/layout/page-container';
import { StatusBadge } from '@/components/research/status-badges';
import { Badge } from '@/components/ui/badge';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/primitives';
import { DatasetBanner } from '@/components/research/dataset-banner';
import { useCalibrationMethods, useDatasets } from '@/hooks/use-pums';
import type { ImplementationStatus } from '@/types/api';

/**
 * Implementation status of each project objective.
 *
 * This page is the answer to "what actually works?" and is written to be
 * defensible in a review: nothing is marked implemented unless the code exists
 * in this repository and is exercised by the test suite.
 */
const OBJECTIVES: Array<{
  objective: string;
  status: ImplementationStatus;
  evidence: string;
}> = [
  {
    objective: 'Statistically calibrated uncertainty',
    status: 'implemented',
    evidence:
      'Split Conformal Risk Control with the finite-sample correction (Angelopoulos et al., ICLR 2024) in backend/app/calibration/conformal.py. Tests verify loss monotonicity, that λ̂ is the smallest feasible threshold, and that the correction is strictly more conservative than the uncorrected baseline.',
  },
  {
    objective: 'Distribution-shift-aware uncertainty',
    status: 'baseline',
    evidence:
      'A composite input-space detector (Wasserstein + Jensen–Shannon + standardised moments + edge density) with data-derived band thresholds. It flags shift and invalidates the conformal guarantee when it fires, but does not yet RESTORE validity under shift — that needs weighted/adaptive conformal prediction.',
  },
  {
    objective: 'Unified hierarchical pixel + object uncertainty',
    status: 'implemented',
    evidence:
      'One dense field is computed once; Levels 2 and 1 are strict aggregations of it, so the levels are consistent by construction rather than reconciled after the fact. Object level is populated from connected components, or from real detections when a detection adapter runs.',
  },
  {
    objective: 'Fine-grained localised uncertainty maps',
    status: 'implemented',
    evidence:
      'Per-pixel normalised predictive entropy transported as a PNG and decoded back to a Float32 matrix in the browser, so thresholding operates on real numbers. Verified end-to-end by a test that decodes the PNG and matches the reported Level-1 mean.',
  },
  {
    objective: 'Robust calibration across modalities',
    status: 'planned',
    evidence:
      'Calibration now runs on one REAL annotated MRI domain (MSD Hippocampus), not phantoms — but that is still a single modality. Cross-modality robustness needs a second downloaded domain (msd-spleen CT is registered for exactly this) and remains unmeasured.',
  },
  {
    objective: 'Lightweight post-hoc estimation',
    status: 'implemented',
    evidence:
      'Single forward pass, no ensembling, no MC dropout, no retraining. Typical end-to-end analysis is well under a second at 256×256 after warmup.',
  },
  {
    objective: 'Standardised evaluation framework',
    status: 'baseline',
    evidence:
      'Dice, IoU, ECE, Brier, AURC, selective risk and uncertainty–error correlation are implemented, unit-tested against analytically known answers, and now measured on a real annotated MRI dataset with patient-disjoint splits. What is still missing is evaluation across multiple standard benchmarks.',
  },
  {
    objective: 'Foundation-model reliability',
    status: 'planned',
    evidence:
      'The model-agnostic interface is in place and a second (torchvision CNN) adapter demonstrates it. No SAM/SAM2 integration and no foundation-model UQ work has been done.',
  },
  {
    objective: 'Supervised model trained on real annotated data',
    status: 'baseline',
    evidence:
      'A random-forest pixel classifier over a multiscale filter bank (Ilastik-style) trained on the MSD Hippocampus train split. Measured on the held-out TEST split: Dice 0.524, IoU 0.361, ECE 0.111, uncertainty–error correlation 0.512. A real but modest baseline — not a deep network and not clinically validated.',
  },
  {
    objective: 'Clinically actionable decision support',
    status: 'experimental',
    evidence:
      'A transparent weighted risk engine with explicit override rules produces a triage signal and referral priority. Every weight and threshold is configuration and is echoed in the API response. It is not clinically validated and makes no diagnostic claim.',
  },
];

const LITERATURE = [
  {
    paper: 'Angelopoulos et al., Conformal Risk Control',
    venue: 'ICLR 2024',
    role: 'Theoretical backbone of the calibration layer',
    status: 'implemented' as ImplementationStatus,
  },
  {
    paper: 'Guo et al., On Calibration of Modern Neural Networks',
    venue: 'ICML 2017',
    role: 'Temperature scaling (adapted to an ECE objective)',
    status: 'implemented' as ImplementationStatus,
  },
  {
    paper: 'El-Yaniv & Wiener, selective prediction',
    venue: 'JMLR 2010',
    role: 'Risk–coverage and selective-risk evaluation',
    status: 'implemented' as ImplementationStatus,
  },
  {
    paper: 'Zhang, Brady & Smith, hidden MRF + EM segmentation',
    venue: 'IEEE TMI 2001',
    role: 'Spatial regularisation in the baseline segmenter',
    status: 'baseline' as ImplementationStatus,
  },
  {
    paper: 'Blot et al., Automatically Adaptive Conformal Risk Control',
    venue: '2024',
    role: 'Target for shift-robust calibration',
    status: 'planned' as ImplementationStatus,
  },
  {
    paper: 'Bereska, Karimi & Samavi, SACP',
    venue: 'MIDL 2025',
    role: 'Spatially-aware conformal prediction sets',
    status: 'planned' as ImplementationStatus,
  },
  {
    paper: 'Zecchin & Simeone, Localized Adaptive Risk Control',
    venue: '2024',
    role: 'Region-conditional rather than marginal guarantees',
    status: 'planned' as ImplementationStatus,
  },
  {
    paper: 'Mossina et al., Morphological conformal segmentation',
    venue: 'MICCAI 2025',
    role: 'Black-box conformal segmentation without logits',
    status: 'planned' as ImplementationStatus,
  },
  {
    paper: 'Zou et al., DEviS evidential calibrated uncertainty',
    venue: 'IEEE TCyb 2025',
    role: 'Single-pass evidential alternative to conformal calibration',
    status: 'planned' as ImplementationStatus,
  },
  {
    paper: 'UncertainSAM',
    venue: 'ICML 2025',
    role: 'Foundation-model uncertainty quantification',
    status: 'planned' as ImplementationStatus,
  },
  {
    paper: 'Lambert et al., Trustworthy Clinical AI (review of 130 papers)',
    venue: 'Artificial Intelligence in Medicine 2024',
    role: 'Source of the research-gap framing for this project',
    status: 'implemented' as ImplementationStatus,
  },
];

const NOT_CLAIMED = [
  'No clinical validation of any kind has been performed. Training on real medical data is NOT the same as being validated for clinical use.',
  'The default segmenter is a random forest over filter responses, not a deep network. Its Dice of ~0.52 on hippocampus MRI is a modest research baseline, well below published deep-learning results on this task.',
  'Results are measured on ONE dataset (MSD Hippocampus, mono-modal MRI). Nothing here demonstrates generalisation to other scanners, sites, modalities or populations.',
  'Conformal guarantees are marginal over that dataset and require exchangeability. They lapse under the distribution shift this project targets, and PUMS reports that rather than hiding it.',
  'The "_shifted" sample images are SIMULATED acquisition corruption, not a real second site. No genuine multi-site OOD evaluation has been done.',
  'Mild acquisition shift (corruption strength 0.25) scores inside the in-distribution range and is measurably NOT detectable by the current detector.',
  'The risk engine output is a research signal about model reliability, never a statement about a patient.',
];

export default function ResearchPage() {
  const methods = useCalibrationMethods();
  const datasets = useDatasets();

  return (
    <PageContainer>
      <PageHeader
        title="Research implementation status"
        description="An honest map of what this repository actually does versus what the proposal aims at. Nothing is marked implemented unless the code exists here and the test suite exercises it."
      />

      <Tabs defaultValue="objectives">
        <TabsList>
          <TabsTrigger value="objectives">Objectives</TabsTrigger>
          <TabsTrigger value="methods">Methods</TabsTrigger>
          <TabsTrigger value="data">Data</TabsTrigger>
          <TabsTrigger value="literature">Literature map</TabsTrigger>
          <TabsTrigger value="limits">Not claimed</TabsTrigger>
        </TabsList>

        <TabsContent value="objectives">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-1.5">
                <Target className="size-4" />
                Project objectives
              </CardTitle>
            </CardHeader>
            <CardContent className="p-0">
              <table className="w-full text-xs">
                <thead className="border-b border-border text-[10px] uppercase tracking-wider text-muted-foreground">
                  <tr>
                    <th className="px-4 py-2 text-left font-semibold">Objective</th>
                    <th className="px-3 py-2 text-left font-semibold">Status</th>
                    <th className="px-4 py-2 text-left font-semibold">Evidence</th>
                  </tr>
                </thead>
                <tbody>
                  {OBJECTIVES.map((row) => (
                    <tr key={row.objective} className="border-b border-border/60 last:border-0">
                      <td className="max-w-[220px] px-4 py-2.5 align-top font-medium">
                        {row.objective}
                      </td>
                      <td className="px-3 py-2.5 align-top">
                        <StatusBadge status={row.status} />
                      </td>
                      <td className="px-4 py-2.5 align-top leading-relaxed text-muted-foreground">
                        {row.evidence}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="methods">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-1.5">
                <GitBranch className="size-4" />
                Calibration method catalogue
              </CardTitle>
              <p className="text-[11px] leading-relaxed text-muted-foreground">
                Read live from the backend registry. A method marked PLANNED returns
                HTTP 501 if invoked — the extension point exists, the algorithm does not.
              </p>
            </CardHeader>
            <CardContent className="p-0">
              <table className="w-full text-xs">
                <thead className="border-b border-border text-[10px] uppercase tracking-wider text-muted-foreground">
                  <tr>
                    <th className="px-4 py-2 text-left font-semibold">Method</th>
                    <th className="px-3 py-2 text-left font-semibold">Status</th>
                    <th className="px-4 py-2 text-left font-semibold">Description</th>
                    <th className="px-4 py-2 text-left font-semibold">Reference</th>
                  </tr>
                </thead>
                <tbody>
                  {methods.data?.methods.map((method) => (
                    <tr key={method.id} className="border-b border-border/60 last:border-0">
                      <td className="px-4 py-2.5 align-top font-medium">{method.name}</td>
                      <td className="px-3 py-2.5 align-top">
                        <StatusBadge status={method.status} />
                      </td>
                      <td className="max-w-md px-4 py-2.5 align-top leading-relaxed text-muted-foreground">
                        {method.description}
                      </td>
                      <td className="px-4 py-2.5 align-top text-[10px] italic text-muted-foreground">
                        {method.reference ?? '—'}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="data" className="space-y-4">
          <DatasetBanner />

          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-1.5">
                <Database className="size-4" />
                Registered datasets
              </CardTitle>
              <p className="text-[11px] leading-relaxed text-muted-foreground">
                No dataset is vendored into this repository. Each carries its
                licence and citation, and anything behind a data-use agreement is
                listed but never auto-downloaded — accepting those terms is your
                decision, not the code&apos;s.
              </p>
            </CardHeader>
            <CardContent className="p-0">
              <div className="overflow-x-auto">
                <table className="w-full text-xs">
                  <thead className="border-b border-border text-[10px] uppercase tracking-wider text-muted-foreground">
                    <tr>
                      <th className="px-4 py-2 text-left font-semibold">Dataset</th>
                      <th className="px-3 py-2 text-left font-semibold">Modality</th>
                      <th className="px-3 py-2 text-left font-semibold">Target</th>
                      <th className="px-3 py-2 text-right font-semibold">Size</th>
                      <th className="px-3 py-2 text-left font-semibold">Status</th>
                      <th className="px-4 py-2 text-left font-semibold">Licence</th>
                    </tr>
                  </thead>
                  <tbody>
                    {datasets.data?.datasets.map((dataset) => (
                      <tr key={dataset.id} className="border-b border-border/60 last:border-0">
                        <td className="px-4 py-2.5 align-top">
                          <p className="font-medium">{dataset.name}</p>
                          <p className="font-mono text-[10px] text-muted-foreground">
                            {dataset.id}
                          </p>
                          <p className="mt-1 max-w-md leading-relaxed text-muted-foreground">
                            {dataset.use_case}
                          </p>
                        </td>
                        <td className="px-3 py-2.5 align-top">
                          <Badge variant="outline">{dataset.modality}</Badge>
                        </td>
                        <td className="max-w-[200px] px-3 py-2.5 align-top leading-relaxed text-muted-foreground">
                          {dataset.target}
                        </td>
                        <td className="tabular px-3 py-2.5 text-right align-top text-muted-foreground">
                          {dataset.approx_mb ? `${dataset.approx_mb.toFixed(0)} MB` : '—'}
                        </td>
                        <td className="px-3 py-2.5 align-top">
                          {dataset.available ? (
                            <Badge variant="ok">Downloaded</Badge>
                          ) : dataset.requires_registration ? (
                            <Badge variant="warn">Needs signup</Badge>
                          ) : (
                            <Badge variant="outline">Available</Badge>
                          )}
                        </td>
                        <td className="px-4 py-2.5 align-top">
                          <a
                            href={dataset.homepage}
                            target="_blank"
                            rel="noreferrer"
                            className="inline-flex items-center gap-1 text-primary hover:underline"
                          >
                            {dataset.licence}
                            <ExternalLink className="size-3" />
                          </a>
                          <p className="mt-1 max-w-md leading-relaxed text-[10px] text-muted-foreground">
                            {dataset.citation}
                          </p>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Adding a dataset</CardTitle>
            </CardHeader>
            <CardContent>
              <pre className="overflow-x-auto rounded-md border border-border bg-muted/50 p-3 font-mono text-[11px] leading-relaxed">
{`# list the catalogue
python scripts/download_dataset.py

# fetch one (licence is printed and must be accepted)
python scripts/download_dataset.py msd-hippocampus

# train the supervised segmenter on its patient-disjoint train split
python scripts/train_pixel_model.py --dataset msd-hippocampus

# export real slices as demo samples, and re-derive the shift bands
python scripts/export_samples.py --dataset msd-hippocampus
python scripts/calibrate_shift_thresholds.py`}
              </pre>
              <p className="mt-2 text-[11px] leading-relaxed text-muted-foreground">
                Splits are patient-disjoint, never slice-level: adjacent slices of
                one patient are near-duplicates, so a slice split would leak the
                same anatomy across the boundary and inflate every reported metric
                including the conformal coverage.
              </p>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="literature">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-1.5">
                <BookOpen className="size-4" />
                Literature → implementation map
              </CardTitle>
              <p className="text-[11px] leading-relaxed text-muted-foreground">
                Which papers from the review have actually influenced code in this
                repository, and which remain future work.
              </p>
            </CardHeader>
            <CardContent className="p-0">
              <table className="w-full text-xs">
                <thead className="border-b border-border text-[10px] uppercase tracking-wider text-muted-foreground">
                  <tr>
                    <th className="px-4 py-2 text-left font-semibold">Paper</th>
                    <th className="px-3 py-2 text-left font-semibold">Venue</th>
                    <th className="px-4 py-2 text-left font-semibold">Role in PUMS</th>
                    <th className="px-4 py-2 text-left font-semibold">Status</th>
                  </tr>
                </thead>
                <tbody>
                  {LITERATURE.map((row) => (
                    <tr key={row.paper} className="border-b border-border/60 last:border-0">
                      <td className="px-4 py-2.5 align-top font-medium">{row.paper}</td>
                      <td className="px-3 py-2.5 align-top text-muted-foreground">
                        {row.venue}
                      </td>
                      <td className="px-4 py-2.5 align-top leading-relaxed text-muted-foreground">
                        {row.role}
                      </td>
                      <td className="px-4 py-2.5 align-top">
                        <StatusBadge status={row.status} />
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="limits">
          <Card className="border-status-warn/30">
            <CardHeader>
              <CardTitle>What this project does not claim</CardTitle>
              <p className="text-[11px] leading-relaxed text-muted-foreground">
                Stated explicitly so no reader has to infer it from an absence.
              </p>
            </CardHeader>
            <CardContent>
              <ul className="space-y-2">
                {NOT_CLAIMED.map((item) => (
                  <li
                    key={item}
                    className="border-l-2 border-status-warn/50 pl-3 text-xs leading-relaxed text-muted-foreground"
                  >
                    {item}
                  </li>
                ))}
              </ul>
            </CardContent>
          </Card>

          <Card className="mt-4">
            <CardHeader>
              <CardTitle>Recommended next steps</CardTitle>
            </CardHeader>
            <CardContent>
              <ol className="space-y-2 text-xs leading-relaxed text-muted-foreground">
                {[
                  'Register a real annotated medical split as the calibration source. Every synthetic-tagged number on this site becomes a genuine measurement the moment that happens — no algorithm changes are needed.',
                  'Add a trained medical segmentation adapter (nnU-Net or a MONAI bundle) behind BaseMedicalModel to replace the intensity-mixture baseline.',
                  'Implement adaptive / weighted conformal risk control so the guarantee survives the shift the detector already identifies. This is the project’s actual research contribution and is currently the largest gap.',
                  'Build paired ID/OOD splits so calibration degradation under shift becomes a measured quantity instead of a heuristic coupling.',
                  'Implement localized (region-conditional) risk control so per-region calibrated risk becomes a guarantee rather than a combination.',
                ].map((step, index) => (
                  <li key={step} className="flex gap-2">
                    <Badge variant="outline" className="mt-0.5 shrink-0">
                      {index + 1}
                    </Badge>
                    <span>{step}</span>
                  </li>
                ))}
              </ol>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </PageContainer>
  );
}
