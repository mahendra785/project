'use client';

import { BookOpen, GitBranch, Target } from 'lucide-react';

import { PageContainer, PageHeader } from '@/components/layout/page-container';
import { StatusBadge } from '@/components/research/status-badges';
import { Badge } from '@/components/ui/badge';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/primitives';
import { useCalibrationMethods } from '@/hooks/use-pums';
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
      'Only one calibration domain (synthetic phantoms) exists. Cross-modality robustness cannot be claimed, or even measured, without multi-modality annotated data.',
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
      'Dice, IoU, ECE, Brier, AURC, selective risk and uncertainty–error correlation are implemented and unit-tested against analytically known answers. What is missing is standard DATASETS, not standard metrics.',
  },
  {
    objective: 'Foundation-model reliability',
    status: 'planned',
    evidence:
      'The model-agnostic interface is in place and a second (torchvision CNN) adapter demonstrates it. No SAM/SAM2 integration and no foundation-model UQ work has been done.',
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
  'No clinical validation of any kind has been performed.',
  'No medical dataset has been used; the calibration and reference sets are procedurally generated phantoms.',
  'No shipped model adapter is a medical model — the default is an unsupervised intensity mixture, the optional CNN is trained on natural images.',
  'No accuracy, Dice or sensitivity figure on this site refers to medical images.',
  'Conformal guarantees are marginal and hold only under exchangeability; they lapse under the distribution shift this project targets, and PUMS reports that rather than hiding it.',
  'The risk engine output is a research signal about model reliability, never a statement about a patient.',
];

export default function ResearchPage() {
  const methods = useCalibrationMethods();

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
