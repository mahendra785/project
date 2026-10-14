'use client';

import { LiveLab } from './live-lab';
import { Mechanism } from './mechanism';
import { Backbones, EveryCase, GapTable, PosthocTable, SimGrid, Verdict } from './results';
import type { ShowcaseData } from './showcase-data';

const SECTIONS = [
  ['novelty', 'Novelty'],
  ['problem', 'Problem'],
  ['prior', 'Prior work'],
  ['method', 'Method'],
  ['workflow', 'Live workflow'],
  ['cases', 'Every real case'],
  ['simulation', 'Simulation'],
  ['posthoc', 'Post-hoc'],
  ['system', 'System'],
  ['reproduce', 'Reproduce'],
] as const;

function Section({ id, eyebrow, title, children, lede }: { id: string; eyebrow: string; title: string; lede?: string; children: React.ReactNode }) {
  return (
    <section id={id} className="scroll-mt-16 grid gap-4 border-t border-border pt-8">
      <div className="grid max-w-3xl gap-1.5">
        <p className="font-mono text-[11px] uppercase tracking-wider text-muted-foreground">{eyebrow}</p>
        <h2 className="text-xl font-semibold tracking-tight">{title}</h2>
        {lede && <p className="text-sm text-muted-foreground">{lede}</p>}
      </div>
      {children}
    </section>
  );
}

function Claim({ tone, label, title, body }: { tone: 'ok' | 'warn' | 'bad'; label: string; title: string; body: string }) {
  const t = tone === 'ok' ? 'bg-status-ok/15 text-status-ok' : tone === 'warn' ? 'bg-status-warn/15 text-status-warn' : 'bg-status-danger/15 text-status-danger';
  return (
    <div className="grid content-start gap-2 rounded-lg border border-border bg-card p-4">
      <span className={`w-max rounded px-1.5 py-0.5 font-mono text-[10.5px] font-semibold uppercase tracking-wide ${t}`}>{label}</span>
      <p className="font-semibold">{title}</p>
      <p className="text-sm text-muted-foreground">{body}</p>
    </div>
  );
}

const PRIOR: Array<[string, string, string, string, string]> = [
  ['Conformal Risk Control (ICLR 2024)', 'pixel FNR (any monotone loss)', 'expectation', 'no', 'no'],
  ['Learn-Then-Test (arXiv 2110.01052)', 'any', 'high-probability', 'yes (FWER)', 'no'],
  ['Conformal Lesion Segmentation (arXiv 2510.17897)', 'voxel FNR per volume', 'high-probability', 'no', 'no'],
  ['CRA / CCRA / CCRA-S (arXiv 2504.07611)', 'pixel FNR per image', 'expectation per stratum', 'no', 'image strata'],
  ['ReCIRC (arXiv 2609.38112)', 'pixel FNR per image', 'marginal + approx. conditional', 'no', 'no'],
  ['Morphological sets (MICCAI 2025)', 'mask containment', 'high-probability', 'no', 'no'],
  ['Per-organ recall (arXiv 2608.18193)', 'voxel FNR per organ', 'high-probability', 'Bonferroni', 'no'],
  ['FDR instance segmentation (arXiv 2504.04482)', 'pixel FDR or FNR', 'expectation', 'one at a time', 'no'],
  ['Andéol et al., COPA 2023 (detection)', 'box or pixel recall', 'expectation', 'one at a time', 'no'],
];

export function Showcase({ data }: { data: ShowcaseData }) {
  const blocks = data.blocks ?? [];
  const all = blocks.flatMap((b) => b.rows);
  const range = (m: string) => {
    const v = all.filter((r) => r.method === m).map((r) => 100 * (r.rel ?? 0));
    return v.length ? [Math.min(...v), Math.max(...v)] : [0, 0];
  };
  const ltt = range('ltt_threshold');
  const dil = range('union_dilation');
  const pixCost = Math.max(0, ...all.filter((r) => r.method === 'pixel_crc').map((r) => Math.abs(100 * (r.rel ?? 0))));
  const kv = blocks.find((b) => b.dataset === 'kvasir-seg');
  const gap = kv?.gap.find((g) => g.alpha === 0.1)?.G_multi;
  const allValid = blocks.every((b) => Object.values(b.verdicts.validity).every((v) => v === 'ok'));
  const simBest = data.sim ? Math.min(...data.sim.rows.map((r) => r.rel)) : null;

  return (
    <div className="mx-auto grid max-w-6xl gap-8 px-4 py-6 md:px-6">
      <header className="grid max-w-3xl gap-2">
        <p className="font-mono text-[11px] uppercase tracking-wider text-muted-foreground">Project-II · lesion-level conformal risk control</p>
        <h1 className="text-3xl font-semibold tracking-tight">Pixel guarantees hide missed lesions</h1>
        <p className="text-muted-foreground">
          This page walks through the whole project in order: the problem, why it is new, how the method works, a live run of
          the full pipeline, and every recorded result, including the ones that did not go our way. Recorded numbers are read
          from <code className="font-mono text-xs">{data.source}</code>. Live numbers are computed in your browser.
        </p>
      </header>

      <nav aria-label="Showcase sections" className="sticky top-0 z-10 -mx-4 flex gap-1 overflow-x-auto border-b border-border bg-background/95 px-4 py-2 backdrop-blur md:-mx-6 md:px-6">
        {SECTIONS.map(([id, label], i) => (
          <a key={id} href={`#${id}`} className="shrink-0 rounded-md px-2.5 py-1 text-xs text-muted-foreground hover:bg-muted hover:text-foreground">
            <span className="mr-1 font-mono text-[10px]">{i + 1}</span>{label}
          </a>
        ))}
      </nav>

      {!data.blocks && (
        <p className="rounded-md border border-status-warn/40 bg-status-warn/10 p-3 text-sm">
          Recorded results were not found under docs/research/results/. Run scripts/reproduce to generate them. The live workflow below still works.
        </p>
      )}

      <Section id="novelty" eyebrow="1 · the answer first" title="What is new, and how strong the claim is"
        lede="A literature search on 14 October 2026 (arXiv and the web, 2021 onward, twelve query strings) found no prior method doing the first two things below.">
        <div className="grid gap-3 md:grid-cols-3">
          <Claim tone="ok" label="New · supported" title="Lesion-level risk" body="A guarantee on ground-truth lesions (connected components), plus a separate one for small lesions. Every published segmentation method bounds pixels or voxels." />
          <Claim tone="ok" label="New · supported" title="Joint control" body={`Pixel, lesion and small-lesion risk held simultaneously, in expectation, with no Bonferroni split. Costs at most ${pixCost.toFixed(1)}% in set size over pixel-only control.`} />
          <Claim tone="warn" label="New · not yet shown on real data" title="Lesion-equalized sets" body={`A prediction-set family designed for a lesion loss. ${simBest !== null ? `Up to ${(-100 * simBest).toFixed(0)}% smaller sets in simulation` : 'Large gains in simulation'}; tied thresholding on both real datasets.`} />
        </div>
        <div className="grid gap-2 rounded-lg border border-border bg-card p-4 text-sm md:grid-cols-2">
          <div className="grid gap-1">
            <p className="font-semibold">Established by the experiments</p>
            <ul className="list-disc space-y-1 pl-5 text-muted-foreground">
              <li>{allValid ? 'Every method met every guarantee it claims, on held-out data.' : 'Some guarantee violations were observed; see the tables.'}</li>
              <li>Pixel CRC under-protects multi-lesion images: miss rate {gap ? `${gap.toFixed(2)}×` : 'above'} α on Kvasir-SEG.</li>
              <li>Under the same lesion-level guarantee, Learn-Then-Test needs {ltt[0].toFixed(0)}–{ltt[1].toFixed(0)}% and morphological sets {Math.max(0, dil[0]).toFixed(0)}–{dil[1].toFixed(0)}% larger sets.</li>
            </ul>
          </div>
          <div className="grid gap-1">
            <p className="font-semibold">Not established</p>
            <ul className="list-disc space-y-1 pl-5 text-muted-foreground">
              <li>Lesion-equalized sets beating joint thresholding on real data (pre-registered H2 failed: a tie).</li>
              <li>Anything with a strong backbone: the reported runs use a CPU random forest.</li>
              <li>Any clinical claim.</li>
            </ul>
          </div>
        </div>
      </Section>

      <Section id="problem" eyebrow="2 · the problem" title="One threshold for every lesion"
        lede="Conformal risk control picks a cutoff so that the expected fraction of missed lesion pixels is at most α. Large lesions dominate that average. To also find a faint small lesion, thresholding must drop its cutoff for the whole image. Move the sliders to see it.">
        <div className="rounded-lg border border-border bg-card p-4"><Mechanism /></div>
      </Section>

      <Section id="prior" eyebrow="3 · prior work" title="How every close paper measures risk">
        <div className="overflow-x-auto rounded-md border border-border">
          <table className="w-full text-xs [&_td]:border-b [&_td]:border-border/60 [&_td]:px-2 [&_td]:py-1.5 [&_th]:border-b [&_th]:border-border [&_th]:bg-muted [&_th]:px-2 [&_th]:py-1.5 [&_th]:text-left [&_th]:font-mono [&_th]:text-[10px] [&_th]:uppercase [&_th]:text-muted-foreground">
            <thead><tr><th>Paper</th><th>Risk measured in</th><th>Guarantee</th><th>Several risks at once</th><th>Size-aware</th></tr></thead>
            <tbody>
              {PRIOR.map((r) => <tr key={r[0]}>{r.map((c, i) => <td key={i}>{c}</td>)}</tr>)}
              <tr className="bg-primary/10 font-medium"><td>This work</td><td>pixel + ground-truth lesion + small lesion</td><td>expectation</td><td>yes, no Bonferroni</td><td>yes</td></tr>
            </tbody>
          </table>
        </div>
      </Section>

      <Section id="method" eyebrow="4 · method" title="Three losses, one threshold, a new set family">
        <div className="grid gap-3 md:grid-cols-3">
          <div className="grid content-start gap-2 rounded-lg border border-border bg-card p-4 text-sm">
            <p className="font-semibold">Losses on lesions</p>
            <p className="text-muted-foreground">Pixel FNR, lesion miss and small-lesion miss. A lesion is found when at least 50% of it is inside the set. All three shrink as the set grows (Lemma 1). Each lesion reduces to one order statistic (Lemma 2), so the lesion loss is as cheap as the pixel loss.</p>
          </div>
          <div className="grid content-start gap-2 rounded-lg border border-border bg-card p-4 text-sm">
            <p className="font-semibold">Joint threshold</p>
            <p className="text-muted-foreground">Calibrate each risk with split CRC, (n·R̂ + 1)/(n + 1) ≤ α, and take the largest λ. Monotonicity means every risk stays ≤ α at once. A max-form variant bounds the per-image worst risk.</p>
          </div>
          <div className="grid content-start gap-2 rounded-lg border border-border bg-card p-4 text-sm">
            <p className="font-semibold">Lesion-equalized sets</p>
            <p className="text-muted-foreground">Inside each candidate of {'{'}p ≥ t₀{'}'}, score = p / peak^ρ. The score ignores model confidence at ρ = 1 (Proposition 2), excludes background above t₀ (Proposition 3), and equals plain thresholding at ρ = 0. (ρ, t₀) is picked on a separate tuning split, so the guarantee holds exactly (Proposition 1).</p>
          </div>
        </div>
      </Section>

      <Section id="workflow" eyebrow="5 · live workflow" title="Run the whole pipeline"
        lede="Every step below runs in your browser, on 300 generated scans, with the same algorithm as the backend. Click a step to see its real intermediate output. Raise γ to make small lesions fainter, and watch where equalization helps and where it stops helping.">
        <LiveLab />
      </Section>

      {blocks.length > 0 && (
        <Section id="cases" eyebrow="6 · every real case" title="Every dataset, every α, every method"
          lede="Pre-registered runs: 3 training seeds × 1000 random tune/cal/test splits per dataset. Baselines calibrate on 50% more data than our method.">
          <EveryCase blocks={blocks} />
          <div className="grid gap-2">
            <p className="text-sm font-semibold">The gap at the pixel-CRC threshold (H1)</p>
            <GapTable blocks={blocks} />
            <p className="text-[11px] text-muted-foreground">G = lesion-miss rate ÷ α. PROCEED needs G ≥ 1.5 at α = 0.10 on the multi-lesion or small-lesion subpopulation.</p>
          </div>
          <div className="grid gap-2">
            <p className="text-sm font-semibold">Backbones (5-fold cross-fitting, out-of-fold Dice)</p>
            <Backbones items={data.backbones} />
          </div>
        </Section>
      )}

      {data.sim && (
        <Section id="simulation" eyebrow="7 · controlled simulation" title="When does equalization help?"
          lede="Small-lesion confidence scales as radius^γ, at two background levels: 600 scans and 500 splits per cell. The pre-registered rule asked for a gain at every γ ≥ 1 that grows with γ.">
          <SimGrid rows={data.sim.rows} diag={data.diag} verdict={data.sim.verdict} />
          <p className="max-w-3xl text-sm text-muted-foreground">The gain peaks at moderate coupling and disappears once most small lesions no longer rise above the background. No score built from the posterior alone can separate them then, and the method falls back to thresholding rather than doing worse.</p>
        </Section>
      )}

      {Object.keys(data.e8).length > 0 && (
        <Section id="posthoc" eyebrow="8 · post-hoc, not pre-registered" title="Closing the multi-lesion gap"
          lede="The joint guarantee averages lesion misses over all images, so multi-lesion images can still exceed α. A risk defined only on images with two or more lesions is valid for CRC, for the same reason as the small-lesion risk, and adding it closes most of the gap.">
          <PosthocTable e8={data.e8} />
        </Section>
      )}

      <Section id="system" eyebrow="9 · system" title="Where it lives in PUMS">
        <div className="overflow-x-auto rounded-md border border-border">
          <table className="w-full text-xs [&_td]:border-b [&_td]:border-border/60 [&_td]:px-2 [&_td]:py-1.5 [&_th]:border-b [&_th]:border-border [&_th]:bg-muted [&_th]:px-2 [&_th]:py-1.5 [&_th]:text-left [&_th]:font-mono [&_th]:text-[10px] [&_th]:uppercase [&_th]:text-muted-foreground">
            <thead><tr><th>Stage</th><th>Code</th><th>Output</th></tr></thead>
            <tbody>
              <tr><td>Load data</td><td className="font-mono">backend/app/datasets/loaders.py</td><td>images + components labelled at native resolution</td></tr>
              <tr><td>Train backbone</td><td className="font-mono">backend/scripts/experiments/train_backbone.py</td><td>out-of-fold posteriors + sha256 manifest</td></tr>
              <tr><td>Score families</td><td className="font-mono">backend/app/calibration/families.py</td><td>threshold · dilation · lesion-equalized</td></tr>
              <tr><td>Losses</td><td className="font-mono">backend/app/calibration/losses.py</td><td>per-image grid counts</td></tr>
              <tr><td>Calibrate</td><td className="font-mono">backend/app/calibration/joint.py</td><td>CRC · union · max-form · LTT · strata · family selection</td></tr>
              <tr><td>Experiments</td><td className="font-mono">mgcrc.py · simulation.py · posthoc_multi.py</td><td>per-split CSV + summary JSON</td></tr>
              <tr><td>Verdicts &amp; paper</td><td className="font-mono">aggregate.py · paper_macros.py</td><td>summary.md · paper/generated/*.tex</td></tr>
              <tr><td>App</td><td className="font-mono">calibration/lesion_crc.py · /api/research/mgcrc</td><td>lesion_aware_crc method · Research tab</td></tr>
            </tbody>
          </table>
        </div>
      </Section>

      <Section id="reproduce" eyebrow="10 · reproduce" title="Regenerate every number">
        <div className="grid gap-2 rounded-lg border border-border bg-card p-4 text-sm">
          <pre className="overflow-x-auto rounded bg-muted p-3 font-mono text-xs">{`./scripts/reproduce.ps1          # full run on CPU (backbones, experiments, simulation, verdicts, paper)
./scripts/reproduce.ps1 -Fast    # reuse cached posteriors
./scripts/reproduce.ps1 -Backbone unet-r34 -Size 512   # GPU backbone, once PyTorch is installed`}</pre>
          <p className="text-muted-foreground">Pre-registration: docs/research/preregistration-mgcrc.md, committed before any result. Theory: docs/research/theory-memo-v2.md. Paper: paper/. Report and slides: report/.</p>
          <p className="text-xs text-muted-foreground">Status: H1 <Verdict value={kv?.verdicts.H1 ?? '–'} /> on Kvasir-SEG · H2 <Verdict value="FAILED" /> on both datasets · simulation rule <Verdict value={data.sim?.verdict ?? '–'} /> · research prototype, not a medical device.</p>
        </div>
      </Section>
    </div>
  );
}
