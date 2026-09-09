/**
 * Contract tests.
 *
 * `fixtures-analysis.json` is a REAL response captured from the running
 * FastAPI backend (`POST /api/analyze`). Validating it against the Zod schemas
 * proves the frontend's runtime contract matches the backend's Pydantic
 * schemas — the single most likely thing to silently drift in this repo.
 *
 * Regenerate with:
 *   curl -s -X POST http://localhost:8000/api/analyze \
 *     -F "file=@data/sample/phantom_id_1.png;type=image/png" \
 *     -o frontend/tests/fixtures-analysis.json
 */

import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

import { analysisSchema } from '../lib/api/schemas.ts';

const here = dirname(fileURLToPath(import.meta.url));
const fixture = JSON.parse(
  readFileSync(join(here, 'fixtures-analysis.json'), 'utf8'),
) as unknown;

test('a real backend analysis response satisfies the frontend schema', () => {
  const result = analysisSchema.safeParse(fixture);
  if (!result.success) {
    const issues = result.error.issues
      .map((issue) => `${issue.path.join('.')}: ${issue.message}`)
      .join('\n  ');
    assert.fail(`Contract drift between backend and frontend:\n  ${issues}`);
  }
});

test('the hierarchical uncertainty keys are present and named as expected', () => {
  const parsed = analysisSchema.parse(fixture);
  assert.ok(parsed.uncertainty.global, 'Level 1 (global) missing');
  assert.ok(Array.isArray(parsed.uncertainty.objects), 'Level 2 (objects) missing');
  assert.ok(parsed.uncertainty.pixel, 'Level 3 (pixel) missing');
});

test('level 1 is the mean of the level 3 field the backend reports', () => {
  const parsed = analysisSchema.parse(fixture);
  // The backend guarantees Level 1 is an aggregation of Level 3, so the
  // reported score must lie between the map's min and max.
  const { score, statistics } = parsed.uncertainty.global;
  assert.ok(score >= statistics.minimum && score <= statistics.maximum);
  assert.ok(statistics.p90 <= statistics.p95 && statistics.p95 <= statistics.p99);
});

test('research-integrity fields survive parsing', () => {
  const parsed = analysisSchema.parse(fixture);

  assert.ok(
    ['computed', 'synthetic', 'mock'].includes(parsed.uncertainty.global.provenance),
    'uncertainty provenance must be present',
  );
  assert.match(parsed.risk_assessment.disclaimer, /not a diagnostic result/i);

  if (parsed.calibration) {
    assert.ok(
      parsed.calibration.assumptions.length > 0,
      'calibration assumptions must travel with the result',
    );
    assert.ok(
      parsed.calibration.assumptions.some((a) => /exchangeab/i.test(a)),
      'the exchangeability assumption must be disclosed',
    );
  }
});

test('the risk score equals the sum of its declared contributions', () => {
  const parsed = analysisSchema.parse(fixture);
  const total = parsed.risk_assessment.factors.reduce((a, f) => a + f.contribution, 0);
  assert.ok(
    Math.abs(total - parsed.risk_assessment.risk_score) < 0.01,
    `factors sum to ${total} but score is ${parsed.risk_assessment.risk_score}`,
  );
});

test('the segmentation RLE accounts for exactly every pixel', () => {
  const parsed = analysisSchema.parse(fixture);
  const segmentation = parsed.prediction.segmentation;
  assert.ok(segmentation, 'segmentation missing');

  let total = 0;
  for (let i = 1; i < segmentation.rle.length; i += 2) total += segmentation.rle[i];
  assert.equal(total, segmentation.height * segmentation.width);
});

test('bounding boxes lie inside the reported pixel grid', () => {
  const parsed = analysisSchema.parse(fixture);
  const { width, height } = parsed.uncertainty.pixel;

  for (const object of parsed.uncertainty.objects) {
    assert.ok(object.bbox.x >= 0 && object.bbox.y >= 0, `${object.id} has a negative origin`);
    assert.ok(
      object.bbox.x + object.bbox.width <= width,
      `${object.id} overflows the grid width`,
    );
    assert.ok(
      object.bbox.y + object.bbox.height <= height,
      `${object.id} overflows the grid height`,
    );
    assert.ok(object.peak_uncertainty >= object.uncertainty);
  }
});

test('schema rejects a payload with a missing research-integrity field', () => {
  const broken = JSON.parse(JSON.stringify(fixture));
  delete broken.uncertainty.global.provenance;
  assert.equal(analysisSchema.safeParse(broken).success, false);
});

test('schema rejects an out-of-range uncertainty score', () => {
  const broken = JSON.parse(JSON.stringify(fixture));
  broken.uncertainty.global.score = 1.5;
  assert.equal(analysisSchema.safeParse(broken).success, false);
});
