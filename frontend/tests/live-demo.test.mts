/**
 * Live demo (/live-demo): contract against real captured backend responses, and the
 * pure helpers that re-threshold score maps in the browser.
 *
 * `fixtures-live-demo.json` is captured from the running backend on the real
 * Kvasir-SEG bundle (status, run at alpha 0.05 seed 0 trimmed to 6 units, validity
 * with 50 splits). Regenerate it with backend's TestClient
 * against /api/demo/{status,run,validity} after building the demo bundle.
 */

import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { test } from 'node:test';
import { fileURLToPath } from 'node:url';

import { demoRunSchema, demoStatusSchema, demoValiditySchema } from '../lib/api/demo-schemas.ts';
import {
  cutoffOfIndex,
  cutoffToPosition,
  decodeScores,
  lesionCentroids,
  lesionCoverage,
  logSampleIndices,
  outline,
  overlayRgba,
  pixelFnr,
  positionToIndex,
  setSize,
  TAU,
} from '../lib/live-demo.ts';

const here = dirname(fileURLToPath(import.meta.url));
const fixture = JSON.parse(readFileSync(join(here, 'fixtures-live-demo.json'), 'utf8')) as {
  status: unknown;
  run: unknown;
  validity: unknown;
};

test('real demo responses satisfy the frontend schemas', () => {
  for (const [name, schema, data] of [
    ['status', demoStatusSchema, fixture.status],
    ['run', demoRunSchema, fixture.run],
    ['validity', demoValiditySchema, fixture.validity],
  ] as const) {
    const result = schema.safeParse(data);
    assert.ok(result.success, `${name}: ${result.success ? '' : JSON.stringify(result.error.issues.slice(0, 3))}`);
  }
});

test('in a real run the joint cutoff is never above pixel CRC, so it finds every polyp pixel CRC finds', () => {
  const run = demoRunSchema.parse(fixture.run);
  const cut = (id: string) => run.methods.find((m) => m.id === id)!.cutoff[0];
  assert.ok(cut('joint') <= cut('pixel'));
  for (const unit of run.units) {
    for (const lesion of unit.lesions) {
      if (lesion.found.pixel) assert.ok(lesion.found.joint);
      // Lemma 2: found exactly when the tau-quantile score clears the cutoff
      assert.equal(lesion.found.pixel, lesion.stat >= unit.cutoffs.pixel - 1e-12);
    }
  }
});

test('decodeScores maps uint16 to [0, 1]', () => {
  const raw = new Uint16Array([0, 65535, 32768]);
  const s = decodeScores(raw.buffer);
  assert.equal(s[0], 0);
  assert.equal(s[1], 1);
  assert.ok(Math.abs(s[2] - 0.5) < 1e-4);
});

test('cutoff grid and scrubber positions round-trip on the grid', () => {
  assert.equal(cutoffOfIndex(0), 1);
  assert.equal(cutoffOfIndex(1000), 0);
  for (const i of [0, 500, 900, 990, 999]) {
    assert.equal(positionToIndex(cutoffToPosition(cutoffOfIndex(i))), i);
  }
  const idx = logSampleIndices(1001, 50);
  assert.ok(idx.every((v, j) => j === 0 || v < idx[j - 1]), 'indices descend (cutoff ascends)');
});

// 4x4 map: lesion 1 = left column pair (4 px), lesion 2 = single pixel.
const W = 4;
const labels = new Uint8Array([1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0]);
// Quantised to the uint16 grid, as every decoded score map is.
const scores = Float32Array.from(
  [0.9, 0.8, 0.1, 0.1, 0.3, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.05, 0.1, 0.1, 0.1, 0.1],
  (v) => Math.round(v * 65535) / 65535,
);

test('lesion coverage, found-at-tau and Lemma 2 agree', () => {
  // cutoff 0.5: lesion 1 has 2/4 in the set (= tau, found), lesion 2 none
  assert.deepEqual(lesionCoverage(scores, labels, 0.5, 2), [0.5, 0]);
  // lesion 1 needs 2 of its 4 pixels, so it is found iff the cutoff is at most its 2nd-highest score, 0.8
  assert.ok(lesionCoverage(scores, labels, 0.8, 2)[0] >= TAU);
  assert.ok(lesionCoverage(scores, labels, 0.81, 2)[0] < TAU);
  assert.deepEqual(lesionCoverage(scores, labels, 0.05, 2), [1, 1]);
});

test('set size and pixel FNR', () => {
  assert.equal(setSize(scores, 0.5), 2);
  assert.equal(setSize(scores, 0.1), 15);
  assert.equal(pixelFnr(scores, labels, 0.5), 3 / 5);
  assert.equal(pixelFnr(scores, labels, 0.05), 0);
});

test('outline, centroids and overlay colours', () => {
  const edge = outline(labels, W, 4);
  assert.equal(edge[0], 1);
  assert.equal(edge[11], 2);
  assert.deepEqual(lesionCentroids(labels, W, 2), [[0.5, 0.5], [3, 2]]);
  const rgba = overlayRgba(scores, labels, W, 4, 0.5, [true, false]);
  assert.deepEqual([...rgba.subarray(0, 4)], [34, 197, 94, 255]); // found outline
  assert.deepEqual([...rgba.subarray(44, 48)], [239, 68, 68, 255]); // missed outline
  assert.equal(rgba[14 * 4 + 3], 0); // background pixel below the cutoff and away from outlines: transparent
});
