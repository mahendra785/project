/**
 * Tests for the client-side uncertainty maths.
 *
 * Run with `npm test` (Node's built-in runner + native TypeScript stripping —
 * no test framework dependency).
 */

import assert from 'node:assert/strict';
import { test } from 'node:test';

import {
  decodeGrayscale,
  makeMatrix,
  matrixHistogram,
  matrixStats,
  uncertaintyColour,
} from '../lib/uncertainty.ts';

test('uncertaintyColour matches the ramp endpoints exactly', () => {
  assert.deepEqual(uncertaintyColour(0), [12, 24, 58]);
  assert.deepEqual(uncertaintyColour(1), [206, 52, 48]);
});

test('uncertaintyColour clamps out-of-range input', () => {
  assert.deepEqual(uncertaintyColour(-5), [12, 24, 58]);
  assert.deepEqual(uncertaintyColour(9), [206, 52, 48]);
});

test('uncertaintyColour interpolates between stops', () => {
  // Midway between stop 0 (12,24,58) and stop 0.25 (24,96,160).
  assert.deepEqual(uncertaintyColour(0.125), [18, 60, 109]);
});

test('uncertaintyColour is continuous — no jumps between adjacent values', () => {
  let previous = uncertaintyColour(0);
  for (let v = 0.01; v <= 1; v += 0.01) {
    const current = uncertaintyColour(v);
    for (let c = 0; c < 3; c += 1) {
      assert.ok(
        Math.abs(current[c] - previous[c]) < 12,
        `channel ${c} jumped at v=${v.toFixed(2)}`,
      );
    }
    previous = current;
  }
});

test('decodeGrayscale inverts the backend 8-bit quantisation', () => {
  // Two pixels: byte 0 -> 0.0, byte 255 -> 1.0, byte 128 -> ~0.502.
  const data = new Uint8ClampedArray([
    0, 0, 0, 255, 255, 255, 255, 255, 128, 128, 128, 255, 64, 64, 64, 255,
  ]);
  const matrix = decodeGrayscale(data, 2, 2);

  assert.equal(matrix.width, 2);
  assert.equal(matrix.height, 2);
  assert.equal(matrix.at(0, 0), 0);
  assert.equal(matrix.at(1, 0), 1);
  assert.ok(Math.abs(matrix.at(0, 1) - 128 / 255) < 1e-6);
});

test('decode round-trip error stays within the quantisation bound', () => {
  // Quantise a set of known values the way the backend does, decode, compare.
  const originals = [0, 0.137, 0.5, 0.618, 0.913, 1];
  const bytes: number[] = [];
  for (const value of originals) {
    const q = Math.round(value * 255);
    bytes.push(q, q, q, 255);
  }
  const matrix = decodeGrayscale(new Uint8ClampedArray(bytes), originals.length, 1);

  originals.forEach((original, index) => {
    // Maximum possible error from rounding to 1/255 steps is 1/510.
    assert.ok(
      // +epsilon covers Float32Array storage precision on top of the 1/510 bound.
      Math.abs(matrix.values[index] - original) <= 1 / 510 + 1e-6,
      `value ${original} decoded to ${matrix.values[index]}`,
    );
  });
});

test('decodeGrayscale rejects degenerate dimensions', () => {
  // Regression: ImageBitmap.close() detaches the bitmap, after which its width
  // and height report 0. Reading them post-close produced a 0x0 matrix that
  // surfaced far away as "createImageData: source width is zero".
  const data = new Uint8ClampedArray([0, 0, 0, 255]);

  assert.throws(() => decodeGrayscale(data, 0, 0), /Invalid uncertainty map dimensions/);
  assert.throws(() => decodeGrayscale(data, 4, 0), /Invalid uncertainty map dimensions/);
  assert.throws(() => decodeGrayscale(data, -2, 2), /Invalid uncertainty map dimensions/);
  assert.throws(() => decodeGrayscale(data, 1.5, 2), /Invalid uncertainty map dimensions/);
});

test('decodeGrayscale rejects pixel data that is too short for the dimensions', () => {
  const data = new Uint8ClampedArray(4 * 3); // only 3 pixels
  assert.throws(() => decodeGrayscale(data, 4, 4), /too short/);
});

test('matrixHistogram bins every pixel exactly once', () => {
  const values = new Float32Array([0, 0.1, 0.25, 0.5, 0.75, 0.99, 1]);
  const counts = matrixHistogram(makeMatrix(values, 7, 1), 10);

  assert.equal(counts.length, 10);
  assert.equal(
    counts.reduce((a, b) => a + b, 0),
    values.length,
  );
  // A value of exactly 1.0 must land in the last bin, not overflow past it.
  assert.equal(counts[9], 2); // 0.99 and 1.0
});

test('matrixStats computes mean, max and the above-threshold fraction', () => {
  const values = new Float32Array([0, 0.25, 0.5, 0.75, 1]);
  const stats = matrixStats(makeMatrix(values, 5, 1), 0.5);

  assert.equal(stats.count, 5);
  assert.ok(Math.abs(stats.mean - 0.5) < 1e-6);
  assert.equal(stats.max, 1);
  // 0.5, 0.75 and 1.0 are >= 0.5 -> 3/5.
  assert.ok(Math.abs(stats.aboveFraction - 0.6) < 1e-6);
});

test('matrixStats threshold is inclusive at the boundary', () => {
  const stats = matrixStats(makeMatrix(new Float32Array([0.5]), 1, 1), 0.5);
  assert.equal(stats.aboveFraction, 1);
});

test('matrix.at returns 0 outside bounds rather than undefined', () => {
  const matrix = makeMatrix(new Float32Array([0.4]), 1, 1);
  assert.equal(matrix.at(5, 5), 0);
});
