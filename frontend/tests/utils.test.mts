/** Formatting helpers — the em-dash contract matters for research integrity. */

import assert from 'node:assert/strict';
import { test } from 'node:test';

import { bytes, humanise, ms, num, pct } from '../lib/utils.ts';

test('num and pct render an em dash for missing values, never a zero', () => {
  // Rendering "0.000" for a value that was never measured would be a
  // fabricated result, so the helpers must fail visibly instead.
  assert.equal(num(null), '—');
  assert.equal(num(undefined), '—');
  assert.equal(num(Number.NaN), '—');
  assert.equal(pct(null), '—');
  assert.equal(pct(undefined), '—');
});

test('num respects the requested precision', () => {
  assert.equal(num(0.123456, 3), '0.123');
  assert.equal(num(0.123456, 5), '0.12346');
  assert.equal(num(0), '0.000');
});

test('pct converts to a percentage', () => {
  assert.equal(pct(0.5), '50.0%');
  assert.equal(pct(0.12345, 2), '12.35%');
  assert.equal(pct(1), '100.0%');
});

test('bytes picks a sensible unit', () => {
  assert.equal(bytes(512), '512 B');
  assert.equal(bytes(2048), '2.0 KB');
  assert.equal(bytes(5 * 1024 * 1024), '5.00 MB');
});

test('ms switches to seconds past a thousand', () => {
  assert.equal(ms(250), '250 ms');
  assert.equal(ms(1500), '1.50 s');
});

test('humanise turns identifiers into labels', () => {
  assert.equal(humanise('SEVERE_SHIFT'), 'Severe Shift');
  assert.equal(humanise('conformal_risk_control'), 'Conformal Risk Control');
  assert.equal(humanise('normalised-predictive-entropy'), 'Normalised Predictive Entropy');
});
