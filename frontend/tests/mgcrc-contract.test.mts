/**
 * Contract test for GET /api/research/mgcrc.
 *
 * `fixtures-mgcrc.json` is a real response captured from the FastAPI app:
 *   cd backend && python -c "from fastapi.testclient import TestClient; from app.main import app; \
 *     open('../frontend/tests/fixtures-mgcrc.json','w').write(TestClient(app).get('/api/research/mgcrc').text)"
 */

import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { test } from 'node:test';
import { fileURLToPath } from 'node:url';

import { mgcrcSummarySchema } from '../lib/api/schemas.ts';

const here = dirname(fileURLToPath(import.meta.url));
const fixture = JSON.parse(readFileSync(join(here, 'fixtures-mgcrc.json'), 'utf8')) as unknown;

test('a real /api/research/mgcrc response satisfies the frontend schema', () => {
  const result = mgcrcSummarySchema.safeParse(fixture);
  if (!result.success) {
    assert.fail(result.error.issues.map((i) => `${i.path.join('.')}: ${i.message}`).join('\n'));
  }
});

test('every block carries computed verdicts and the reference method', () => {
  const parsed = mgcrcSummarySchema.parse(fixture);
  assert.equal(parsed.provenance, 'computed');
  for (const block of parsed.blocks) {
    assert.ok(['KILL', 'INCONCLUSIVE', 'PROCEED'].includes(block.verdicts.H1));
    assert.ok(block.rows.some((r) => r.method === 'union_threshold'));
  }
});
