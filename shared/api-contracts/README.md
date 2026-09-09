# Shared API contract

The **backend Pydantic schemas are the source of truth**. `backend/app/schemas/`
defines every payload, and FastAPI publishes the resulting OpenAPI document at
`http://localhost:8000/openapi.json`.

The frontend mirrors that contract in two places:

| File | Purpose |
| --- | --- |
| `frontend/types/api.ts` | Hand-written TypeScript interfaces mirroring the Pydantic models. |
| `frontend/lib/api/schemas.ts` | Zod schemas used to **validate responses at runtime**. |

## Why validate at runtime as well as compile time

TypeScript types vanish at runtime. If the backend contract drifts — a renamed
field, a `null` where a number was expected — plain `fetch().then(r => r.json())`
hands the UI a malformed object and the failure surfaces later as a confusing
render error. The Zod layer turns that into an explicit, located error at the
API boundary, which matters more than usual here because several fields carry
research-integrity meaning (`provenance`, `status`, `assumptions`) and silently
dropping them would misrepresent results.

## Keeping the two in sync

```bash
# 1. Start the backend
cd backend && uvicorn app.main:app --reload

# 2. Inspect the live contract
curl http://localhost:8000/openapi.json | python -m json.tool > /tmp/openapi.json

# 3. Optional: generate types instead of hand-maintaining them
npx openapi-typescript http://localhost:8000/openapi.json -o frontend/types/generated.ts
```

Generation is deliberately *not* wired into the build: the hand-written types
carry documentation comments explaining what each research field means, which a
generator would strip.

## Conventions that cross the boundary

| Convention | Rule |
| --- | --- |
| Field naming | `snake_case` on the wire, both sides. No camelCase transformation layer. |
| `global` | Serialised as `global` via a Pydantic alias (`global_` in Python, since it is a keyword). |
| Dense arrays | Never sent as JSON floats. Masks use RLE; uncertainty maps use PNG endpoints. |
| Coordinates | All bounding boxes and masks are in **inference-resolution** pixels (default 256x256), not original image pixels. |
| Enums | Lower-case for `status`/`provenance`; UPPER_CASE for `RiskLevel` and `ShiftStatus`. |
| Errors | Always `{ code, message, detail }` with an appropriate HTTP status. |
| Nullability | `calibration` and `distribution_shift` are `null` when the corresponding stage was disabled or failed; the UI must handle that, not assume presence. |
