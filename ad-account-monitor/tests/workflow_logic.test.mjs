// Unit tests for the Code node logic. Run with: node --test tests/
// Each src/*.js file is executed the way n8n runs it: as a function body
// with `$input` and `$` available.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');

function runNode(file, inputJson, otherNodes = {}) {
  const body = readFileSync(join(root, 'src', file), 'utf8');
  const fn = new Function('$input', '$', body);
  const $input = { item: { json: inputJson } };
  const $ = (name) => ({ item: { json: otherNodes[name] } });
  return fn($input, $).json;
}

const validate = (body) => runNode('validate.js', { body });

const validEvent = (overrides = {}) =>
  validate({
    event_type: 'account_disabled',
    platform: 'meta',
    account_id: 'act_1',
    account_name: 'Brand US #3',
    occurred_at: '2026-10-01T10:00:00Z',
    details: { reason: 'UNUSUAL_ACTIVITY' },
    ...overrides,
  }).event;

const claudeAnswer = (answer, stopReason = 'end_turn') => ({
  stop_reason: stopReason,
  content: [
    { type: 'thinking', thinking: '' },
    { type: 'text', text: JSON.stringify(answer) },
  ],
});

const guardrails = (event, response) =>
  runNode('guardrails.js', response, { 'Validate & Normalize': { event } });

// --- Validate & Normalize ---------------------------------------------------

test('accepts a valid event and normalizes it', () => {
  const out = validate({
    event_type: ' Account_Disabled ',
    platform: 'META',
    account_id: 'act_1',
    occurred_at: '2026-10-01T10:00:00Z',
    details: { reason: 'x' },
  });
  assert.equal(out.valid, true);
  assert.deepEqual(out.errors, []);
  assert.equal(out.event.event_type, 'account_disabled');
  assert.equal(out.event.platform, 'meta');
  assert.equal(out.event.occurred_at, '2026-10-01T10:00:00.000Z');
  assert.equal(out.event.event_id, 'meta:act_1:account_disabled:2026-10-01T10:00:00.000Z');
  assert.equal(out.event.details_json, '{"reason":"x"}');
});

test('rejects missing fields, unknown platform and bad dates', () => {
  const out = validate({ platform: 'myspace', occurred_at: 'yesterday' });
  assert.equal(out.valid, false);
  assert.equal(out.errors.length, 4);
});

test('rejects an empty body', () => {
  const out = runNode('validate.js', {});
  assert.equal(out.valid, false);
});

test('keeps an event_id sent by the partner', () => {
  assert.equal(validEvent({ event_id: 'evt-42' }).event_id, 'evt-42');
});

test('truncates very large details', () => {
  const event = validEvent({ details: { blob: 'x'.repeat(10_000) } });
  assert.equal(event.details_json.length, 2000);
});

// --- Build Claude Request ---------------------------------------------------

test('builds a structured-output request that contains the event', () => {
  const event = validEvent();
  const { request_body: body } = runNode('build_request.js', { event });
  assert.equal(body.output_config.format.type, 'json_schema');
  assert.ok(body.messages[0].content.includes('"account_id": "act_1"'));
});

// --- Apply Guardrails -------------------------------------------------------

test('uses the AI answer when it is valid and not below the floor', () => {
  const out = guardrails(
    validEvent({ event_type: 'ad_rejected' }),
    claudeAnswer({
      severity: 'high',
      category: 'policy',
      summary: 'Ad rejected for personal attributes.',
      recommended_action: 'Edit the copy and request a review.',
    }),
  );
  assert.equal(out.severity, 'high');
  assert.equal(out.ai_status, 'ok');
  assert.equal(out.summary, 'Ad rejected for personal attributes.');
});

test('never lets the AI lower severity below the rule floor', () => {
  const out = guardrails(
    validEvent(),
    claudeAnswer({
      severity: 'low',
      category: 'access',
      summary: 'Looks fine.',
      recommended_action: 'Nothing to do.',
    }),
  );
  assert.equal(out.severity, 'critical');
  assert.equal(out.ai_status, 'overridden');
});

test('falls back to rules when the API call failed', () => {
  const out = guardrails(validEvent(), { error: { message: '529 overloaded' } });
  assert.equal(out.severity, 'critical');
  assert.equal(out.category, 'access');
  assert.equal(out.ai_status, 'fallback');
  assert.ok(out.recommended_action.length > 0);
});

test('falls back to rules on refusal or truncated output', () => {
  const answer = { severity: 'high', category: 'access', summary: 's', recommended_action: 'a' };
  for (const stop of ['refusal', 'max_tokens']) {
    const out = guardrails(validEvent(), claudeAnswer(answer, stop));
    assert.equal(out.ai_status, 'fallback', stop);
  }
});

test('falls back to rules when the answer breaks the schema', () => {
  const out = guardrails(
    validEvent({ event_type: 'spend_limit_reached' }),
    claudeAnswer({ severity: 'urgent', category: 'budget', summary: 's', recommended_action: 'a' }),
  );
  assert.equal(out.severity, 'high');
  assert.equal(out.ai_status, 'fallback');
});

test('unknown event types default to medium when the AI is unavailable', () => {
  const out = guardrails(validEvent({ event_type: 'something_new' }), {});
  assert.equal(out.severity, 'medium');
  assert.equal(out.category, 'other');
});

test('output row has exactly the Google Sheets columns', () => {
  const out = guardrails(validEvent(), {});
  assert.deepEqual(Object.keys(out), [
    'received_at', 'occurred_at', 'event_id', 'platform', 'account_id', 'account_name',
    'event_type', 'severity', 'category', 'summary', 'recommended_action', 'ai_status',
    'details_json',
  ]);
});
