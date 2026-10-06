// Builds workflow/ad-account-monitor.json from the Code node sources in src/.
// The JS lives in separate files so it can be read, linted and unit-tested;
// n8n itself only needs the single exported workflow file.
//
//   node scripts/build_workflow.mjs          -> write the workflow file
//   node scripts/build_workflow.mjs --check  -> fail if the file is out of date

import { readFileSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const outputPath = join(root, 'workflow', 'ad-account-monitor.json');
const code = (name) => readFileSync(join(root, 'src', name), 'utf8');

const codeNode = (id, name, file, position) => ({
  id,
  name,
  type: 'n8n-nodes-base.code',
  typeVersion: 2,
  position,
  parameters: { mode: 'runOnceForEachItem', jsCode: code(file) },
});

const booleanIf = (id, name, expression, position) => ({
  id,
  name,
  type: 'n8n-nodes-base.if',
  typeVersion: 2,
  position,
  parameters: {
    conditions: {
      options: { caseSensitive: true, leftValue: '', typeValidation: 'strict' },
      conditions: [
        {
          id: `${id}-condition`,
          leftValue: expression,
          rightValue: '',
          operator: { type: 'boolean', operation: 'true', singleValue: true },
        },
      ],
      combinator: 'and',
    },
    options: {},
  },
});

const respond = (id, name, status, bodyExpression, position) => ({
  id,
  name,
  type: 'n8n-nodes-base.respondToWebhook',
  typeVersion: 1.1,
  position,
  parameters: {
    respondWith: 'json',
    responseBody: bodyExpression,
    options: { responseCode: status },
  },
});

const telegramText = [
  '=🚨 {{ $json.severity.toUpperCase() }} · {{ $json.platform }} · {{ $json.account_name || $json.account_id }}',
  'Event: {{ $json.event_type }}',
  '',
  '{{ $json.summary }}',
  '',
  'Next step: {{ $json.recommended_action }}',
  '',
  'event_id: {{ $json.event_id }} · AI: {{ $json.ai_status }}',
].join('\n');

const nodes = [
  {
    id: 'webhook',
    name: 'Webhook',
    type: 'n8n-nodes-base.webhook',
    typeVersion: 2,
    position: [0, 300],
    webhookId: '6a1f0c4e-3b1d-4f5e-9a7c-2d8e4b6f1a01',
    parameters: {
      httpMethod: 'POST',
      path: 'ad-account-events',
      authentication: 'headerAuth',
      responseMode: 'responseNode',
      options: {},
    },
  },
  codeNode('validate', 'Validate & Normalize', 'validate.js', [220, 300]),
  booleanIf('is-valid', 'Is Valid?', '={{ $json.valid }}', [440, 300]),
  respond(
    'respond-400',
    'Respond 400',
    400,
    "={{ JSON.stringify({ status: 'rejected', errors: $json.errors }) }}",
    [660, 460],
  ),
  respond(
    'respond-202',
    'Respond 202',
    202,
    "={{ JSON.stringify({ status: 'accepted', event_id: $json.event.event_id }) }}",
    [660, 200],
  ),
  codeNode('build-request', 'Build Claude Request', 'build_request.js', [880, 200]),
  {
    id: 'classify',
    name: 'Classify with Claude',
    type: 'n8n-nodes-base.httpRequest',
    typeVersion: 4.2,
    position: [1100, 200],
    // An API error must not stop the workflow: guardrails fall back to rules.
    onError: 'continueRegularOutput',
    retryOnFail: true,
    maxTries: 3,
    waitBetweenTries: 2000,
    parameters: {
      method: 'POST',
      url: 'https://api.anthropic.com/v1/messages',
      authentication: 'genericCredentialType',
      genericAuthType: 'httpHeaderAuth',
      sendHeaders: true,
      headerParameters: {
        parameters: [{ name: 'anthropic-version', value: '2023-06-01' }],
      },
      sendBody: true,
      specifyBody: 'json',
      jsonBody: '={{ JSON.stringify($json.request_body) }}',
      options: { timeout: 60000 },
    },
  },
  codeNode('guardrails', 'Apply Guardrails', 'guardrails.js', [1320, 200]),
  {
    id: 'sheets',
    name: 'Log to Google Sheets',
    type: 'n8n-nodes-base.googleSheets',
    typeVersion: 4.5,
    position: [1540, 100],
    parameters: {
      operation: 'append',
      documentId: { __rl: true, mode: 'url', value: 'REPLACE_WITH_GOOGLE_SHEET_URL' },
      sheetName: { __rl: true, mode: 'name', value: 'events' },
      columns: {
        mappingMode: 'autoMapInputData',
        value: {},
        matchingColumns: [],
        schema: [],
      },
      options: {},
    },
  },
  booleanIf(
    'needs-alert',
    'Needs Alert?',
    "={{ ['critical', 'high'].includes($json.severity) }}",
    [1540, 300],
  ),
  {
    id: 'telegram',
    name: 'Telegram Alert',
    type: 'n8n-nodes-base.telegram',
    typeVersion: 1.2,
    position: [1760, 280],
    parameters: {
      chatId: 'REPLACE_WITH_TELEGRAM_CHAT_ID',
      text: telegramText,
      additionalFields: { appendAttribution: false },
    },
  },
];

const link = (node) => ({ node, type: 'main', index: 0 });

const workflow = {
  name: 'Ad Account Health Monitor',
  nodes,
  connections: {
    Webhook: { main: [[link('Validate & Normalize')]] },
    'Validate & Normalize': { main: [[link('Is Valid?')]] },
    'Is Valid?': { main: [[link('Respond 202')], [link('Respond 400')]] },
    'Respond 202': { main: [[link('Build Claude Request')]] },
    'Build Claude Request': { main: [[link('Classify with Claude')]] },
    'Classify with Claude': { main: [[link('Apply Guardrails')]] },
    'Apply Guardrails': { main: [[link('Log to Google Sheets'), link('Needs Alert?')]] },
    'Needs Alert?': { main: [[link('Telegram Alert')], []] },
  },
  settings: { executionOrder: 'v1' },
  pinData: {},
};

const output = `${JSON.stringify(workflow, null, 2)}\n`;

if (process.argv.includes('--check')) {
  if (readFileSync(outputPath, 'utf8') !== output) {
    console.error('workflow/ad-account-monitor.json is out of date: run `node scripts/build_workflow.mjs`');
    process.exit(1);
  }
  console.log('workflow file is up to date');
} else {
  writeFileSync(outputPath, output);
  console.log(`wrote ${outputPath}`);
}
