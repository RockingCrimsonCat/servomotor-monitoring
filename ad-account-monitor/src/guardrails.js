// n8n Code node: "Apply Guardrails" (mode: Run Once for Each Item).
// Takes Claude's triage answer and decides what to trust:
//   - answer missing / not valid JSON / refused -> rule-based fallback
//   - AI may raise severity, but never drop it below the rule-based floor
// The file body runs inside n8n, so a top-level `return` is expected here.

const event = $('Validate & Normalize').item.json.event;
const response = $input.item.json;

const SEVERITIES = ['low', 'medium', 'high', 'critical'];
const CATEGORIES = ['access', 'billing', 'policy', 'budget', 'tracking', 'other'];
const rank = (severity) => SEVERITIES.indexOf(severity);

// Known event types: minimum severity + what to do if the AI is unavailable.
const RULES = {
  account_disabled: {
    floor: 'critical',
    category: 'access',
    action: 'Pause dependent campaigns and open an appeal with the platform or the account partner.',
  },
  payment_failed: {
    floor: 'critical',
    category: 'billing',
    action: 'Check the payment method and contact the account partner to restore billing.',
  },
  spend_limit_reached: {
    floor: 'high',
    category: 'budget',
    action: 'Confirm the spend cap with the UA manager and raise it if the budget allows.',
  },
  pixel_inactive: {
    floor: 'high',
    category: 'tracking',
    action: 'Check the pixel / CAPI integration and recent site deployments.',
  },
  ad_rejected: {
    floor: 'medium',
    category: 'policy',
    action: 'Review the rejection reason, fix the creative or request a review.',
  },
  budget_changed: {
    floor: 'low',
    category: 'budget',
    action: 'No action needed unless the change was unexpected.',
  },
};
const UNKNOWN_RULE = {
  floor: 'low',
  fallbackSeverity: 'medium',
  category: 'other',
  action: 'Unknown event type: review manually and add a rule for it.',
};
const rule = RULES[event.event_type] ?? UNKNOWN_RULE;

function parseAiAnswer(resp) {
  if (!resp || resp.stop_reason !== 'end_turn' || !Array.isArray(resp.content)) {
    return null;
  }
  const textBlock = resp.content.find((block) => block.type === 'text');
  if (!textBlock) return null;

  const match = textBlock.text.match(/\{[\s\S]*\}/);
  if (!match) return null;

  let answer;
  try {
    answer = JSON.parse(match[0]);
  } catch {
    return null;
  }

  const isText = (value) => typeof value === 'string' && value.trim().length > 0;
  if (
    !SEVERITIES.includes(answer.severity) ||
    !CATEGORIES.includes(answer.category) ||
    !isText(answer.summary) ||
    !isText(answer.recommended_action)
  ) {
    return null;
  }
  return answer;
}

const ai = parseAiAnswer(response);
const accountLabel = event.account_name || event.account_id;

let severity;
let category;
let summary;
let recommendedAction;
let aiStatus;

if (ai === null) {
  severity = rule.fallbackSeverity ?? rule.floor;
  category = rule.category;
  summary = `${event.platform} account ${accountLabel}: ${event.event_type}`;
  recommendedAction = rule.action;
  aiStatus = 'fallback';
} else {
  const overridden = rank(ai.severity) < rank(rule.floor);
  severity = overridden ? rule.floor : ai.severity;
  category = ai.category;
  summary = ai.summary.trim().slice(0, 300);
  recommendedAction = ai.recommended_action.trim().slice(0, 300);
  aiStatus = overridden ? 'overridden' : 'ok';
}

return {
  json: {
    received_at: event.received_at,
    occurred_at: event.occurred_at,
    event_id: event.event_id,
    platform: event.platform,
    account_id: event.account_id,
    account_name: event.account_name,
    event_type: event.event_type,
    severity,
    category,
    summary,
    recommended_action: recommendedAction,
    ai_status: aiStatus,
    details_json: event.details_json,
  },
};
