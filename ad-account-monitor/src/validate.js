// n8n Code node: "Validate & Normalize" (mode: Run Once for Each Item).
// Checks the webhook payload and turns it into one flat, predictable event.
// The file body runs inside n8n, so a top-level `return` is expected here.

const body = $input.item.json.body ?? {};

const PLATFORMS = ['meta', 'google', 'tiktok', 'snapchat', 'other'];
const MAX_DETAILS_CHARS = 2000;

const errors = [];
const str = (value) => (typeof value === 'string' ? value.trim() : '');

const eventType = str(body.event_type).toLowerCase();
const platform = str(body.platform).toLowerCase();
const accountId = str(body.account_id);
const accountName = str(body.account_name);

if (!eventType) errors.push('event_type is required');
if (!PLATFORMS.includes(platform)) {
  errors.push(`platform must be one of: ${PLATFORMS.join(', ')}`);
}
if (!accountId) errors.push('account_id is required');

let occurredAt = new Date().toISOString();
if (body.occurred_at !== undefined) {
  const parsed = new Date(body.occurred_at);
  if (Number.isNaN(parsed.getTime())) {
    errors.push('occurred_at must be an ISO 8601 date');
  } else {
    occurredAt = parsed.toISOString();
  }
}

const details =
  body.details && typeof body.details === 'object' && !Array.isArray(body.details)
    ? body.details
    : {};
const detailsJson = JSON.stringify(details).slice(0, MAX_DETAILS_CHARS);

// A stable id lets the same event be recognised if a partner sends it twice.
const eventId =
  str(body.event_id) || `${platform}:${accountId}:${eventType}:${occurredAt}`;

return {
  json: {
    valid: errors.length === 0,
    errors,
    event: {
      event_id: eventId,
      received_at: new Date().toISOString(),
      occurred_at: occurredAt,
      platform,
      account_id: accountId,
      account_name: accountName,
      event_type: eventType,
      details_json: detailsJson,
    },
  },
};
