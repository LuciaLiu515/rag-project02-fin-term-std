export function getApiBaseUrl() {
  return process.env.REACT_APP_API_BASE_URL || 'http://127.0.0.1:8009';
}

export async function postJson(path, body) {
  const url = `${getApiBaseUrl()}${path}`;
  const resp = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });

  const text = await resp.text();
  let data;
  try {
    data = text ? JSON.parse(text) : null;
  } catch (e) {
    data = text;
  }

  if (!resp.ok) {
    const msg = typeof data === 'string' ? data : JSON.stringify(data);
    throw new Error(`HTTP ${resp.status}: ${msg}`);
  }

  return data;
}
