// Run in the developer console on https://jarvis.mexcantech.io after signing
// in as your administrator. Uses the real browser session and normal CSRF flow.
// To roll back, change only the following provider to 'kokoro'.
(async () => {
  const provider = 'qwen-trial';
  const identityResponse = await fetch('/api/auth/me', { cache: 'no-store' });
  const identity = await identityResponse.json();
  if (!identityResponse.ok || !identity.authenticated || identity.user?.role !== 'admin'
      || identity.mustChangePassword || !identity.csrfToken) {
    throw new Error('Sign in as administrator and finish any required password change first.');
  }
  const changed = await fetch('/api/tts', {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json', 'X-CAAL-CSRF': identity.csrfToken },
    body: JSON.stringify({ provider }),
  });
  if (!changed.ok) throw new Error(`TTS selection failed: HTTP ${changed.status}`);
  const read = await fetch('/api/tts', { cache: 'no-store' });
  const state = await read.json();
  if (!read.ok || state.provider !== provider
      || (provider === 'qwen-trial' && !state.qwen_configured)) {
    throw new Error('TTS readback failed. Do not start a test yet.');
  }
  console.log({ provider: state.provider, qwen_configured: state.qwen_configured,
    qwen_voice: state.qwen_voice, applies_to: state.applies_to });
})();
