import assert from 'node:assert/strict';
import { describe, it } from 'node:test';
import {
  browserLocalModel,
  browserModels,
  describeEndpointCode,
  isModelName,
  normalizeEndpoint,
} from './endpoint.ts';

const OK = (raw: string) => {
  const result = normalizeEndpoint(raw);
  assert.equal(result.ok, true, raw);
  return result.ok ? result.endpoint : '';
};

const CODE = (raw: string) => {
  const result = normalizeEndpoint(raw);
  assert.equal(result.ok, false, raw);
  return result.ok ? '' : result.code;
};

describe('normalizeEndpoint', () => {
  it('accepts a local endpoint and gives back one stored form', () => {
    assert.equal(OK('http://localhost:11434'), 'http://localhost:11434');
    assert.equal(OK('  http://localhost:11434/  '), 'http://localhost:11434');
    assert.equal(OK('HTTP://LocalHost:11434'), 'http://localhost:11434');
    assert.equal(OK('http://host.docker.internal:11434'), 'http://host.docker.internal:11434');
    assert.equal(OK('http://192.168.1.50:11434'), 'http://192.168.1.50:11434');
    assert.equal(OK('http://10.0.0.12:11434'), 'http://10.0.0.12:11434');
    assert.equal(OK('http://172.16.5.4:8080'), 'http://172.16.5.4:8080');
    assert.equal(OK('http://127.0.0.1:11434'), 'http://127.0.0.1:11434');
    assert.equal(OK('http://[::1]:11434'), 'http://[::1]:11434');
    assert.equal(OK('http://[fd00::1]:11434'), 'http://[fd00::1]:11434');
  });

  it('refuses anything that is not a plain local http endpoint', () => {
    assert.equal(CODE('https://localhost:11434'), 'scheme_not_http');
    assert.equal(CODE('localhost:11434'), 'scheme_not_http');
    assert.equal(CODE('file:///etc/passwd'), 'scheme_not_http');
    assert.equal(CODE('http://user:secret@localhost:11434'), 'credentials_not_allowed');
    assert.equal(CODE('http://localhost:11434/api/tags'), 'path_not_allowed');
    assert.equal(CODE('http://localhost:11434/?x=1'), 'path_not_allowed');
    assert.equal(CODE('http://localhost:11434#frag'), 'path_not_allowed');
    assert.equal(CODE('http://localhost'), 'port_required');
    assert.equal(CODE('http://192.168.1.50'), 'port_required');
    assert.equal(CODE('http://localhost:abc'), 'invalid_port');
    assert.equal(CODE('http://localhost:0'), 'invalid_port');
    assert.equal(CODE('http://localhost:65536'), 'invalid_port');
  });

  it('refuses a public name or address however it is dressed up', () => {
    assert.equal(CODE('http://ollama.example.com:11434'), 'host_not_local');
    assert.equal(CODE('http://8.8.8.8:11434'), 'host_not_local');
    assert.equal(CODE('http://169.254.169.254:80'), 'host_not_local');
    assert.equal(CODE('http://172.32.0.1:11434'), 'host_not_local');
    assert.equal(CODE('http://100.64.0.1:11434'), 'host_not_local');
    assert.equal(CODE('http://localhost.attacker.example:11434'), 'host_not_local');
    assert.equal(CODE('http://[2606:4700::1111]:11434'), 'host_not_local');
    assert.equal(CODE('http://[::ffff:8.8.8.8]:11434'), 'host_not_local');
  });

  it('refuses nonsense without throwing', () => {
    assert.equal(CODE(''), 'invalid_endpoint');
    assert.equal(CODE('   '), 'invalid_endpoint');
    assert.equal(CODE('http://'), 'invalid_endpoint');
    assert.equal(CODE('http://l\u03bfcalhost:11434'), 'invalid_endpoint');
    assert.equal(CODE('http://' + 'a'.repeat(400) + ':11434'), 'invalid_endpoint');
    assert.equal(CODE('http://localhost:11434\nX: 1'), 'invalid_endpoint');
    assert.equal(normalizeEndpoint(undefined as unknown as string).ok, false);
    assert.equal(normalizeEndpoint(11434 as unknown as string).ok, false);
  });
});

describe('isModelName', () => {
  it('accepts an ollama model name and refuses anything else', () => {
    assert.equal(isModelName('qwen3:8b'), true);
    assert.equal(isModelName('mistral-small3.2:latest'), true);
    assert.equal(isModelName('library/llama3.2:3b'), true);
    assert.equal(isModelName('bad model'), false);
    assert.equal(isModelName(''), false);
    assert.equal(isModelName('x'.repeat(200)), false);
    assert.equal(isModelName(null), false);
    assert.equal(isModelName({ name: 'qwen3:8b' }), false);
  });
});

describe('browserLocalModel', () => {
  const PAYLOAD = {
    endpoint: 'http://192.168.1.50:11434',
    model: 'qwen3:8b',
    local_only: true,
    applies_to: 'new_sessions',
    routing: { primary: 'ollama', escalation: 'hermes', coding: 'hermes_delegation' },
  };

  it('reduces a backend answer to what the browser may see', () => {
    assert.deepEqual(browserLocalModel(PAYLOAD), {
      endpoint: 'http://192.168.1.50:11434',
      model: 'qwen3:8b',
      localOnly: true,
      appliesTo: 'new_sessions',
      routing: { primary: 'ollama', escalation: 'hermes', coding: 'hermes_delegation' },
    });
  });

  it('drops anything that is not the shape it claims to be', () => {
    assert.equal(browserLocalModel(null), null);
    assert.equal(browserLocalModel('nope'), null);
    assert.equal(browserLocalModel({ ...PAYLOAD, endpoint: 5 }), null);
    assert.equal(browserLocalModel({ ...PAYLOAD, endpoint: 'https://evil.example.com' }), null);
    assert.equal(browserLocalModel({ ...PAYLOAD, model: 'bad model' }), null);
  });

  it('carries no field the backend did not promise', () => {
    const view = browserLocalModel({ ...PAYLOAD, hermes_api_key: 'sk-secret' });
    assert.ok(view);
    assert.equal(JSON.stringify(view).includes('sk-secret'), false);
  });

  it('accepts an empty model, which is what an unconfigured JARVIS has', () => {
    const view = browserLocalModel({ ...PAYLOAD, model: '' });
    assert.equal(view?.model, '');
  });
});

describe('browserModels', () => {
  it('keeps plausible names, drops the rest, and caps the list', () => {
    const rows = Array.from({ length: 200 }, (_, index) => 'model' + index + ':v1');
    const view = browserModels({
      endpoint: 'http://10.0.0.12:11434',
      models: [...rows, 'bad name', '', 7, null, 'qwen3:8b', 'qwen3:8b'],
    });
    assert.equal(view?.endpoint, 'http://10.0.0.12:11434');
    assert.equal(view?.models.length, 100);
    assert.equal(
      view?.models.every((name) => !name.includes(' ')),
      true
    );
  });

  it('refuses an answer that does not name a local endpoint', () => {
    assert.equal(browserModels({ endpoint: 'http://8.8.8.8:11434', models: [] }), null);
    assert.equal(browserModels({ models: ['qwen3:8b'] }), null);
    assert.equal(browserModels(null), null);
  });

  it('passes an empty ollama through as an empty list', () => {
    assert.deepEqual(browserModels({ endpoint: 'http://localhost:11434', models: [] }), {
      endpoint: 'http://localhost:11434',
      models: [],
    });
  });
});

describe('describeEndpointCode', () => {
  it('has a sentence for every code the backend can send', () => {
    const codes = [
      'invalid_endpoint',
      'scheme_not_http',
      'credentials_not_allowed',
      'path_not_allowed',
      'port_required',
      'invalid_port',
      'host_not_local',
      'unresolvable',
      'invalid_model',
      'unreachable',
      'timeout',
      'upstream_error',
      'unexpected_response',
      'forbidden',
      'unauthorized',
      'rate_limited',
      'backend_unavailable',
    ];
    for (const code of codes) {
      const sentence = describeEndpointCode(code);
      assert.ok(sentence.length > 10, code);
      assert.equal(sentence.includes(code), false);
    }
  });

  it('never echoes an unknown code back at the operator', () => {
    const sentence = describeEndpointCode('<img src=x onerror=alert(1)>');
    assert.equal(sentence.includes('<img'), false);
    assert.equal(sentence, describeEndpointCode('whatever_else'));
  });
});
