import 'package:caal_mobile/services/caal_token_source.dart';
import 'package:caal_mobile/services/config_service.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  group('normalizeServerUrl', () {
    test('returns null for null, empty and whitespace-only values', () {
      expect(normalizeServerUrl(null), isNull);
      expect(normalizeServerUrl(''), isNull);
      expect(normalizeServerUrl('   '), isNull);
      expect(normalizeServerUrl('\t\n '), isNull);
    });

    test('strips surrounding and interior whitespace', () {
      expect(normalizeServerUrl('  http://192.168.1.100:3000  '),
          'http://192.168.1.100:3000');
      expect(normalizeServerUrl('http://192.168.1.100 :3000'),
          'http://192.168.1.100:3000');
      expect(normalizeServerUrl('\nhttp://caal.local:3000\t'),
          'http://caal.local:3000');
    });

    test('infers http:// when the scheme is missing', () {
      expect(normalizeServerUrl('192.168.1.100:3000'),
          'http://192.168.1.100:3000');
      expect(normalizeServerUrl('caal.local'), 'http://caal.local');
      expect(normalizeServerUrl('localhost:3000'), 'http://localhost:3000');
    });

    test('preserves an explicit https scheme', () {
      expect(normalizeServerUrl('https://caal.example.com'),
          'https://caal.example.com');
      expect(normalizeServerUrl('https://caal.example.com:8443'),
          'https://caal.example.com:8443');
    });

    test('strips paths, queries and fragments', () {
      expect(normalizeServerUrl('http://192.168.1.100:3000/api/connection-details'),
          'http://192.168.1.100:3000');
      expect(normalizeServerUrl('http://caal.local:3000/foo?bar=1#frag'),
          'http://caal.local:3000');
      expect(normalizeServerUrl('caal.local/foo/bar'), 'http://caal.local');
    });

    test('strips every trailing slash', () {
      expect(normalizeServerUrl('http://caal.local:3000/'),
          'http://caal.local:3000');
      expect(normalizeServerUrl('http://caal.local:3000///'),
          'http://caal.local:3000');
      expect(normalizeServerUrl('caal.local///'), 'http://caal.local');
    });

    test('lowercases scheme and host', () {
      expect(normalizeServerUrl('HTTP://CAAL.Local:3000'),
          'http://caal.local:3000');
      expect(normalizeServerUrl('HTTPS://CAAL.Local'), 'https://caal.local');
    });

    test('drops the port when it is the scheme default', () {
      expect(normalizeServerUrl('http://caal.local:80'), 'http://caal.local');
      expect(normalizeServerUrl('https://caal.local:443'), 'https://caal.local');
      expect(normalizeServerUrl('https://caal.local:80'), 'https://caal.local:80');
    });

    test('keeps IPv6 hosts bracketed', () {
      expect(normalizeServerUrl('http://[::1]:3000'), 'http://[::1]:3000');
      expect(normalizeServerUrl('http://[::1]'), 'http://[::1]');
    });

    test('rejects unsupported schemes', () {
      expect(normalizeServerUrl('ftp://caal.local'), isNull);
      expect(normalizeServerUrl('ws://caal.local'), isNull);
      expect(normalizeServerUrl('wss://caal.local:3000'), isNull);
      expect(normalizeServerUrl('file:///tmp/x'), isNull);
    });

    test('rejects values without a usable host', () {
      expect(normalizeServerUrl('http://'), isNull);
      expect(normalizeServerUrl('http:///api'), isNull);
      expect(normalizeServerUrl('://caal.local'), isNull);
      expect(normalizeServerUrl('/just/a/path'), isNull);
      expect(normalizeServerUrl(':3000'), isNull);
    });

    test('rejects hosts containing illegal characters', () {
      expect(normalizeServerUrl('not a url!!'), isNull);
      expect(normalizeServerUrl('http://caal_local!/x'), isNull);
      expect(normalizeServerUrl('http://caal..local'), isNull);
    });

    test('rejects an out-of-range or non-numeric port', () {
      expect(normalizeServerUrl('http://caal.local:99999'), isNull);
      expect(normalizeServerUrl('http://caal.local:abc'), isNull);
      expect(normalizeServerUrl('http://caal.local:-1'), isNull);
    });

    test('is idempotent', () {
      const inputs = [
        '  http://192.168.1.100:3000/api/ ',
        'CAAL.local',
        'https://caal.example.com:8443///',
      ];
      for (final input in inputs) {
        final once = normalizeServerUrl(input);
        expect(once, isNotNull, reason: 'expected $input to normalize');
        expect(normalizeServerUrl(once), once);
      }
    });

    test('maps equivalent spellings onto one identical value', () {
      const equivalent = [
        'http://192.168.1.100:3000',
        'http://192.168.1.100:3000/',
        'http://192.168.1.100:3000///',
        '  http://192.168.1.100:3000  ',
        '192.168.1.100:3000',
        'HTTP://192.168.1.100:3000/api/connection-details',
      ];
      final normalized = equivalent.map(normalizeServerUrl).toSet();
      expect(normalized, {'http://192.168.1.100:3000'});
    });

    test('keeps genuinely different servers distinct', () {
      expect(normalizeServerUrl('http://caal.local:3000'),
          isNot(normalizeServerUrl('http://caal.local:3001')));
      expect(normalizeServerUrl('http://caal.local:3000'),
          isNot(normalizeServerUrl('https://caal.local:3000')));
      expect(normalizeServerUrl('http://caal.local:3000'),
          isNot(normalizeServerUrl('http://other.local:3000')));
    });
  });

  group('isSameServerUrl', () {
    test('treats equivalent spellings as the same server', () {
      expect(isSameServerUrl('http://192.168.1.100:3000/', '192.168.1.100:3000'),
          isTrue);
      expect(
          isSameServerUrl(' HTTP://Caal.Local:3000 ', 'http://caal.local:3000//'),
          isTrue);
    });

    test('treats different servers as different', () {
      expect(isSameServerUrl('http://caal.local:3000', 'http://caal.local:3001'),
          isFalse);
      expect(isSameServerUrl('http://caal.local:3000', 'https://caal.local:3000'),
          isFalse);
    });

    test('treats two invalid values as the same (both unusable)', () {
      expect(isSameServerUrl('', '   '), isTrue);
      expect(isSameServerUrl(null, ''), isTrue);
    });

    test('treats an invalid value and a valid value as different', () {
      expect(isSameServerUrl('', 'http://caal.local:3000'), isFalse);
      expect(isSameServerUrl('http://caal.local:3000', 'garbage!!'), isFalse);
    });
  });

  group('webhookBaseUrl', () {
    test('forces port 8889 and preserves the http scheme', () {
      expect(webhookBaseUrl('http://192.168.1.100:3000'),
          'http://192.168.1.100:8889');
      expect(webhookBaseUrl('192.168.1.100'), 'http://192.168.1.100:8889');
    });

    test('preserves the https scheme', () {
      expect(webhookBaseUrl('https://caal.example.com'),
          'https://caal.example.com:8889');
      expect(webhookBaseUrl('https://caal.example.com:3000/api/'),
          'https://caal.example.com:8889');
    });

    test('normalizes the input before deriving the webhook base', () {
      expect(webhookBaseUrl('  HTTPS://CAAL.Example.com:3000/api/x?y=1#z  '),
          'https://caal.example.com:8889');
      expect(webhookBaseUrl('http://caal.local:3000///'),
          'http://caal.local:8889');
    });

    test('keeps IPv6 hosts bracketed', () {
      expect(webhookBaseUrl('http://[::1]:3000'), 'http://[::1]:8889');
    });

    test('returns null for invalid or empty input', () {
      expect(webhookBaseUrl(null), isNull);
      expect(webhookBaseUrl(''), isNull);
      expect(webhookBaseUrl('   '), isNull);
      expect(webhookBaseUrl('ftp://caal.local'), isNull);
      expect(webhookBaseUrl('not a url!!'), isNull);
    });

    test('is stable across equivalent server URL spellings', () {
      const equivalent = [
        'http://192.168.1.100:3000',
        'http://192.168.1.100:3000/',
        '192.168.1.100:3000',
        'HTTP://192.168.1.100:3000/api/connection-details',
      ];
      expect(equivalent.map(webhookBaseUrl).toSet(),
          {'http://192.168.1.100:8889'});
    });
  });

  group('caalConnectionDetailsUrl', () {
    test('appends the connection-details path to the normalized base', () {
      expect(caalConnectionDetailsUrl('http://192.168.1.100:3000'),
          Uri.parse('http://192.168.1.100:3000/api/connection-details'));
    });

    test('does not double up slashes or duplicate the path', () {
      expect(caalConnectionDetailsUrl('http://192.168.1.100:3000///'),
          Uri.parse('http://192.168.1.100:3000/api/connection-details'));
      expect(
          caalConnectionDetailsUrl(
              'http://192.168.1.100:3000/api/connection-details'),
          Uri.parse('http://192.168.1.100:3000/api/connection-details'));
    });

    test('infers http:// and preserves https', () {
      expect(caalConnectionDetailsUrl('192.168.1.100:3000'),
          Uri.parse('http://192.168.1.100:3000/api/connection-details'));
      expect(caalConnectionDetailsUrl('https://caal.example.com'),
          Uri.parse('https://caal.example.com/api/connection-details'));
    });

    test('throws ArgumentError for an unusable server URL', () {
      expect(() => caalConnectionDetailsUrl(''), throwsArgumentError);
      expect(() => caalConnectionDetailsUrl('   '), throwsArgumentError);
      expect(() => caalConnectionDetailsUrl('ftp://caal.local'),
          throwsArgumentError);
      expect(() => caalConnectionDetailsUrl('not a url!!'), throwsArgumentError);
    });
  });
}
