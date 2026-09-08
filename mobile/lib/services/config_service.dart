import 'package:flutter/foundation.dart';
import 'package:shared_preferences/shared_preferences.dart';

/// Port the CAAL webhook / control API listens on.
///
/// The webhook API always runs on this port, regardless of which port the
/// frontend (the configured server URL) is served from.
const int caalWebhookPort = 8889;

/// Host labels: alphanumerics plus `.`, `-` and `_`, never doubled dots and
/// never leading/trailing punctuation.
final RegExp _regNameHost = RegExp(r'^[a-z0-9]([a-z0-9\-_]|\.(?!\.))*[a-z0-9]$|^[a-z0-9]$');

/// Bracket-less IPv6 literal, as returned by [Uri.host].
final RegExp _ipv6Host = RegExp(r'^[0-9a-f:.]+$');

/// A server URL reduced to the only three parts CAAL cares about.
typedef _ServerUrlParts = ({String scheme, String host, int? port});

/// Parses a user-supplied server URL into its normalized parts.
///
/// Returns `null` when [raw] cannot be understood as an http(s) origin.
_ServerUrlParts? _parseServerUrl(String? raw) {
  if (raw == null) return null;

  // URLs can never contain whitespace, so any whitespace (leading, trailing or
  // pasted into the middle by a keyboard/autocorrect) is safe to remove.
  final compact = raw.replaceAll(RegExp(r'\s'), '');
  if (compact.isEmpty) return null;

  // Infer http:// when the user typed a bare host such as "192.168.1.100:3000".
  final withScheme = RegExp(r'^[a-zA-Z][a-zA-Z0-9+.\-]*://').hasMatch(compact)
      ? compact
      : 'http://$compact';

  final uri = Uri.tryParse(withScheme);
  if (uri == null) return null;

  final scheme = uri.scheme.toLowerCase();
  if (scheme != 'http' && scheme != 'https') return null;

  final host = uri.host.toLowerCase();
  if (host.isEmpty) return null;
  final isIpv6 = host.contains(':');
  if (isIpv6 ? !_ipv6Host.hasMatch(host) : !_regNameHost.hasMatch(host)) {
    return null;
  }

  int? port;
  if (uri.hasPort) {
    port = uri.port;
    if (port < 1 || port > 65535) return null;
    // Drop the port when it is the scheme default so that e.g.
    // "http://host:80" and "http://host" are treated as the same server.
    if (port == (scheme == 'https' ? 443 : 80)) port = null;
  }

  return (scheme: scheme, host: host, port: port);
}

/// Renders [parts] back into an origin string, optionally overriding the port.
String _renderOrigin(_ServerUrlParts parts, {int? forcePort}) {
  final host = parts.host.contains(':') ? '[${parts.host}]' : parts.host;
  final port = forcePort ?? parts.port;
  return '${parts.scheme}://$host${port == null ? '' : ':$port'}';
}

/// Normalizes a user-supplied CAAL server URL to a bare origin.
///
/// Whitespace is removed, a missing scheme becomes `http://`, the scheme and
/// host are lowercased, default ports are dropped, and any path, query,
/// fragment or trailing slash is stripped. Returns `null` when the value is
/// empty or cannot be used as an http(s) server address.
///
/// The result is stable and idempotent, so two spellings of the same server
/// normalize to the exact same string and can be compared with `==`.
String? normalizeServerUrl(String? raw) {
  final parts = _parseServerUrl(raw);
  return parts == null ? null : _renderOrigin(parts);
}

/// Whether [a] and [b] point at the same server once normalized.
///
/// Two values that are both unusable count as the same (both mean
/// "not configured").
bool isSameServerUrl(String? a, String? b) =>
    normalizeServerUrl(a) == normalizeServerUrl(b);

/// The single source of truth for the CAAL webhook API base URL.
///
/// Preserves the server URL's http/https scheme and forces [caalWebhookPort].
/// Returns `null` when [rawServerUrl] is empty or invalid.
String? webhookBaseUrl(String? rawServerUrl) {
  final parts = _parseServerUrl(rawServerUrl);
  return parts == null ? null : _renderOrigin(parts, forcePort: caalWebhookPort);
}

/// Service for managing app configuration stored in SharedPreferences.
///
/// Handles connection settings (server URL) that persist between app launches.
/// Values are normalized with [normalizeServerUrl] on both read and write, so
/// every consumer sees the same canonical spelling.
class ConfigService extends ChangeNotifier {
  static const _keyServerUrl = 'caal_server_url';

  SharedPreferences? _prefs;

  /// Initialize SharedPreferences. Must be called before accessing any values.
  Future<void> init() async {
    _prefs = await SharedPreferences.getInstance();
  }

  /// Whether the app has been configured with a valid server URL.
  bool get isConfigured => serverUrl.isNotEmpty;

  /// The normalized CAAL server URL (e.g. `http://192.168.1.100:3000`),
  /// or an empty string when unset or unusable.
  String get serverUrl =>
      normalizeServerUrl(_prefs?.getString(_keyServerUrl)) ?? '';

  /// The webhook API base URL derived from [serverUrl], or `''` when unset.
  String get webhookUrl => webhookBaseUrl(serverUrl) ?? '';

  /// Save the server URL after normalizing it.
  ///
  /// Returns `false` (and stores nothing) when [url] is not a usable server
  /// address. Does not notify listeners when the normalized value is unchanged.
  Future<bool> setServerUrl(String url) async {
    final normalized = normalizeServerUrl(url);
    if (normalized == null) return false;
    if (normalized == serverUrl) return true;

    await _prefs?.setString(_keyServerUrl, normalized);
    notifyListeners();
    return true;
  }

  /// Clear all configuration (for testing or reset).
  Future<void> clear() async {
    await _prefs?.remove(_keyServerUrl);
    notifyListeners();
  }
}
