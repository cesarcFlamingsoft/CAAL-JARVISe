import 'package:livekit_client/livekit_client.dart';

import 'config_service.dart';

/// Builds the CAAL `/api/connection-details` URL for [serverUrl].
///
/// [serverUrl] is normalized first (see [normalizeServerUrl]), so any scheme,
/// path, query, fragment or trailing slash the user typed is discarded before
/// the endpoint path is appended.
///
/// Throws an [ArgumentError] when [serverUrl] is empty or unusable.
Uri caalConnectionDetailsUrl(String? serverUrl) {
  final baseUrl = normalizeServerUrl(serverUrl);
  if (baseUrl == null) {
    throw ArgumentError.value(
      serverUrl,
      'serverUrl',
      'Not a usable CAAL server URL',
    );
  }
  return Uri.parse('$baseUrl/api/connection-details');
}

/// Creates an EndpointTokenSource configured for CAAL's API.
///
/// This calls the CAAL frontend's /api/connection-details endpoint
/// which generates LiveKit tokens.
EndpointTokenSource createCaalTokenSource(String serverUrl) {
  return EndpointTokenSource(url: caalConnectionDetailsUrl(serverUrl));
}
