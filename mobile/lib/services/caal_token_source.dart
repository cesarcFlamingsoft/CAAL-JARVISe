import 'package:livekit_client/livekit_client.dart';

/// Creates an EndpointTokenSource configured for CAAL's API.
///
/// This calls the CAAL frontend's /api/connection-details endpoint
/// which generates LiveKit tokens.
EndpointTokenSource createCaalTokenSource(String baseUrl) {
  final normalizedBaseUrl = baseUrl.endsWith('/')
      ? baseUrl.substring(0, baseUrl.length - 1)
      : baseUrl;

  return EndpointTokenSource(
    url: Uri.parse('$normalizedBaseUrl/api/connection-details'),
  );
}
