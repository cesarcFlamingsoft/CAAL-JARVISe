import 'dart:convert';

import 'package:flutter/foundation.dart';
import 'package:livekit_client/livekit_client.dart' as sdk;

/// Controller that listens for mcp_error data packets from the backend.
/// Shows connection errors for MCP servers (n8n, Home Assistant, etc).
class ConnectionErrorCtrl extends ChangeNotifier {
  final sdk.Room room;
  late final sdk.EventsListener<sdk.RoomEvent> _listener;

  List<String> _errors = [];
  List<String> get errors => _errors;

  bool get hasErrors => _errors.isNotEmpty;

  ConnectionErrorCtrl({required this.room}) {
    _listener = room.createListener();
    _listener.on<sdk.DataReceivedEvent>(_handleDataReceived);
  }

  void _handleDataReceived(sdk.DataReceivedEvent event) {
    // Only handle mcp_error messages
    if (event.topic != 'mcp_error') return;

    try {
      final jsonString = utf8.decode(event.data);
      final data = jsonDecode(jsonString) as Map<String, dynamic>;

      if (data['type'] == 'mcp_error' && data['errors'] != null) {
        _errors = (data['errors'] as List<dynamic>)
            .map((e) => e.toString())
            .toList();
        notifyListeners();
      }
    } catch (error) {
      debugPrint('[ConnectionErrorCtrl] Failed to parse error: $error');
    }
  }

  /// Clear errors (e.g., after user dismisses them)
  void clearErrors() {
    _errors = [];
    notifyListeners();
  }

  @override
  void dispose() {
    _listener.dispose();
    super.dispose();
  }
}
