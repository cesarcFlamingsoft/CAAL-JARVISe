import 'dart:async';

import 'package:flutter/material.dart';
import 'package:intl/intl.dart';
import 'package:livekit_client/livekit_client.dart' as sdk;
import 'package:livekit_components/livekit_components.dart' as components;
import 'package:logging/logging.dart';
import 'package:uuid/uuid.dart';
import 'package:wakelock_plus/wakelock_plus.dart';
import '../services/caal_token_source.dart';
import '../services/config_service.dart';

enum AppScreenState { welcome, agent }

enum AgentScreenState { visualizer, transcription }

class AppCtrl extends ChangeNotifier {
  static const uuid = Uuid();
  static final _logger = Logger('AppCtrl');

  /// Stable per-controller client id used to get a dedicated LiveKit room.
  ///
  /// The web frontend now supports client-specific rooms so reconnects from
  /// mobile do not collide with browser sessions or other devices.
  final String _clientId = uuid.v4();

  // Configuration - always stored in normalized form.
  String _serverUrl;

  String get serverUrl => _serverUrl;

  // States
  AppScreenState appScreenState = AppScreenState.welcome;
  AgentScreenState agentScreenState = AgentScreenState.visualizer;

  bool isUserCameEnabled = false;
  bool isScreenshareEnabled = false;

  final messageCtrl = TextEditingController();
  final messageFocusNode = FocusNode();

  // Session objects - can be recreated if native resources are disposed
  sdk.Room _room = sdk.Room(roomOptions: const sdk.RoomOptions(enableVisualizer: true));
  late components.RoomContext _roomContext = components.RoomContext(room: _room);
  late sdk.Session _session = _createSession();

  // Public getters for current instances
  sdk.Room get room => _room;
  components.RoomContext get roomContext => _roomContext;
  sdk.Session get session => _session;

  sdk.Session _createSession() {
    return sdk.Session.fromConfigurableTokenSource(
      createCaalTokenSource(serverUrl).cached(),
      tokenOptions: sdk.TokenRequestOptions(
        participantName: 'Cesar',
        participantIdentity: 'mobile_$_clientId',
      ),
      options: sdk.SessionOptions(
        room: _room,
        // Don't auto-enable camera on startup to avoid Windows native renderer issues
      ),
    );
  }

  /// Tracks if session objects need recreation
  bool _needsRecreation = false;

  /// Key that changes when session objects are recreated, forcing widget rebuild
  int sessionKey = 0;

  /// Mark session objects as needing recreation (called on error)
  void _markNeedsRecreation() {
    _needsRecreation = true;
  }

  /// Recreate all session objects (Room, RoomContext, Session).
  /// Called when native resources have been disposed (e.g., app swiped away).
  Future<void> _recreateSessionObjects() async {
    if (!_needsRecreation || _disposed) return;

    _logger.info('Recreating session objects...');

    // Remove listener from old session
    _session.removeListener(_handleSessionChange);

    // Dispose old objects (ignore errors - they may already be disposed)
    try {
      await _session.dispose();
    } catch (e) {
      _logger.fine('Session dispose error (expected): $e');
    }
    try {
      await _room.dispose();
    } catch (e) {
      _logger.fine('Room dispose error (expected): $e');
    }
    try {
      _roomContext.dispose();
    } catch (e) {
      _logger.fine('RoomContext dispose error (expected): $e');
    }

    // Create fresh objects
    _room = sdk.Room(roomOptions: const sdk.RoomOptions(enableVisualizer: true));
    _roomContext = components.RoomContext(room: _room);
    _session = _createSession();
    _session.addListener(_handleSessionChange);

    _needsRecreation = false;
    sessionKey++; // Increment to force widget rebuild
    _logger.info('Session objects recreated (key: $sessionKey)');
    notifyListeners();
  }

  bool isSendButtonEnabled = false;
  bool isSessionStarting = false;
  bool _hasCleanedUp = false;
  bool _disposed = false;

  /// Subscription to the root logger, cancelled on [dispose] so that a
  /// recreated [AppCtrl] does not stack duplicate log sinks.
  StreamSubscription<LogRecord>? _logSubscription;

  AppCtrl({
    required String serverUrl,
  })  : _serverUrl = normalizeServerUrl(serverUrl) ?? '' {
    final format = DateFormat('HH:mm:ss');
    Logger.root.level = Level.FINE;
    _logSubscription = Logger.root.onRecord.listen((record) {
      debugPrint('${format.format(record.time)}: ${record.message}');
    });

    messageCtrl.addListener(() {
      final newValue = messageCtrl.text.isNotEmpty;
      if (newValue != isSendButtonEnabled) {
        isSendButtonEnabled = newValue;
        notifyListeners();
      }
    });

    session.addListener(_handleSessionChange);
  }

  /// Update server URL config.
  /// Called when user changes settings.
  ///
  /// The incoming value is normalized first, so a differently spelled but
  /// equivalent URL (extra slashes, missing scheme, different casing) is a
  /// no-op and leaves an active session untouched. An unusable value is
  /// ignored rather than tearing the session down.
  Future<void> updateConfig({
    required String serverUrl,
  }) async {
    if (_disposed) return;

    final normalized = normalizeServerUrl(serverUrl);
    if (normalized == null) {
      _logger.warning('Ignoring invalid server URL: "$serverUrl"');
      return;
    }
    if (normalized == _serverUrl) {
      _logger.fine('Server URL unchanged after normalization, keeping session');
      return;
    }

    _logger.info('Updating config - serverUrl: $normalized');
    _serverUrl = normalized;

    // Recreate session with new server URL
    _markNeedsRecreation();
    await _recreateSessionObjects();

    notifyListeners();
  }

  Future<void> _cleanUp() async {
    if (_hasCleanedUp) return;
    _hasCleanedUp = true;

    _session.removeListener(_handleSessionChange);

    // Native resources may already be gone (e.g. the app was swiped away),
    // so tear each one down independently and never let one failure strand
    // the rest.
    for (final step in <(String, Future<void> Function())>[
      ('session', () async => _session.dispose()),
      ('room', () async => _room.dispose()),
      ('roomContext', () async => _roomContext.dispose()),
    ]) {
      try {
        await step.$2();
      } catch (error) {
        _logger.fine('${step.$1} dispose error (expected): $error');
      }
    }

    messageCtrl.dispose();
    messageFocusNode.dispose();
  }

  @override
  void dispose() {
    _disposed = true;
    // Cancel synchronously: the async cleanup below must not let one more
    // log record reach a sink owned by a dead controller.
    unawaited(_logSubscription?.cancel());
    _logSubscription = null;
    unawaited(_cleanUp());
    super.dispose();
  }

  /// Swallows notifications issued after [dispose], which async work
  /// (connection attempts, session recreation) can still emit.
  @override
  void notifyListeners() {
    if (_disposed) return;
    super.notifyListeners();
  }

  void sendMessage() async {
    isSendButtonEnabled = false;

    final text = messageCtrl.text;
    messageCtrl.clear();
    notifyListeners();

    if (text.isEmpty) return;
    await session.sendText(text);
  }

  void toggleUserCamera(components.MediaDeviceContext? deviceCtx) {
    isUserCameEnabled = !isUserCameEnabled;
    notifyListeners();
    
    try {
      if (isUserCameEnabled) {
        deviceCtx?.enableCamera();
        _logger.info('Camera enabled');
      } else {
        deviceCtx?.disableCamera();
        _logger.info('Camera disabled');
      }
    } catch (error) {
      _logger.warning('Could not toggle camera: $error');
      // Revert the state on error
      isUserCameEnabled = !isUserCameEnabled;
      notifyListeners();
    }
  }

  void toggleScreenShare() {
    isScreenshareEnabled = !isScreenshareEnabled;
    notifyListeners();
  }

  void toggleAgentScreenMode() {
    agentScreenState =
        agentScreenState == AgentScreenState.visualizer ? AgentScreenState.transcription : AgentScreenState.visualizer;
    notifyListeners();
  }

  void connect() async {
    if (isSessionStarting) {
      _logger.fine('Connection attempt ignored: session already starting.');
      return;
    }

    _logger.info('Starting session connection…');
    isSessionStarting = true;
    notifyListeners();

    try {
      await _session.start();
      if (_session.connectionState == sdk.ConnectionState.connected) {
        appScreenState = AppScreenState.agent;
        WakelockPlus.enable();
        
        // Enable microphone after connection (already requested in SessionOptions)
        try {
          await _room.localParticipant?.setMicrophoneEnabled(true);
          _logger.info('Microphone enabled');
        } catch (micError) {
          _logger.warning('Could not enable microphone: $micError');
        }
        
        notifyListeners();
      }
    } catch (error, stackTrace) {
      final errorStr = error.toString();

      // Check if this is a native renderer or MediaStreamTrack error
      if (errorStr.contains('disposed') || 
          errorStr.contains('MediaStreamTrack') ||
          errorStr.contains('NativeRenderer') ||
          errorStr.contains('native')) {
        _logger.warning('Native resources issue, marking for recreation...');
        _markNeedsRecreation();
        await _recreateSessionObjects();

        // Retry connection with fresh objects
        try {
          await _session.start();
          if (_session.connectionState == sdk.ConnectionState.connected) {
            appScreenState = AppScreenState.agent;
            WakelockPlus.enable();
            
            // Try enabling microphone on retry
            try {
              await _room.localParticipant?.setMicrophoneEnabled(true);
              _logger.info('Microphone enabled (after retry)');
            } catch (micError) {
              _logger.warning('Could not enable microphone on retry: $micError');
            }
            
            notifyListeners();
            return;
          }
        } catch (retryError, retryStack) {
          _logger.severe('Retry connection error: $retryError', retryError, retryStack);
        }
      } else {
        _logger.severe('Connection error: $error', error, stackTrace);
      }

      appScreenState = AppScreenState.welcome;
      notifyListeners();
    } finally {
      if (isSessionStarting) {
        isSessionStarting = false;
        notifyListeners();
      }
    }
  }

  Future<void> disconnect() async {
    await session.end();
    session.restoreMessageHistory(const []);
    WakelockPlus.disable();
    appScreenState = AppScreenState.welcome;
    agentScreenState = AgentScreenState.visualizer;
    notifyListeners();
  }

  void _handleSessionChange() {
    final sdk.ConnectionState state = _session.connectionState;
    AppScreenState? nextScreen;
    switch (state) {
      case sdk.ConnectionState.connected:
      case sdk.ConnectionState.reconnecting:
        nextScreen = AppScreenState.agent;
        break;
      case sdk.ConnectionState.disconnected:
        nextScreen = AppScreenState.welcome;
        break;
      case sdk.ConnectionState.connecting:
        nextScreen = null;
        break;
    }

    if (nextScreen != null && nextScreen != appScreenState) {
      appScreenState = nextScreen;
      notifyListeners();
    }
  }
}
