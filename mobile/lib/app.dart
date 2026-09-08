import 'package:flutter/material.dart';
import 'package:livekit_client/livekit_client.dart' as sdk;
import 'package:livekit_components/livekit_components.dart' as components;
import 'package:provider/provider.dart';

import 'controllers/app_ctrl.dart';
import 'controllers/audio_filter_ctrl.dart';
import 'controllers/connection_error_ctrl.dart';
import 'controllers/tool_status_ctrl.dart';
import 'controllers/wake_word_state_ctrl.dart';
import 'screens/agent_screen.dart';
import 'screens/setup_screen.dart';
import 'screens/welcome_screen.dart';
import 'services/config_service.dart';
import 'ui/color_pallette.dart' show LKColorPaletteLight, LKColorPaletteDark;
import 'widgets/app_layout_switcher.dart';
import 'widgets/connection_error_banner.dart';
import 'widgets/session_error_banner.dart';

/// Defers disposal of a replaced value; see [KeyedResource.scheduleDisposal].
typedef DisposalScheduler = void Function(VoidCallback task);

void _disposeAfterFrame(VoidCallback task) {
  final binding = WidgetsBinding.instance;
  binding.addPostFrameCallback((_) => task());
  // A post-frame callback only runs if another frame is scheduled.
  binding.scheduleFrame();
}

/// Holds one disposable value per key, so that a value is built once instead
/// of on every rebuild, and the value it replaces is always disposed.
///
/// Controllers that register room event listeners must never be constructed
/// inside a `build`/`Consumer` callback: each rebuild would add another
/// listener that nobody disposes. Build them through a [KeyedResource] keyed
/// on whatever identifies the session instead.
///
/// Disposal of a replaced value is deferred via [scheduleDisposal] (by default
/// to after the current frame) so the outgoing widget subtree can finish
/// building against a still-live value.
class KeyedResource<K, V> {
  KeyedResource({
    required V Function(K key) create,
    required void Function(V value) disposeValue,
    DisposalScheduler? scheduleDisposal,
  })  : _create = create,
        _disposeValue = disposeValue,
        _scheduleDisposal = scheduleDisposal ?? _disposeAfterFrame;

  final V Function(K key) _create;
  final void Function(V value) _disposeValue;
  final DisposalScheduler _scheduleDisposal;

  bool _hasValue = false;
  bool _disposed = false;
  late K _key;
  late V _value;

  /// The value for [key], creating it only when [key] differs from the key the
  /// current value was created with.
  V of(K key) {
    if (_hasValue && _key == key) return _value;

    final hadPrevious = _hasValue;
    final V? previous = hadPrevious ? _value : null;

    _value = _create(key);
    _key = key;
    _hasValue = true;

    if (hadPrevious) {
      _scheduleDisposal(() => _disposeValue(previous as V));
    }
    return _value;
  }

  /// Disposes the current value, if any. Safe to call more than once.
  void dispose() {
    if (_disposed) return;
    _disposed = true;
    if (_hasValue) {
      _hasValue = false;
      _disposeValue(_value);
    }
  }
}

/// Identifies which session the per-session controllers belong to.
typedef _SessionKey = ({int sessionKey, String serverUrl});

/// The controllers that own listeners on the current [sdk.Room].
///
/// They live and die with one session: when [AppCtrl] recreates its room the
/// whole bundle is replaced and the old one disposed.
class _SessionControllers {
  _SessionControllers({required sdk.Room room, required String serverUrl})
      : toolStatus = ToolStatusCtrl(room: room),
        wakeWordState = WakeWordStateCtrl(room: room, serverUrl: serverUrl),
        audioFilter = AudioFilterCtrl(room: room),
        connectionError = ConnectionErrorCtrl(room: room);

  final ToolStatusCtrl toolStatus;
  final WakeWordStateCtrl wakeWordState;
  final AudioFilterCtrl audioFilter;
  final ConnectionErrorCtrl connectionError;

  void dispose() {
    toolStatus.dispose();
    wakeWordState.dispose();
    audioFilter.dispose();
    connectionError.dispose();
  }
}

class JarvisApp extends StatefulWidget {
  final ConfigService configService;

  const JarvisApp({super.key, required this.configService});

  @override
  State<JarvisApp> createState() => _JarvisAppState();
}

class _JarvisAppState extends State<JarvisApp> {
  AppCtrl? _appCtrl;

  late final KeyedResource<_SessionKey, _SessionControllers> _sessionControllers =
      KeyedResource<_SessionKey, _SessionControllers>(
    create: (key) => _SessionControllers(
      room: _appCtrl!.room,
      serverUrl: key.serverUrl,
    ),
    disposeValue: (controllers) => controllers.dispose(),
  );

  @override
  void initState() {
    super.initState();
    _initializeAppCtrl();
  }

  void _initializeAppCtrl() {
    if (widget.configService.isConfigured) {
      _appCtrl = AppCtrl(
        serverUrl: widget.configService.serverUrl,
      );
    }
  }

  void _onConfigured() {
    if (!mounted) return;
    final previous = _appCtrl;
    setState(() {
      _appCtrl = AppCtrl(
        serverUrl: widget.configService.serverUrl,
      );
    });
    previous?.dispose();
  }

  @override
  void dispose() {
    _sessionControllers.dispose();
    _appCtrl?.dispose();
    super.dispose();
  }

  ThemeData buildTheme({required bool isLight}) {
    final colorPallete = isLight ? LKColorPaletteLight() : LKColorPaletteDark();

    return ThemeData(
      useMaterial3: true,
      cardColor: colorPallete.bg2,
      inputDecorationTheme: InputDecorationTheme(
        fillColor: colorPallete.bg2,
        hintStyle: TextStyle(
          color: colorPallete.fg4,
          fontSize: 14,
        ),
      ),
      buttonTheme: ButtonThemeData(
        disabledColor: Colors.red,
        colorScheme: ColorScheme.dark(
          primary: Colors.white,
          secondary: Colors.white,
          surface: const Color(0xFF45997C),
        ),
      ),
      colorScheme: isLight
          ? const ColorScheme.light(
              primary: Colors.black,
              secondary: Colors.black,
              surface: Colors.white,
            )
          : const ColorScheme.dark(
              primary: Colors.white,
              secondary: Colors.white,
              surface: Color(0xFF1A1A1A),
            ),
      textTheme: const TextTheme(
        bodyMedium: TextStyle(
          fontSize: 17,
          fontWeight: FontWeight.w400,
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    // Show setup screen if not configured
    if (_appCtrl == null) {
      return ChangeNotifierProvider.value(
        value: widget.configService,
        child: MaterialApp(
          title: 'JARVIS',
          theme: buildTheme(isLight: true),
          darkTheme: buildTheme(isLight: false),
          themeMode: ThemeMode.dark,
          home: SetupScreen(
            configService: widget.configService,
            onConfigured: _onConfigured,
          ),
        ),
      );
    }

    // Normal app flow with AppCtrl
    return ChangeNotifierProvider.value(
      value: widget.configService,
      child: ChangeNotifierProvider.value(
        value: _appCtrl!,
        child: Consumer<AppCtrl>(
          builder: (ctx, appCtrl, _) {
            // Built once per session, never per rebuild: these controllers own
            // room event listeners and must be disposed when the session is
            // replaced.
            final sessionKey = (
              sessionKey: appCtrl.sessionKey,
              serverUrl: widget.configService.serverUrl,
            );
            final controllers = _sessionControllers.of(sessionKey);

            return MultiProvider(
              key: ValueKey(sessionKey),
              providers: [
                ChangeNotifierProvider<sdk.Session>.value(value: appCtrl.session),
                ChangeNotifierProvider<components.RoomContext>.value(value: appCtrl.roomContext),
                ChangeNotifierProvider<ToolStatusCtrl>.value(value: controllers.toolStatus),
                ChangeNotifierProvider<WakeWordStateCtrl>.value(value: controllers.wakeWordState),
                ChangeNotifierProvider<AudioFilterCtrl>.value(value: controllers.audioFilter),
                ChangeNotifierProvider<ConnectionErrorCtrl>.value(value: controllers.connectionError),
              ],
              child: components.SessionContext(
                session: appCtrl.session,
                child: MaterialApp(
                  title: 'JARVIS',
                  theme: buildTheme(isLight: true),
                  darkTheme: buildTheme(isLight: false),
                  themeMode: ThemeMode.dark,
                  home: Stack(
                    children: [
                      Selector<AppCtrl, AppScreenState>(
                        selector: (ctx, appCtx) => appCtx.appScreenState,
                        builder: (ctx, screen, _) => AppLayoutSwitcher(
                          frontBuilder: (ctx) => const WelcomeScreen(),
                          backBuilder: (ctx) => const AgentScreen(),
                          isFront: screen == AppScreenState.welcome,
                        ),
                      ),
                      const SessionErrorBanner(),
                      const ConnectionErrorBanner(),
                    ],
                  ),
                ),
              ),
            );
          },
        ),
      ),
    );
  }
}
