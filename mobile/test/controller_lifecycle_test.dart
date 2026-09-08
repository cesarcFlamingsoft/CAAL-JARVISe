import 'package:caal_mobile/app.dart';
import 'package:caal_mobile/controllers/app_ctrl.dart';
import 'package:caal_mobile/controllers/wake_word_state_ctrl.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:livekit_client/livekit_client.dart' as sdk;
import 'package:logging/logging.dart';

sdk.Room _newRoom() =>
    sdk.Room(roomOptions: const sdk.RoomOptions(enableVisualizer: true));

class _FakeDisposable {
  _FakeDisposable(this.key);

  final String key;
  int disposeCount = 0;

  void dispose() => disposeCount++;
}

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  group('AppCtrl server URL handling', () {
    test('normalizes the URL given to the constructor', () {
      final ctrl = AppCtrl(serverUrl: '  HTTP://Caal.Local:3000/api/  ');
      addTearDown(ctrl.dispose);

      expect(ctrl.serverUrl, 'http://caal.local:3000');
    });

    test('does not tear down the session for an equivalent URL', () async {
      final ctrl = AppCtrl(serverUrl: 'http://caal.local:3000');
      addTearDown(ctrl.dispose);

      final room = ctrl.room;
      final session = ctrl.session;
      final roomContext = ctrl.roomContext;
      final sessionKey = ctrl.sessionKey;

      var notified = 0;
      ctrl.addListener(() => notified++);

      for (final equivalent in const [
        'http://caal.local:3000',
        'http://caal.local:3000/',
        'http://caal.local:3000///',
        '  HTTP://Caal.Local:3000  ',
        'caal.local:3000',
        'http://caal.local:3000/api/connection-details',
      ]) {
        await ctrl.updateConfig(serverUrl: equivalent);
      }

      expect(identical(ctrl.room, room), isTrue, reason: 'room was replaced');
      expect(identical(ctrl.session, session), isTrue,
          reason: 'session was replaced');
      expect(identical(ctrl.roomContext, roomContext), isTrue,
          reason: 'roomContext was replaced');
      expect(ctrl.sessionKey, sessionKey);
      expect(notified, 0, reason: 'listeners were notified for a no-op change');
      expect(ctrl.serverUrl, 'http://caal.local:3000');
    });

    test('ignores an invalid URL instead of tearing down the session',
        () async {
      final ctrl = AppCtrl(serverUrl: 'http://caal.local:3000');
      addTearDown(ctrl.dispose);

      final room = ctrl.room;
      final sessionKey = ctrl.sessionKey;

      await ctrl.updateConfig(serverUrl: '');
      await ctrl.updateConfig(serverUrl: '   ');
      await ctrl.updateConfig(serverUrl: 'not a url!!');

      expect(identical(ctrl.room, room), isTrue);
      expect(ctrl.sessionKey, sessionKey);
      expect(ctrl.serverUrl, 'http://caal.local:3000');
    });

    test('recreates the session for a genuinely different URL', () async {
      final ctrl = AppCtrl(serverUrl: 'http://caal.local:3000');
      addTearDown(ctrl.dispose);

      final room = ctrl.room;
      final sessionKey = ctrl.sessionKey;

      await ctrl.updateConfig(serverUrl: 'http://other.local:3000');

      expect(identical(ctrl.room, room), isFalse);
      expect(ctrl.sessionKey, greaterThan(sessionKey));
      expect(ctrl.serverUrl, 'http://other.local:3000');
    });
  });

  group('AppCtrl lifecycle', () {
    test('cancels its root logger subscription on dispose', () async {
      final captured = <String>[];
      final originalDebugPrint = debugPrint;
      debugPrint = (String? message, {int? wrapWidth}) {
        captured.add(message ?? '');
      };
      addTearDown(() => debugPrint = originalDebugPrint);

      final ctrl = AppCtrl(serverUrl: 'http://caal.local:3000');
      Logger('lifecycle-probe').info('before-dispose-marker');
      await pumpEventQueue();
      expect(captured.any((line) => line.contains('before-dispose-marker')),
          isTrue,
          reason: 'AppCtrl should log while alive');

      ctrl.dispose();
      captured.clear();

      Logger('lifecycle-probe').info('after-dispose-marker');
      await pumpEventQueue();

      expect(captured.any((line) => line.contains('after-dispose-marker')),
          isFalse,
          reason: 'log subscription leaked past dispose');
    });

    test('does not notify listeners after dispose', () async {
      final ctrl = AppCtrl(serverUrl: 'http://caal.local:3000');
      ctrl.dispose();

      expect(() => ctrl.notifyListeners(), returnsNormally);
      expect(() => ctrl.updateConfig(serverUrl: 'http://other.local:3000'),
          returnsNormally);
      await pumpEventQueue();
    });
  });

  group('WakeWordStateCtrl', () {
    test('derives its status URL from the shared webhook base URL', () {
      final room = _newRoom();
      final ctrl = WakeWordStateCtrl(
        room: room,
        serverUrl: 'https://caal.example.com:3000/api/',
      );
      addTearDown(ctrl.dispose);

      expect(ctrl.statusUrl,
          Uri.parse('https://caal.example.com:8889/wake-word/status'));
    });

    test('infers http and strips trailing slashes when deriving the base', () {
      final room = _newRoom();
      final ctrl = WakeWordStateCtrl(
        room: room,
        serverUrl: '  192.168.1.100:3000///  ',
      );
      addTearDown(ctrl.dispose);

      expect(ctrl.statusUrl,
          Uri.parse('http://192.168.1.100:8889/wake-word/status'));
    });

    test('has no status URL for an unusable server URL', () {
      final room = _newRoom();
      final ctrl = WakeWordStateCtrl(room: room, serverUrl: '  ');
      addTearDown(ctrl.dispose);

      expect(ctrl.statusUrl, isNull);
    });

    test('is inert after dispose', () async {
      final room = _newRoom();
      final ctrl = WakeWordStateCtrl(
        room: room,
        serverUrl: 'http://caal.local:3000',
      );

      ctrl.dispose();

      expect(ctrl.isDisposed, isTrue);
      expect(ctrl.reset, returnsNormally);
      expect(ctrl.notifyListeners, returnsNormally);
    });
  });

  group('KeyedResource', () {
    test('creates the value once and reuses it for the same key', () {
      final created = <String>[];
      final resource = KeyedResource<String, _FakeDisposable>(
        create: (key) {
          created.add(key);
          return _FakeDisposable(key);
        },
        disposeValue: (value) => value.dispose(),
        scheduleDisposal: (task) => task(),
      );
      addTearDown(resource.dispose);

      final first = resource.of('a');
      expect(identical(resource.of('a'), first), isTrue);
      expect(identical(resource.of('a'), first), isTrue);
      expect(created, ['a']);
    });

    test('replaces and disposes the previous value when the key changes', () {
      final pending = <VoidCallback>[];
      final resource = KeyedResource<String, _FakeDisposable>(
        create: _FakeDisposable.new,
        disposeValue: (value) => value.dispose(),
        scheduleDisposal: pending.add,
      );
      addTearDown(() {
        resource.dispose();
        for (final task in pending) {
          task();
        }
      });

      final first = resource.of('a');
      final second = resource.of('b');

      expect(identical(first, second), isFalse);
      expect(second.key, 'b');

      // Disposal of the replaced value is deferred, never immediate, so the
      // outgoing widget subtree can finish the current frame.
      expect(first.disposeCount, 0);
      for (final task in pending.toList()) {
        task();
      }
      pending.clear();
      expect(first.disposeCount, 1);
      expect(second.disposeCount, 0, reason: 'live value must stay alive');
    });

    test('disposes the current value exactly once on dispose', () {
      final resource = KeyedResource<String, _FakeDisposable>(
        create: _FakeDisposable.new,
        disposeValue: (value) => value.dispose(),
        scheduleDisposal: (task) => task(),
      );

      final value = resource.of('a');
      resource.dispose();
      resource.dispose();

      expect(value.disposeCount, 1);
    });

    test('does not create a value when it was never requested', () {
      var creates = 0;
      final resource = KeyedResource<String, _FakeDisposable>(
        create: (key) {
          creates++;
          return _FakeDisposable(key);
        },
        disposeValue: (value) => value.dispose(),
        scheduleDisposal: (task) => task(),
      );

      resource.dispose();

      expect(creates, 0);
    });
  });
}
