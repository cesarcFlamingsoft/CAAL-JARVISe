import 'package:flutter_test/flutter_test.dart';
import 'package:shared_preferences/shared_preferences.dart';

import 'package:caal_mobile/app.dart';
import 'package:caal_mobile/services/config_service.dart';

void main() {
  testWidgets('unconfigured JARVIS opens the setup screen', (tester) async {
    SharedPreferences.setMockInitialValues({});
    final configService = ConfigService();
    await configService.init();

    await tester.pumpWidget(JarvisApp(configService: configService));

    expect(find.text('Server URL'), findsOneWidget);
  });
}
