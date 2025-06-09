// lib/main.dart
import 'package:flutter/material.dart';
import 'package:flutter/foundation.dart';
import 'package:calendar_chatbot/pages/home_page.dart';
import 'package:intl/date_symbol_data_local.dart';

// Conditionally import PWA utils only for web
import 'utils/pwa_utils.dart' if (dart.library.io) 'utils/pwa_utils_stub.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  
  await initializeDateFormatting();
  
  // Register PWA features for web
  if (kIsWeb) {
    try {
      PWAUtils.registerServiceWorker();
      await PWAUtils.requestNotificationPermission();
      print('PWA features initialized');
    } catch (e) {
      print('PWA features not available: $e');
    }
  }
  
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Schedule Assistant',
      theme: ThemeData(
        primarySwatch: Colors.blue,
        visualDensity: VisualDensity.adaptivePlatformDensity,
        // PWA-friendly theme
        appBarTheme: AppBarTheme(
          elevation: 0,
          backgroundColor: Colors.blue,
          foregroundColor: Colors.white,
        ),
        elevatedButtonTheme: ElevatedButtonThemeData(
          style: ElevatedButton.styleFrom(
            elevation: 2,
            padding: EdgeInsets.symmetric(horizontal: 24, vertical: 12),
          ),
        ),
      ),
      home: const HomePage(),
      debugShowCheckedModeBanner: false,
    );
  }
}