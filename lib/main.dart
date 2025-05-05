// main.dart
import 'package:flutter/material.dart';
import 'package:flutter/foundation.dart';
import 'package:calendar_chatbot/pages/home_page.dart';
import 'package:calendar_chatbot/database/database_helper.dart';

void main() async {
  // Ensure Flutter is initialized properly
  WidgetsFlutterBinding.ensureInitialized();
  
  // Print diagnostics to help troubleshoot
  print('Starting application...');
  print('Running on ${kIsWeb ? 'Web platform' : 'Mobile platform'}');
  
  // Initialize database with proper error handling
  try {
    await DatabaseHelper.instance.database;
    print('Database initialized successfully');
  } catch (e) {
    print('Error initializing database: $e');
    // Continue anyway - the app will use empty data
  }
  
  // Run the app
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    print('Building MyApp widget');
    return MaterialApp(
      title: 'Calendar & Chatbot',
      theme: ThemeData(
        primarySwatch: Colors.blue,
        visualDensity: VisualDensity.adaptivePlatformDensity,
      ),
      home: const HomePage(),
      debugShowCheckedModeBanner: false,
    );
  }
}