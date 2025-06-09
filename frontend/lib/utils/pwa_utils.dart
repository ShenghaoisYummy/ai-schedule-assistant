import 'dart:html' as html;
import 'package:flutter/foundation.dart';

class PWAUtils {
  static bool get isPWA {
    if (!kIsWeb) return false;
    try {
      return html.window.matchMedia('(display-mode: standalone)').matches;
    } catch (e) {
      return false;
    }
  }
  
  static bool get isWebApp {
    return kIsWeb;
  }
  
  static void showInstallPrompt() {
    if (kIsWeb) {
      // Trigger custom install prompt through JavaScript
      try {
        html.window.postMessage({'type': 'SHOW_INSTALL_PROMPT'}, '*');
      } catch (e) {
        print('Could not show install prompt: $e');
      }
    }
  }
  
  static void registerServiceWorker() {
    if (!kIsWeb) return;
    
    try {
      // Check if service worker is supported using proper Dart syntax
      final navigator = html.window.navigator;
      if (navigator.serviceWorker != null) {
        navigator.serviceWorker!.register('/sw.js');
        print('Service Worker registered successfully');
      } else {
        print('Service Worker not supported');
      }
    } catch (e) {
      print('Service Worker registration failed: $e');
    }
  }
  
  static Future<void> requestNotificationPermission() async {
    if (!kIsWeb) return;
    
    try {
      final permission = await html.Notification.requestPermission();
      print('Notification permission: $permission');
    } catch (e) {
      print('Notification permission request failed: $e');
    }
  }
  
  static void showNotification(String title, String body) {
    if (!kIsWeb) return;
    
    try {
      if (html.Notification.permission == 'granted') {
        html.Notification(title, body: body, icon: '/icons/Icon-192.png');
      } else {
        print('Notification permission not granted');
      }
    } catch (e) {
      print('Failed to show notification: $e');
    }
  }
  
  // Check if app is running in standalone mode (installed as PWA)
  static bool get isStandalone {
    if (!kIsWeb) return false;
    try {
      return html.window.matchMedia('(display-mode: standalone)').matches;
    } catch (e) {
      return false;
    }
  }
  
  // Get app install status
  static bool get canInstall {
    if (!kIsWeb) return false;
    try {
      // This would be set by the beforeinstallprompt event
      return html.window.localStorage['canInstallPWA'] == 'true';
    } catch (e) {
      return false;
    }
  }
}
