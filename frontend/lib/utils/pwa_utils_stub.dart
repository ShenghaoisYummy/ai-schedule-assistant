// Stub implementation for non-web platforms
class PWAUtils {
  static bool get isPWA => false;
  static bool get isWebApp => false;
  static bool get isStandalone => false;
  static bool get canInstall => false;
  
  static void showInstallPrompt() {}
  static void registerServiceWorker() {}
  static Future<void> requestNotificationPermission() async {}
  static void showNotification(String title, String body) {}
} 