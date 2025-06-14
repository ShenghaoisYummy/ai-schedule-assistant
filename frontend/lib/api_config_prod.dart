class ApiConfig {
  // Production API URL pointing to Railway
  static const String baseUrl = "https://ai-schedule-assistant-production.up.railway.app";

  static const String analyzeEndpoint = "/analyze";
  static const String healthEndpoint = "/health";

  static String get analyzeUrl => baseUrl + analyzeEndpoint;
  static String get healthUrl => baseUrl + healthEndpoint;
}
