class ApiConfig {
  // Direct connection to Railway backend with fallback
  static const String baseUrl = "/api";

  static const String analyzeEndpoint = "/analyze";
  static const String healthEndpoint = "/health";

  static String get analyzeUrl => baseUrl + analyzeEndpoint;
  static String get healthUrl => baseUrl + healthEndpoint;
}
