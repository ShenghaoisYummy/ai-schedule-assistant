class ApiConfig {
  // In production, the API is on the same domain, so the base URL is just a relative path.
  static const String baseUrl = "/api";

  static const String analyzeEndpoint = "/analyze";
  static const String healthEndpoint = "/health";

  static String get analyzeUrl => baseUrl + analyzeEndpoint;
  static String get healthUrl => baseUrl + healthEndpoint;
}
