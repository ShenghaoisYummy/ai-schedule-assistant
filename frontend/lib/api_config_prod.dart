class ApiConfig {
  // Production API URL pointing to Railway with CORS proxy
  static const String baseUrl = "https://cors-anywhere.herokuapp.com/https://ai-schedule-assistant-production.up.railway.app/api";

  static const String analyzeEndpoint = "/analyze";
  static const String healthEndpoint = "/health";

  static String get analyzeUrl => baseUrl + analyzeEndpoint;
  static String get healthUrl => baseUrl + healthEndpoint;
}
