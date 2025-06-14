export default async function handler(req, res) {
  // Set CORS headers
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type");

  // Handle OPTIONS request for CORS preflight
  if (req.method === "OPTIONS") {
    return res.status(200).end();
  }

  try {
    const apiResponse = await fetch(
      "https://ai-schedule-assistant-production.up.railway.app/api/health"
    );
    const data = await apiResponse.json();

    res.status(200).json(data);
  } catch (error) {
    console.error("Health check error:", error);
    res
      .status(500)
      .json({ error: "Failed to fetch from API", details: error.message });
  }
}
