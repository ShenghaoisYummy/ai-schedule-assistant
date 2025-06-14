export default async function handler(req, res) {
  // Set CORS headers
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type");

  // Handle OPTIONS request for CORS preflight
  if (req.method === "OPTIONS") {
    return res.status(200).end();
  }

  if (req.method !== "POST") {
    return res.status(405).json({ error: "Method not allowed" });
  }

  try {
    console.log("Proxying request to backend:", req.body);

    const apiResponse = await fetch(
      "https://ai-schedule-assistant-production.up.railway.app/api/analyze",
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(req.body),
      }
    );

    const data = await apiResponse.json();
    console.log("Backend response:", data);

    res.status(apiResponse.status).json(data);
  } catch (error) {
    console.error("Analyze error:", error);
    res
      .status(500)
      .json({ error: "Failed to fetch from API", details: error.message });
  }
}
