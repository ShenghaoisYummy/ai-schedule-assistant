// Using global fetch available in Node 18+ runtime

export default async function handler(req, res) {
  // Allow CORS pre-flight
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Access-Control-Allow-Methods", "GET, OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type");
  if (req.method === "OPTIONS") {
    return res.status(200).end();
  }

  try {
    const backendResp = await fetch(
      "https://ai-schedule-assistant-production.up.railway.app/api/health"
    );
    const data = await backendResp.json();
    return res.status(backendResp.status).json(data);
  } catch (err) {
    console.error("Health proxy error:", err);
    return res
      .status(500)
      .json({ error: "Proxy failed", details: err.message });
  }
}
