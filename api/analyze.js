// Using global fetch available in Node 18+ runtime

export default async function handler(req, res) {
  // Allow CORS pre-flight
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type");
  if (req.method === "OPTIONS") {
    return res.status(200).end();
  }

  if (req.method !== "POST") {
    return res.status(405).json({ error: "Method not allowed" });
  }

  try {
    const backendResp = await fetch(
      "https://ai-schedule-assistant-production.up.railway.app/api/analyze",
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(req.body),
      }
    );

    const data = await backendResp.json();
    return res.status(backendResp.status).json(data);
  } catch (err) {
    console.error("Analyze proxy error:", err);
    return res
      .status(500)
      .json({ error: "Proxy failed", details: err.message });
  }
}
