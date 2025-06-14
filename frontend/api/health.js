export default async function handler(req, res) {
  try {
    const apiResponse = await fetch(
      "https://ai-schedule-assistant-production.up.railway.app/api/health"
    );
    const data = await apiResponse.json();

    res.status(200).json(data);
  } catch (error) {
    res.status(500).json({ error: "Failed to fetch from API" });
  }
}
