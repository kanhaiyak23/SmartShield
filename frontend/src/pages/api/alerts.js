// Next.js API route - proxy to backend
export default async function handler(req, res) {
  const backendUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:3001';
  const { limit = '100', severity } = req.query;
  
  try {
    let url = `${backendUrl}/api/alerts?limit=${limit}`;
    if (severity) {
      url += `&severity=${severity}`;
    }
    
    const response = await fetch(url);
    const data = await response.json();
    res.status(200).json(data);
  } catch (error) {
    res.status(500).json({ error: 'Failed to fetch alerts' });
  }
}


