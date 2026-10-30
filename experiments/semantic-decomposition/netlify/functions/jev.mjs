// A credential-protected bridge for the local Python DocETL process.
export default async (request) => {
  const token = process.env.JEV_PROXY_TOKEN;
  if (!token) return new Response("JEV_PROXY_TOKEN is not configured", {status: 503});
  if (request.headers.get("authorization") !== `Bearer ${token}`) return new Response("Unauthorized", {status: 401});
  if (request.method !== "POST") return new Response("Method not allowed", {status: 405});
  const base = process.env.TYPESAFE_BASE_URL;
  const key = process.env.TYPESAFE_API_KEY;
  if (!base || !key) return new Response("Netlify AI Gateway is not configured", {status: 503});
  const root = base.replace(/\/$/, "");
  const endpoint = root + (root.endsWith("/v1") ? "/systemone" : "/v1/systemone");
  const upstream = await fetch(endpoint, {
    method: "POST",
    headers: {"content-type": "application/json", accept: "application/json", authorization: `Bearer ${key}`},
    body: await request.text(),
  });
  return new Response(await upstream.text(), {status: upstream.status, headers: {"content-type": "application/json"}});
};
