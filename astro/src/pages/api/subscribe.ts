import type { APIRoute } from "astro";

export const prerender = false;

export const POST: APIRoute = async ({ request }) => {
  const { email } = await request.json();

  if (!email || !/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)) {
    return new Response(JSON.stringify({ error: "Invalid email address." }), {
      status: 400,
      headers: { "Content-Type": "application/json" },
    });
  }

  const apiKey = import.meta.env.BREVO_API_KEY;
  const listId = Number(import.meta.env.BREVO_LIST_ID);

  if (!apiKey || !listId) {
    console.error("Missing BREVO_API_KEY or BREVO_LIST_ID env vars");
    return new Response(JSON.stringify({ error: "Server misconfiguration." }), {
      status: 500,
      headers: { "Content-Type": "application/json" },
    });
  }

  const res = await fetch("https://api.brevo.com/v3/contacts", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "api-key": apiKey,
    },
    body: JSON.stringify({
      email,
      listIds: [listId],
      updateEnabled: true, // re-subscribes if contact already exists
    }),
  });

  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    // Brevo returns 204 on success, but also check for "already exists" (400 + code 28)
    if (res.status === 400 && body?.code === "duplicate_parameter") {
      // Contact already subscribed — treat as success
      return new Response(JSON.stringify({ ok: true }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      });
    }
    console.error("Brevo error", res.status, body);
    return new Response(JSON.stringify({ error: "Subscription failed. Please try again." }), {
      status: 500,
      headers: { "Content-Type": "application/json" },
    });
  }

  return new Response(JSON.stringify({ ok: true }), {
    status: 200,
    headers: { "Content-Type": "application/json" },
  });
};
