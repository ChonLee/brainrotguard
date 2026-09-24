const STATIC_CACHE = "brainrotguard-static-v3";
const RUNTIME_CACHE = "brainrotguard-runtime-v3";
// A navigation that never resolves looks identical to a frozen app: the cached
// shell paints and nothing else ever happens. Bound it so the origin being
// unreachable surfaces as an error page instead of an infinite splash screen.
const NAVIGATION_TIMEOUT_MS = 8000;
const STATIC_ASSETS = [
  "/manifest.webmanifest",
  "/static/style.css",
  "/static/favicon-32.png",
  "/static/favicon.png",
  "/static/brg-icon-512.png",
  "/static/logo.png",
  "/static/brg-logo.png",
  "/static/bg-doodle.png",
  "/static/thumb-preview.js",
];

self.addEventListener("install", (event) => {
  event.waitUntil(
    caches.open(STATIC_CACHE).then((cache) => cache.addAll(STATIC_ASSETS))
  );
  self.skipWaiting();
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(
        keys
          .filter((key) => key !== STATIC_CACHE && key !== RUNTIME_CACHE)
          .map((key) => caches.delete(key))
      )
    )
  );
  self.clients.claim();
});

self.addEventListener("fetch", (event) => {
  const request = event.request;
  if (request.method !== "GET") {
    return;
  }

  const url = new URL(request.url);
  if (url.origin !== self.location.origin) {
    return;
  }

  if (request.mode === "navigate") {
    event.respondWith(handleNavigation(request));
    return;
  }

  if (url.pathname.startsWith("/static/") || url.pathname === "/manifest.webmanifest") {
    event.respondWith(staleWhileRevalidate(request));
  }
});

// Navigations are never served from cache — approval state is the whole point of
// the app, and a stale page could show a video as approved when it no longer is.
// Network only, with a timeout, falling back to an explicit error page.
async function handleNavigation(request) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), NAVIGATION_TIMEOUT_MS);
  try {
    return await fetch(request, { signal: controller.signal });
  } catch (err) {
    return unreachableResponse();
  } finally {
    clearTimeout(timer);
  }
}

function unreachableResponse() {
  const origin = self.location.origin;
  const body = `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>BrainRotGuard is unreachable</title>
<style>
  body { margin: 0; min-height: 100vh; display: flex; align-items: center;
         justify-content: center; background: #0f0f23; color: #f0f0f5;
         font-family: system-ui, -apple-system, sans-serif; text-align: center; }
  main { padding: 2rem; max-width: 22rem; }
  h1 { font-size: 1.4rem; margin: 0 0 0.75rem; }
  p { line-height: 1.5; opacity: 0.85; margin: 0 0 1.5rem; }
  code { background: rgba(255,255,255,0.1); padding: 0.15em 0.4em; border-radius: 4px;
         font-size: 0.9em; word-break: break-all; }
  a { display: inline-block; background: #5b5bd6; color: #fff; text-decoration: none;
      padding: 0.7rem 1.6rem; border-radius: 999px; font-weight: 600; }
</style>
</head>
<body>
<main>
  <h1>Can't reach BrainRotGuard</h1>
  <p>No response from <code>${origin}</code>. It may be offline, or this device
     may not have a network route to it.</p>
  <a href="/">Try again</a>
</main>
</body>
</html>`;
  return new Response(body, {
    status: 503,
    headers: {
      "Content-Type": "text/html; charset=utf-8",
      "Cache-Control": "no-store",
    },
  });
}

async function staleWhileRevalidate(request) {
  const cached = await caches.match(request);
  const fetchPromise = fetch(request)
    .then(async (response) => {
      if (response && response.ok) {
        const cache = await caches.open(RUNTIME_CACHE);
        cache.put(request, response.clone());
      }
      return response;
    })
    .catch(() => cached);

  return cached || fetchPromise;
}
