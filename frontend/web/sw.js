// Service Worker for Schedule Assistant PWA
const CACHE_NAME = "schedule-assistant-v1";
const ASSETS_TO_CACHE = [
  "/",
  "/index.html",
  "/main.dart.js",
  "/flutter_bootstrap.js",
  "/manifest.json",
  "/favicon.png",
  "/icons/Icon-192.png",
  "/icons/Icon-512.png",
  "/icons/Icon-maskable-192.png",
  "/icons/Icon-maskable-512.png",
  "https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap",
];

// Install event - cache assets
self.addEventListener("install", (event) => {
  console.log("Service Worker: Installing...");
  event.waitUntil(
    caches
      .open(CACHE_NAME)
      .then((cache) => {
        console.log("Service Worker: Caching App Shell");
        return cache.addAll(ASSETS_TO_CACHE);
      })
      .then(() => self.skipWaiting())
  );
});

// Activate event - clean up old caches
self.addEventListener("activate", (event) => {
  console.log("Service Worker: Activating...");
  event.waitUntil(
    caches
      .keys()
      .then((cacheNames) => {
        return Promise.all(
          cacheNames.map((cacheName) => {
            if (cacheName !== CACHE_NAME) {
              console.log("Service Worker: Deleting old cache", cacheName);
              return caches.delete(cacheName);
            }
          })
        );
      })
      .then(() => self.clients.claim())
  );
});

// Fetch event - serve from cache or network
self.addEventListener("fetch", (event) => {
  // Skip cross-origin requests
  if (
    !event.request.url.startsWith(self.location.origin) &&
    !event.request.url.startsWith("https://fonts.googleapis.com") &&
    !event.request.url.startsWith("https://fonts.gstatic.com")
  ) {
    return;
  }

  // For API calls, always go to network first
  if (event.request.url.includes("/api/")) {
    event.respondWith(
      fetch(event.request).catch(() => {
        return caches.match("/index.html");
      })
    );
    return;
  }

  // For everything else, try cache first, then network
  event.respondWith(
    caches
      .match(event.request)
      .then((response) => {
        return (
          response ||
          fetch(event.request).then((fetchResponse) => {
            // Don't cache API responses
            if (!event.request.url.includes("/api/")) {
              return caches.open(CACHE_NAME).then((cache) => {
                cache.put(event.request, fetchResponse.clone());
                return fetchResponse;
              });
            }
            return fetchResponse;
          })
        );
      })
      .catch(() => {
        // If both cache and network fail, return the offline page
        if (event.request.mode === "navigate") {
          return caches.match("/index.html");
        }
        return new Response("Network error happened", {
          status: 408,
          headers: { "Content-Type": "text/plain" },
        });
      })
  );
});

// Background sync for when connection is restored
self.addEventListener("sync", (event) => {
  if (event.tag === "chat-sync") {
    event.waitUntil(syncChatMessages());
  }
});

async function syncChatMessages() {
  // Implement your chat sync logic here
  console.log("Syncing chat messages...");
}

// Push notifications (optional)
self.addEventListener("push", (event) => {
  const options = {
    body: event.data
      ? event.data.text()
      : "New message from Schedule Assistant",
    icon: "/icons/Icon-192.png",
    badge: "/icons/Icon-96.png",
    vibrate: [100, 50, 100],
    data: {
      dateOfArrival: Date.now(),
      primaryKey: 1,
    },
    actions: [
      {
        action: "explore",
        title: "Open App",
        icon: "/icons/Icon-192.png",
      },
      {
        action: "close",
        title: "Close",
        icon: "/icons/Icon-192.png",
      },
    ],
  };

  event.waitUntil(
    self.registration.showNotification("Schedule Assistant", options)
  );
});
