const CACHE_NAME = "schedule-assistant-v1.0.0";
const ASSETS_TO_CACHE = [
  "/",
  "/index.html",
  "/main.dart.js",
  "/manifest.json",
  "/flutter_service_worker.js",
  "/icons/Icon-192.png",
  "/icons/Icon-512.png",
  "/favicon.png",
];

// Install event - cache essential resources
self.addEventListener("install", (event) => {
  console.log("Service Worker: Installing...");
  event.waitUntil(
    caches
      .open(CACHE_NAME)
      .then((cache) => {
        console.log("Service Worker: Caching App Shell");
        return cache.addAll(ASSETS_TO_CACHE);
      })
      .then(() => {
        return self.skipWaiting();
      })
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
      .then(() => {
        return self.clients.claim();
      })
  );
});

// Fetch event - serve from cache, fallback to network
self.addEventListener("fetch", (event) => {
  event.respondWith(
    caches
      .match(event.request)
      .then((cachedResponse) => {
        // Return cached version or fetch from network
        return (
          cachedResponse ||
          fetch(event.request).then((response) => {
            // Don't cache non-successful responses
            if (
              !response ||
              response.status !== 200 ||
              response.type !== "basic"
            ) {
              return response;
            }

            // Clone the response
            const responseToCache = response.clone();

            caches.open(CACHE_NAME).then((cache) => {
              cache.put(event.request, responseToCache);
            });

            return response;
          })
        );
      })
      .catch(() => {
        // Fallback for offline pages
        if (event.request.destination === "document") {
          return caches.match("/index.html");
        }
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
