# scripts

Browser-based visual companion for the `brainstorming` skill — a local server that renders mockups, diagrams, and option comparisons in the browser and streams the user's choices back over a WebSocket.

## Code in this directory

- `server.cjs` — zero-dependency Node HTTP + WebSocket server (RFC 6455 handshake/framing implemented by hand); serves the frame and relays events
- `helper.js` — browser-side client; connects back to the server over WebSocket and queues events
- `frame-template.html` — HTML shell served to the browser session
- `start-server.sh` — launch the server on a random high port, one session directory per run; prints JSON with the connection URL
- `stop-server.sh` — kill the server process and clean up the session directory (keeps persistent `.superpowers/` dirs)
