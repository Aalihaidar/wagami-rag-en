"use strict";

/* Minimal Server-Sent Events reader for a fetch() Response -- EventSource can't be used here
 * because /chat/stream is a POST with a JSON body. Loaded before chat.js; no build step.
 *
 * Each event's `data` is expected to be one JSON document (the server always sends it on a
 * single line), and is returned already parsed.
 */

/** Parses one SSE block (the text between blank lines) into {event, data}, or null when it
 * carries no data (e.g. a `:` comment line used as a keep-alive). */
function parseSseBlock(block) {
  let event = "message";
  const dataLines = [];
  for (const line of block.split("\n")) {
    if (line.startsWith(":")) continue;
    const separator = line.indexOf(":");
    const field = separator === -1 ? line : line.slice(0, separator);
    let value = separator === -1 ? "" : line.slice(separator + 1);
    if (value.startsWith(" ")) value = value.slice(1);
    if (field === "event") event = value;
    else if (field === "data") dataLines.push(value);
  }
  if (dataLines.length === 0) return null;
  return { event, data: JSON.parse(dataLines.join("\n")) };
}

/** Async generator over the events of a streaming Response. Chunk boundaries can fall anywhere
 * -- mid-event, or in the middle of a multi-byte character -- so bytes are decoded in streaming
 * mode and events are only emitted once their terminating blank line has arrived. Stops reading
 * (and releases the connection) when the consumer stops iterating. */
async function* readSseEvents(response) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      // Normalised on the whole buffer, not per chunk: a "\r\n" can be split across two chunks.
      buffer = (buffer + decoder.decode(value, { stream: true })).replace(/\r\n/g, "\n");
      let boundary = buffer.indexOf("\n\n");
      while (boundary !== -1) {
        const parsed = parseSseBlock(buffer.slice(0, boundary));
        buffer = buffer.slice(boundary + 2);
        if (parsed) yield parsed;
        boundary = buffer.indexOf("\n\n");
      }
    }
    buffer += decoder.decode();
    if (buffer.trim()) {
      const parsed = parseSseBlock(buffer);
      if (parsed) yield parsed;
    }
  } finally {
    reader.cancel().catch(() => {});
  }
}
