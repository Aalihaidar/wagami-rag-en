"use strict";

/* Telegram Mini App support. The chat page is also opened from Telegram (the bot's "Open"
 * button, its menu button or a direct link); Telegram passes the launch details in the URL hash
 * (#tgWebAppData=...), and the page should tell it when it is ready and ask for the full
 * height. Loaded only in that case: for every other visitor this file does nothing and no
 * request goes to telegram.org.
 */

(function () {
  if (!location.hash.includes("tgWebAppData")) return;

  const script = document.createElement("script");
  script.src = "https://telegram.org/js/telegram-web-app.js";
  script.addEventListener("load", () => {
    const webApp = window.Telegram && window.Telegram.WebApp;
    if (!webApp) return;
    webApp.ready();
    webApp.expand();
  });
  document.head.appendChild(script);
})();
