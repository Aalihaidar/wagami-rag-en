"use strict";

/* Telegram Mini App support. The chat page is also opened from Telegram (the bot's "Open"
 * button, its menu button or a direct link); Telegram passes the launch details in the URL hash
 * (#tgWebAppData=...). Once its SDK has loaded the page tells Telegram it is ready, asks for the
 * full height, and follows Telegram's own light/dark theme -- which can differ from the device's,
 * the only theme the stylesheet otherwise knows -- by setting data-theme on <html> (chat.css has
 * a forced palette for each), then colours Telegram's header bar like the page.
 *
 * Loaded only in that case: for every other visitor this file does nothing and no request goes
 * to telegram.org.
 */

(function () {
  if (!location.hash.includes("tgWebAppData")) return;

  /** The page's own background, as the header/background colour Telegram should use. */
  function pageBackground() {
    return getComputedStyle(document.documentElement).getPropertyValue("--bg").trim();
  }

  function followTelegramTheme(webApp) {
    if (webApp.colorScheme === "dark" || webApp.colorScheme === "light") {
      document.documentElement.dataset.theme = webApp.colorScheme;
    }
    // Hex colours for these need Bot API 6.9; an older Telegram keeps its own.
    if (webApp.isVersionAtLeast && webApp.isVersionAtLeast("6.9")) {
      const color = pageBackground();
      if (color) {
        webApp.setHeaderColor(color);
        webApp.setBackgroundColor(color);
      }
    }
  }

  const script = document.createElement("script");
  script.src = "https://telegram.org/js/telegram-web-app.js";
  script.addEventListener("load", () => {
    const webApp = window.Telegram && window.Telegram.WebApp;
    if (!webApp) return;
    webApp.ready();
    webApp.expand();
    followTelegramTheme(webApp);
    webApp.onEvent("themeChanged", () => followTelegramTheme(webApp));
  });
  document.head.appendChild(script);
})();
