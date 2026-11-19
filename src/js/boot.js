const statusElement = document.querySelector("#modelStatus");
const errorElement = document.querySelector("#errorMessage");

statusElement.textContent = "Starting person detector...";

import("./app.js?v=20261119-console2").catch((error) => {
  console.error("Detector startup failed", error);
  statusElement.textContent = `Detector failed: ${error?.message || "Unknown error"}`;
  errorElement.textContent = `Startup error: ${error?.message || "Unknown error"}. Reload the page to try again.`;
  errorElement.hidden = false;
});
