// Main JavaScript for Soccer Prediction App

document.addEventListener("DOMContentLoaded", function () {
  // Get the current date in YYYY-MM-DD format
  const today = new Date().toISOString().split("T")[0];

  // Highlight today's matches if we're on the matches page
  if (window.location.pathname.includes("/matches")) {
    const dateParam = new URLSearchParams(window.location.search).get("date");

    // If today is selected or no date is specified, highlight the "Today" button
    if (!dateParam || dateParam === today) {
      document
        .querySelector("a.btn-outline-primary:nth-child(2)")
        .classList.remove("btn-outline-primary");
      document.querySelector("a.btn:nth-child(2)").classList.add("btn-primary");
    }
  }

  // Add tooltips to any elements with data-bs-toggle="tooltip" attribute
  const tooltipTriggerList = [].slice.call(
    document.querySelectorAll('[data-bs-toggle="tooltip"]')
  );
  tooltipTriggerList.map(function (tooltipTriggerEl) {
    return new bootstrap.Tooltip(tooltipTriggerEl);
  });
});
