// Simple login validation
document.getElementById('loginForm').addEventListener('submit', function (e) {
    e.preventDefault();
    const username = document.getElementById("username").value;
    const password = document.getElementById("password").value;
    if (username === "admin" && password === "123456") {
      window.location.href = "dashboard.html"; // Redirect to dashboard
    } else {
      document.getElementById("error").innerText = "Invalid credentials.";
    }
  });
  