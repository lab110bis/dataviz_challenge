const express = require("express");
const app = express();

const PORT = 5000;

app.use(function(req, res, next) {
  res.header("Access-Control-Allow-Origin", "*");
  res.header(
    "Access-Control-Allow-Headers",
    "Origin, X-Requested-With, Content-Type, Accept"
  );
  next();
});

app.get("/", function(req, res) {
  res.send("Root");
});

app.use("/data", express.static("data"));

app.listen(PORT, function() {
  console.log("Server running on http://localhost:" + PORT);
});
