const form = document.querySelector("#prediction-form");
const result = document.querySelector("#result");
const error = document.querySelector("#error");
const prediction = document.querySelector("#prediction");
const reset = document.querySelector("#reset");
const submitButton = form.querySelector("button[type='submit']");
const chart = document.querySelector("#analysis-chart");
const chartStatus = document.querySelector("#chart-status");
let analysisData = null;
let currentPrediction = null;
let forecastingReport = null;

function formatRupees(value) {
  return `₹${Number(value).toLocaleString("en-IN", {maximumFractionDigits: 0})}`;
}

function getProfile() {
  const values = Object.fromEntries(new FormData(form).entries());
  return {age: values.Age, city: values.City_Category};
}

function renderAnalysis(data, prediction = currentPrediction) {
  analysisData = data;
  const profile = getProfile();
  const ageEntries = Object.entries(data.age_average);
  const maxAge = Math.max(...ageEntries.map(([, value]) => value));
  const hasPrediction = prediction !== null;
  document.querySelector("#purchase-stat-label").textContent = hasPrediction
    ? "YOUR PREDICTED PURCHASE"
    : "AVERAGE PURCHASE";
  document.querySelector("#average-purchase").textContent = formatRupees(
    hasPrediction ? prediction : data.average_purchase
  );
  document.querySelector("#rows-analyzed").textContent = Number(data.total_rows).toLocaleString("en-IN");
  document.querySelector("#age-stat-label").textContent = hasPrediction
    ? "SELECTED AGE GROUP"
    : "TOP AGE GROUP";
  document.querySelector("#top-age").textContent = hasPrediction
    ? profile.age
    : [...ageEntries].sort((a, b) => b[1] - a[1])[0][0];
  document.querySelector("#profile-context").innerHTML = prediction === null
    ? `Showing patterns for <b>${profile.age}</b> shoppers in City <b>${profile.city}</b>`
    : `Your estimated purchase <b>${formatRupees(prediction)}</b> is shown against <b>${profile.age}</b> shoppers in City <b>${profile.city}</b>`;
  const query = new URLSearchParams({age: profile.age, city: profile.city});
  if (prediction !== null) query.set("prediction", prediction);
  chart.classList.add("is-loading");
  chartStatus.hidden = false;
  chartStatus.classList.remove("error");
  chartStatus.textContent = "Updating visual analysis...";
  query.set("refresh", Date.now());
  chart.src = `/api/analysis-chart?${query.toString()}`;
}

chart.addEventListener("load", () => {
  chart.classList.remove("is-loading");
  chartStatus.hidden = true;
});

chart.addEventListener("error", () => {
  chart.classList.remove("is-loading");
  chartStatus.hidden = false;
  chartStatus.classList.add("error");
  chartStatus.textContent = "Visual analysis could not be refreshed. Try again.";
});

function renderForecasting(report) {
  forecastingReport = report;
  const best = report.best_model;
  document.querySelector("#model-table").innerHTML = `
    <div class="model-row header"><span>MODEL</span><span>RMSE</span><span>R²</span></div>
    ${report.models.map((item) => `<div class="model-row ${item.model === best.model ? "best" : ""}">
      <span>${item.model}${item.model === best.model ? " ★" : ""}</span><span>${item.rmse.toLocaleString("en-IN")}</span><span>${item.r2}</span>
    </div>`).join("")}`;
  renderDrivers();
  document.querySelector("#workflow").innerHTML = report.workflow.map((step) => `<span>${step}</span>`).join("");
  document.querySelector("#product-hypotheses").innerHTML = report.hypotheses.product_level.map((item) => `<li>${item}</li>`).join("");
  document.querySelector("#store-hypotheses").innerHTML = report.hypotheses.store_level.map((item) => `<li>${item}</li>`).join("");
}

function activeDriver(feature, profile) {
  return feature === `Age_${profile.age}`
    || feature === `City_Category_${profile.city}`
    || feature === `Gender_${profile.gender}`
    || feature === "Marital_Status"
    || feature.startsWith(`Stay_In_Current_City_Years_${profile.stay}`);
}

function renderDrivers() {
  if (!forecastingReport) return;
  const profileValues = Object.fromEntries(new FormData(form).entries());
  const profile = {
    age: profileValues.Age,
    city: profileValues.City_Category,
    gender: profileValues.Gender,
    stay: profileValues.Stay_In_Current_City_Years,
  };
  const hasPrediction = currentPrediction !== null;
  const items = [...forecastingReport.feature_importance].sort((a, b) => {
    return Number(activeDriver(b.feature, profile)) - Number(activeDriver(a.feature, profile))
      || b.importance - a.importance;
  });
  const maxImportance = items[0].importance;
  document.querySelector("#drivers-title").textContent = hasPrediction
    ? "Top purchase drivers for this profile"
    : "Top purchase drivers";
  document.querySelector("#drivers-context").textContent = hasPrediction
    ? "Highlighted factors are active for the submitted customer profile"
    : "Overall model feature importance";
  document.querySelector("#feature-list").innerHTML = items.map((item) => `
    <div class="feature-item ${hasPrediction && activeDriver(item.feature, profile) ? "active" : ""}">
      <span>${item.feature}</span><span class="feature-value">${(item.importance * 100).toFixed(1)}%</span>
      <div class="feature-track"><span style="width: ${(item.importance / maxImportance) * 100}%"></span></div>
    </div>`).join("");
}

function updateForecastingContext(prediction) {
  const panel = document.querySelector("#forecast-personalized");
  const copy = document.querySelector("#forecast-copy");
  if (prediction === null) {
    panel.hidden = true;
    copy.textContent = "Compare regression models and identify the factors that matter most for demand, inventory, and marketing capacity planning.";
    return;
  }
  const profile = getProfile();
  const bestModel = forecastingReport ? forecastingReport.best_model.model : "best evaluated model";
  document.querySelector("#forecast-profile").textContent = `${profile.age} shopper · City ${profile.city}`;
  document.querySelector("#forecast-prediction").textContent = formatRupees(prediction);
  document.querySelector("#forecast-badge").textContent = "FORECAST READY";
  copy.textContent = `Personalized demand forecast generated with the ${bestModel} benchmark for the selected customer profile.`;
  panel.hidden = false;
}

function updateHeroVisual(prediction) {
  const heroArt = document.querySelector(".hero-art");
  const badge = document.querySelector("#hero-badge");
  const forecast = document.querySelector("#hero-forecast");
  if (prediction === null) {
    heroArt.classList.remove("personalized");
    badge.innerHTML = "SALE<br><b>UP</b><br>TO 70%";
    forecast.innerHTML = "<span>BLACK FRIDAY</span><strong>Smart shopping<br>starts here.</strong>";
    return;
  }
  const profile = getProfile();
  heroArt.classList.add("personalized");
  badge.innerHTML = "YOUR<br><b>FORECAST</b><br>READY";
  forecast.innerHTML = `<span>PERSONALIZED DEMAND</span><strong>${formatRupees(prediction)}<br>${profile.age} · CITY ${profile.city}</strong>`;
}

fetch("/api/analysis")
  .then((response) => response.json())
  .then((data) => renderAnalysis(data))
  .catch(() => {
    document.querySelector(".live-label").textContent = "ANALYSIS UNAVAILABLE";
  });

fetch("/api/forecasting-report")
  .then((response) => response.json())
  .then(renderForecasting)
  .catch(() => {
    document.querySelector("#model-table").innerHTML = '<p class="loading">Forecast report unavailable.</p>';
  });

form.addEventListener("input", () => {
  currentPrediction = null;
  if (analysisData !== null) renderAnalysis(analysisData, null);
  renderDrivers();
  updateHeroVisual(null);
});

form.addEventListener("change", () => {
  currentPrediction = null;
  if (analysisData !== null) renderAnalysis(analysisData, null);
  renderDrivers();
  updateHeroVisual(null);
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  error.hidden = true;
  const data = Object.fromEntries(new FormData(form).entries());
  data.Marital_Status = Number(data.Marital_Status);
  submitButton.disabled = true;
  submitButton.querySelector("span:first-child").textContent = "Calculating...";
  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(data),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "Prediction failed.");
    prediction.textContent = formatRupees(payload.prediction);
    currentPrediction = payload.prediction;
    if (analysisData !== null) renderAnalysis(analysisData);
    renderDrivers();
    updateForecastingContext(currentPrediction);
    updateHeroVisual(currentPrediction);
    result.hidden = false;
    form.hidden = true;
    document.querySelector(".analytics").scrollIntoView({behavior: "smooth", block: "start"});
  } catch (requestError) {
    error.textContent = requestError.message;
    error.hidden = false;
  } finally {
    submitButton.disabled = false;
    submitButton.querySelector("span:first-child").textContent = "Calculate prediction";
  }
});

reset.addEventListener("click", () => {
  currentPrediction = null;
  result.hidden = true;
  form.hidden = false;
  if (analysisData !== null) renderAnalysis(analysisData, null);
  updateForecastingContext(null);
  renderDrivers();
  updateHeroVisual(null);
});
