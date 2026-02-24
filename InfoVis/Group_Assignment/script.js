d3.csv("combined_nei_grouped_2010_2023.csv").then(data => {

   data.forEach(d => {
    d.Year = +d.Year;
    d["CO2 emissions (non-biogenic) "] = +d["CO2 emissions (non-biogenic) "];
    d["Methane (CH4) emissions "] = +d["Methane (CH4) emissions "];
    d["Nitrous Oxide (N2O) emissions "] = +d["Nitrous Oxide (N2O) emissions "];
  });

  // ----------------------------
  // Gas configuration
  // ----------------------------
  const gases = [
    { key: "CO2 emissions (non-biogenic) ", label: "CO2", color: "#1f77b4" },
    { key: "Methane (CH4) emissions ", label: "CH4", color: "#2ca02c" },
    { key: "Nitrous Oxide (N2O) emissions ", label: "N2O", color: "#d62728" }
  ];

  // ----------------------------
  // Dropdowns
  // ----------------------------
  const yearSelect  = d3.select("#yearDropdown");
  const stateSelect = d3.select("#stateDropdown");
  const citySelect  = d3.select("#cityDropdown");

  const years = Array.from(new Set(data.map(d => d.Year))).sort(d3.ascending);
  yearSelect.selectAll("option")
    .data(years)
    .join("option")
    .attr("value", d => d)
    .text(d => d);

  const states = Array.from(new Set(data.map(d => d.State))).sort();
  stateSelect.selectAll("option")
    .data(states)
    .join("option")
    .attr("value", d => d)
    .text(d => d);

  function updateCities(state) {
    const cities = Array.from(
      new Set(data.filter(d => d.State === state).map(d => d.City))
    ).sort();

    citySelect.selectAll("option").remove();
    citySelect.selectAll("option")
      .data(cities)
      .join("option")
      .attr("value", d => d)
      .text(d => d);
  }

  updateCities(states[0]);

  // ----------------------------
  // SVG setup
  // ----------------------------
  const svg = d3.select("#barplot");
  const margin = { top: 40, right: 40, bottom: 50, left: 260 };
  const width  = +svg.attr("width") - margin.left - margin.right;
  const height = +svg.attr("height") - margin.top - margin.bottom;
  const g = svg.append("g").attr("transform", `translate(${margin.left},${margin.top})`);

  // ----------------------------
  // Main update function
  // ----------------------------
  function updateBarplot() {
    const year  = +yearSelect.node().value;
    const state = stateSelect.node().value;
    const city  = citySelect.node().value;

    const filtered = data.filter(d => d.Year === year && d.State === state && d.City === city);

    if (filtered.length === 0) {
      g.selectAll("*").remove();
      g.append("text")
        .attr("x", width / 2)
        .attr("y", height / 2)
        .attr("text-anchor", "middle")
        .text("No data available");
      return;
    }

    // ----------------------------
    // Aggregate by Industry Type
    // ----------------------------
    const aggregated = d3.rollups(
      filtered,
      v => ({
        co2: d3.sum(v, d => d["CO2 emissions (non-biogenic) "]),
        ch4: d3.sum(v, d => d["Methane (CH4) emissions "]),
        n2o: d3.sum(v, d => d["Nitrous Oxide (N2O) emissions "])
      }),
      d => d["Industry Type (sectors)"]
    ).map(([industry, v]) => ({
      industry,
      co2: v.co2,
      ch4: v.ch4,
      n2o: v.n2o
    }));

    // ----------------------------
    // Stack generator
    // ----------------------------
    const stack = d3.stack().keys(["co2", "ch4", "n2o"]);
    const series = stack(aggregated);

    // ----------------------------
    // Scales
    // ----------------------------
    const y = d3.scaleBand()
      .domain(aggregated.map(d => d.industry))
      .range([0, height])
      .padding(0.2);

    const x = d3.scaleLinear()
      .domain([0, d3.max(aggregated, d => d.co2 + d.ch4 + d.n2o)])
      .nice()
      .range([0, width]);

    const color = d3.scaleOrdinal()
      .domain(["co2", "ch4", "n2o"])
      .range(["#1f77b4", "#2ca02c", "#d62728"]);

    // ----------------------------
    // Clear plot
    // ----------------------------
    g.selectAll("*").remove();

    // ----------------------------
    // Axes
    // ----------------------------
    g.append("g").call(d3.axisLeft(y));
    g.append("g").attr("transform", `translate(0,${height})`).call(d3.axisBottom(x));

    // ----------------------------
    // Stacked bars
    // ----------------------------
    g.selectAll(".layer")
      .data(series)
      .join("g")
      .attr("fill", d => color(d.key))
      .selectAll("rect")
      .data(d => d)
      .join("rect")
      .attr("y", d => y(d.data.industry))
      .attr("x", d => x(d[0]))
      .attr("height", y.bandwidth())
      .attr("width", d => x(d[1]) - x(d[0]));

    // ----------------------------
    // Title
    // ----------------------------
    g.append("text")
      .attr("x", width / 2)
      .attr("y", -15)
      .attr("text-anchor", "middle")
      .attr("font-size", "16px")
      .text(`Stacked Emissions by Industry – ${city}, ${state} (${year})`);

    // ----------------------------
    // X-axis label
    // ----------------------------
    g.append("text")
      .attr("x", width / 2)
      .attr("y", height + 40)
      .attr("text-anchor", "middle")
      .text("Total emissions (metric tons)");

    // ----------------------------
    // Legend
    // ----------------------------
    const legend = g.selectAll(".legend")
      .data(gases)
      .join("g")
      .attr("class", "legend")
      .attr("transform", (d,i) => `translate(0, ${-margin.top + i*20})`);

    legend.append("rect")
      .attr("x", width - 18)
      .attr("width", 18)
      .attr("height", 18)
      .attr("fill", d => d.color);

    legend.append("text")
      .attr("x", width - 24)
      .attr("y", 9)
      .attr("dy", "0.35em")
      .attr("text-anchor", "end")
      .text(d => d.label);
  }

  // ----------------------------
  // Event listeners
  // ----------------------------
  yearSelect.on("change", updateBarplot);
  stateSelect.on("change", () => {
    updateCities(stateSelect.node().value);
    updateBarplot();
  });
  citySelect.on("change", updateBarplot);

  // ----------------------------
  // Initial render
  // ----------------------------
  updateBarplot();
});