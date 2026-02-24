const color = d3.scaleOrdinal([
    "#1f77b4", "#2ca02c", "#d62728"
  ]);
  
  d3.csv("./combined_nei_grouped_2010_2023_final.csv").then(raw => {
    raw.forEach(d => {
      for (let k in d) {
        d[k.trim()] = d[k];
        if (k !== k.trim()) delete d[k];
      }
  
      d.Year = +d.Year;
      // Keep internal values as "ALL" to match your CSV exactly
      d.State = d.State ? d.State.trim() : "";
      d.City = d.City ? d.City.trim() : "";
  
      d.CO2 = +d["CO2 emissions (non-biogenic)"] || 0;
      d.CH4 = +d["Methane (CH4) emissions"] || 0;
      d.N2O = +d["Nitrous Oxide (N2O) emissions"] || 0;
    });
  
    setupSelectors(raw);
  });
  
  function setupSelectors(data) {
    const pollutants = ["CO2", "CH4", "N2O"];
    const stateSelect = d3.select("#state-select");
    const citySelect = d3.select("#city-select");
  
    // Get unique states and sort them
    const states = [...new Set(data.map(d => d.State))].filter(s => s).sort((a, b) => {
      // Keep "ALL" at the very top
      if (a === "ALL") return -1;
      if (b === "ALL") return 1;
      return a.localeCompare(b);
    });
  
    stateSelect.selectAll("option").remove();
    stateSelect.selectAll("option")
      .data(states)
      .enter()
      .append("option")
      .attr("value", d => d) // Internal value remains "ALL"
      .text(d => d === "ALL" ? "All States" : d); // Displayed as "All States"
  
    // Set default to "ALL" so it matches your CSV rows
    stateSelect.property("value", "ALL");
  
    stateSelect.on("change", updateCities);
    citySelect.on("change", updateChart);
    d3.selectAll("#pollutant-controls input").on("change", updateChart);
  
    // Run initial setup
    updateCities();
  
    function updateCities() {
      const state = stateSelect.property("value");
      const cities = [...new Set(data.filter(d => d.State === state).map(d => d.City))].filter(c => c).sort();
  
      citySelect.selectAll("option").remove();
      citySelect.selectAll("option")
        .data(cities)
        .enter()
        .append("option")
        .attr("value", d => d)
        .text(d => d === "ALL" ? "All Cities" : d); // Handle city dropdown display
  
      citySelect.property("value", cities[0]);
      updateChart();
    }
  
    function updateChart() {
      const selectedPollutants = pollutants.filter(p =>
        d3.select(`#pollutant-controls input[value='${p}']`).property("checked")
      );
  
      updateFilteredChart(
        data,
        selectedPollutants,
        stateSelect.property("value"),
        citySelect.property("value")
      );
    }
  }
  
  function updateFilteredChart(data, selectedPollutants, state, city) {
    // Filters for rows where State is "ALL" and City is "ALL"
    const filtered = data.filter(d => d.State === state && d.City === city);
  
    const yearlyMap = d3.rollup(
      filtered,
      v => ({
        CO2: d3.sum(v, d => d.CO2),
        CH4: d3.sum(v, d => d.CH4),
        N2O: d3.sum(v, d => d.N2O)
      }),
      d => d.Year
    );
  
    const chartData = [];
    for (let [year, values] of yearlyMap) {
      for (let pollutant of selectedPollutants) {
        chartData.push({
          date: new Date(year, 0, 1),
          pollutant,
          value: values[pollutant]
        });
      }
    }
  
    chartData.sort((a, b) => a.date - b.date);
    drawLineChart(chartData);
  }
  
  function drawLineChart(data) {
    const width = 900, height = 450;
    // Margin left 85 to ensure 1,000,000+ labels don't get cut off
    const margin = { top: 40, right: 100, bottom: 40, left: 85 };
  
    let svg = d3.select("#chart svg");
    if (svg.empty()) {
      svg = d3.select("#chart")
        .append("svg")
        .attr("width", width)
        .attr("height", height);
  
      svg.append("g").attr("class", "x-axis").attr("transform", `translate(0,${height - margin.bottom})`);
      svg.append("g").attr("class", "y-axis").attr("transform", `translate(${margin.left},0)`);
    }
  
    const x = d3.scaleTime()
      .domain(d3.extent(data, d => d.date))
      .range([margin.left, width - margin.right]);
  
    const y = d3.scaleLinear()
      .domain([0, d3.max(data, d => d.value) || 0])
      .nice()
      .range([height - margin.bottom, margin.top]);
  
    svg.select(".x-axis").transition().call(d3.axisBottom(x).ticks(d3.timeYear.every(1)).tickFormat(d3.timeFormat("%Y")));
    
    // custom formatter for y axis when all is picked (billion and million for large numbers)
    const formatYAxis = (d) => {
    if (d === 0) return 0;
    if (d >= 1e9) {
      return d3.format(".1f")(d / 1e9) + " Billion";
    }
    if (d >= 1e6) {
      return d3.format(".1f")(d / 1e6) + " Million";
    }
    return d3.format(",")(d);
  };
  
  svg.select(".y-axis")
    .transition()
    .duration(500)
    .call(d3.axisLeft(y).tickFormat(formatYAxis));
  
    const grouped = d3.groups(data, d => d.pollutant);
    const line = d3.line().curve(d3.curveMonotoneX).x(d => x(d.date)).y(d => y(d.value));
  
    let tooltip = d3.select(".tooltip");
    if (tooltip.empty()) tooltip = d3.select("body").append("div").attr("class", "tooltip").style("opacity", 0);
  
    const lines = svg.selectAll(".line").data(grouped, d => d[0]);
    lines.enter().append("path").attr("class", "line")
      .merge(lines)
      .transition()
      .attr("fill", "none")
      .attr("stroke", d => color(d[0]))
      .attr("stroke-width", 2)
      .attr("d", d => line(d[1]));
    lines.exit().remove();
  
    const circles = svg.selectAll(".circle-group").data(grouped, d => d[0]);
    const circlesEnter = circles.enter().append("g").attr("class", "circle-group");
  
    circlesEnter.merge(circles).each(function ([pollutant, values]) {
      const circleSel = d3.select(this).selectAll("circle").data(values, d => d.date);
  
      circleSel.enter().append("circle")
        .attr("r", 4)
        .attr("fill", color(pollutant))
        .on("mouseenter", (e, d) => tooltip.style("opacity", 1).html(
          `<strong>${pollutant}</strong><br>Year: ${d.date.getFullYear()}<br>Value: ${d3.format(",")(d.value.toFixed(0))}`
        ))
        .on("mousemove", e => tooltip.style("left", (e.pageX + 10) + "px").style("top", (e.pageY - 20) + "px"))
        .on("mouseleave", () => tooltip.style("opacity", 0))
        .merge(circleSel)
        .transition()
        .attr("cx", d => x(d.date))
        .attr("cy", d => y(d.value));
  
      circleSel.exit().remove();
    });
  
    circles.exit().remove();
  }