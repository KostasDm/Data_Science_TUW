const color = d3.scaleOrdinal(["#1f77b4", "#2ca02c", "#d62728"]);

const compactFormat = d => {
  if (d === 0) return "0";
  if (d >= 1e9) return (d / 1e9).toFixed(1) + "B";
  if (d >= 1e6) return (d / 1e6).toFixed(1) + "M";
  if (d >= 1e3) return (d / 1e3).toFixed(1) + "K";
  return d;
};

d3.csv("./combined_nei_grouped_2010_2023_final.csv").then(raw => {
  raw.forEach(d => {
    for (let k in d) {
      d[k.trim()] = d[k];
      if (k !== k.trim()) delete d[k];
    }
    d.Year = +d.Year;
    d.State = d.State ? d.State.trim() : "ALL";
    d.City = d.City ? d.City.trim() : "ALL";
    d.CO2 = +d["CO2 emissions (non-biogenic)"] || 0;
    d.CH4 = +d["Methane (CH4) emissions"] || 0;
    d.N2O = +d["Nitrous Oxide (N2O) emissions"] || 0;
    d.Industry = d["Industry Type (sectors)"] || "Other";
  });

  setupSelectors(raw);
  setupBarplot(); 
  drawLineChart();
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
    const cities = [...new Set(data.filter(d=>d.State===state).map(d=>d.City))].sort();

    citySelect.selectAll("option").remove();
    citySelect.selectAll("option")
      .data(cities)
      .enter().append("option")
      .attr("value", d=>d)
      .text(d => d==="ALL"? "All Cities": d);
    citySelect.property("value", cities[0] || "ALL");
    updateChart();
  }

  function updateChart(){
    const selectedPollutants = pollutants.filter(p =>
      d3.select(`#pollutant-controls input[value='${p}']`).property("checked")
    );
    const state = stateSelect.property("value");
    const city = citySelect.property("value");
    updateLineChart(data, state, city, selectedPollutants);

    // Update barplot for first available year
    const firstYear = d3.min(data.filter(d=>d.State===state && d.City===city), d=>d.Year);
    updateBarplot(firstYear || null, state, city, data);
  }
}

// Line plot 

function drawLineChart(){
  const width = 900, height = 300;
  const margin = {top:40, right:100, bottom:40, left:80};
  let svg = d3.select("#chart svg");
  if(svg.empty()){
    svg = d3.select("#chart")
      .append("svg")
      .attr("width", width)
      .attr("height", height);
    svg.append("g").attr("class","x-axis").attr("transform", `translate(0,${height-margin.bottom})`);
    svg.append("g").attr("class","y-axis").attr("transform", `translate(${margin.left},0)`);
  }
}

function updateLineChart(data, state, city, selectedPollutants){
  const width = 900, height = 300;
  const margin = {top:40, right:100, bottom:40, left:80};
  const filtered = data.filter(d => d.State===state && d.City===city);
  if(filtered.length===0) return;

  const yearlyMap = d3.rollups(
    filtered,
    v => ({
      CO2: d3.sum(v,d=>d.CO2),
      CH4: d3.sum(v,d=>d.CH4),
      N2O: d3.sum(v,d=>d.N2O)
    }),
    d=>d.Year
  );

  const chartData = [];
  for(let [year, values] of yearlyMap){
    for(let pollutant of selectedPollutants){
      chartData.push({date: new Date(year,0,1), pollutant, value: values[pollutant], year, state, city});
    }
  }

  chartData.sort((a,b)=>a.date-b.date);
  const svg = d3.select("#chart svg");

  const x = d3.scaleTime().domain(d3.extent(chartData,d=>d.date)).range([margin.left,width-margin.right]);
  const y = d3.scaleLinear().domain([0,d3.max(chartData,d=>d.value)||0]).nice().range([height-margin.bottom,margin.top]);

  svg.select(".x-axis").transition().call(d3.axisBottom(x).ticks(d3.timeYear.every(1)).tickFormat(d3.timeFormat("%Y")));
  svg.select(".y-axis").transition().call(d3.axisLeft(y).tickFormat(d3.format(",")).tickFormat(compactFormat));

  svg.append("text")
  .attr("class", "y-axis-label")
  .attr("transform", `rotate(-90)`)
  .attr("x", -height/2)
  .attr("y", margin.left - 50)
  .attr("text-anchor", "middle")
  .text("Emissions (tons)");

  const grouped = d3.groups(chartData,d=>d.pollutant);
  const line = d3.line().curve(d3.curveMonotoneX).x(d=>x(d.date)).y(d=>y(d.value));
  const tooltip = d3.select(".tooltip");

  const lines = svg.selectAll(".line").data(grouped,d=>d[0]);
  lines.enter().append("path").attr("class","line")
    .merge(lines)
    .transition()
    .attr("fill","none")
    .attr("stroke", d=>color(d[0]))
    .attr("stroke-width",2)
    .attr("d", d=>line(d[1]));
  lines.exit().remove();

  const circles = svg.selectAll(".circle-group").data(grouped,d=>d[0]);
  const circlesEnter = circles.enter().append("g").attr("class","circle-group");

  circlesEnter.merge(circles).each(function([pollutant, values]){
    const circleSel = d3.select(this).selectAll("circle").data(values,d=>d.date);
    circleSel.enter().append("circle")
      .attr("r",4)
      .attr("fill", color(pollutant))
      .on("mouseenter",(e,d)=>{
        tooltip.style("opacity",1)
          .html(`<strong>${pollutant}</strong><br>Year: ${d.year}<br>State: ${d.state}<br>City: ${d.city}<br>Value: ${d3.format(",")(d.value)}`);
        updateBarplot(d.year,d.state,d.city,data);
      })
      .on("mousemove",e=>tooltip.style("left",(e.pageX+10)+"px").style("top",(e.pageY-20)+"px"))
      .on("mouseleave",()=>tooltip.style("opacity",0))
      .merge(circleSel)
      .attr("cx", d=>x(d.date))
      .attr("cy", d=>y(d.value));
    circleSel.exit().remove();
  });
  circles.exit().remove();
}


// Bar plot

function setupBarplot(){
  const svg = d3.select("#barplot");
  svg.selectAll("*").remove();
  svg.append("g").attr("class","barplot-g");

  // Add empty axes and title
  const margin = {top:30,right:50,bottom:10,left:350};
  const width = +svg.attr("width")-margin.left-margin.right;
  const height = +svg.attr("height")-margin.top-margin.bottom;
  const g = svg.select(".barplot-g").attr("transform",`translate(${margin.left},${margin.top})`);
  g.append("text")
    .attr("x", width/2)
    .attr("y", -10)
    .attr("text-anchor","middle")
    .text("Barplot – No data yet");
}

function updateBarplot(year, state, city, data) {
  const svg = d3.select("#barplot");
  const margin = { top: 30, right: 50, bottom: 40, left: 350 };
  const width = +svg.attr("width") - margin.left - margin.right;
  const height = +svg.attr("height") - margin.top - margin.bottom;

  const g = svg.select(".barplot-g")
    .attr("transform", `translate(${margin.left},${margin.top})`);

  // Clear previous content
  g.selectAll("*").remove();

  if (!year) {
    g.append("text")
      .attr("x", width / 2)
      .attr("y", height / 2)
      .attr("text-anchor", "middle")
      .text("No data for this selection");
    return;
  }

  const filtered = data.filter(d =>
    d.Year === year && d.State === state && d.City === city
  );

  if (!filtered.length) {
    g.append("text")
      .attr("x", width / 2)
      .attr("y", height / 2)
      .attr("text-anchor", "middle")
      .text("No data");
    return;
  }

  const pollutants = ["CO2", "CH4", "N2O"];
  const selectedPollutants = pollutants.filter(p =>
    d3.select(`#pollutant-controls input[value='${p}']`).property("checked")
  );

  if (!selectedPollutants.length) {
    g.append("text")
      .attr("x", width / 2)
      .attr("y", height / 2)
      .attr("text-anchor", "middle")
      .text("No pollutants selected");
    return;
  }

  const aggregated = d3.rollups(
    filtered,
    v => {
      const obj = {};
      selectedPollutants.forEach(p => {
        obj[p] = d3.sum(v, d => d[p]);
      });
      return obj;
    },
    d => d.Industry
  ).map(([industry, values]) => ({
    industry,
    ...values
  }));

  const totalEmissions = d3.sum(aggregated, d =>
    selectedPollutants.reduce((s, p) => s + d[p], 0)
  );

  const threshold = 0.003; // 0.3%
  const mainIndustries = [];
  const other = { industry: "Minor sectors (<0.3%)" };
  selectedPollutants.forEach(p => (other[p] = 0));

  aggregated.forEach(d => {
    const industryTotal = selectedPollutants.reduce((s, p) => s + d[p], 0);
    const share = industryTotal / totalEmissions;

    if (share < threshold) {
      selectedPollutants.forEach(p => {
        other[p] += d[p];
      });
    } else {
      mainIndustries.push(d);
    }
  });

  if (selectedPollutants.some(p => other[p] > 0)) {
    mainIndustries.push(other);
  }

  mainIndustries.sort((a, b) => {
    if (a.industry === "Minor sectors (<0.3%)") return 1;
    if (b.industry === "Minor sectors (<0.3%)") return -1;

    return d3.descending(
      selectedPollutants.reduce((s, p) => s + a[p], 0),
      selectedPollutants.reduce((s, p) => s + b[p], 0)
    );
  });

  const x = d3.scaleLinear()
    .domain([
      0,
      d3.max(mainIndustries, d =>
        selectedPollutants.reduce((s, p) => s + d[p], 0)
      )
    ])
    .nice()
    .range([0, width]);

  const y = d3.scaleBand()
    .domain(mainIndustries.map(d => d.industry))
    .range([0, height])
    .padding(0.2);

  const stack = d3.stack().keys(selectedPollutants);
  const series = stack(mainIndustries);

  g.append("g").call(d3.axisLeft(y));

  g.append("g")
    .attr("class", "x-axis")
    .attr("transform", `translate(0,${height})`)
    .call(d3.axisBottom(x).tickFormat(compactFormat));

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
 
  const tooltip = d3.select(".tooltip"); 

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
    .attr("width", d => x(d[1]) - x(d[0]))
    .on("mouseenter", (event, d) => {
  
  const pollutant = d3.select(event.currentTarget.parentNode).datum().key; // pollutant name
  const value = d.data[pollutant];

  tooltip
      .style("opacity", 1)
      .html(`
        <strong>${pollutant}</strong><br>
        Industry: ${d.data.industry}<br>
        Year: ${year}<br>
        State: ${state}<br>
        City: ${city}<br>
        Value: ${d3.format(",")(value)}
      `);
  })
   .on("mousemove", event => {
     tooltip
      .style("left", (event.pageX + 10) + "px")
      .style("top", (event.pageY - 20) + "px");
  })
   .on("mouseleave", () => {
     tooltip.style("opacity", 0);
  });

  g.append("text")
    .attr("x", width / 2)
    .attr("y", -10)
    .attr("text-anchor", "middle")
    .attr("font-weight", "bold")
    .text(`Emissions by Industry – ${city}, ${state} (${year})`);

  g.append("text")
  .attr("x", width / 2)
  .attr("y", height + margin.bottom - 5) 
  .attr("text-anchor", "middle")
  .text("Emissions (tons)");

}
