(async function initGlucoseTimeline() {
  const root = document.querySelector(".glucose-app");
  if (!root) return;

  const [readingsRaw, eventsRaw, foodResponseRows] = await Promise.all([
    d3.json(root.dataset.glucoseUrl),
    d3.json(root.dataset.eventsUrl),
    d3.json(root.dataset.responseUrl),
  ]);

  const readings = readingsRaw.map((item) => ({
    ...item,
    date: new Date(item.timestamp),
  }));
  const events = eventsRaw
    .map((item) => ({
      ...item,
      date: new Date(item.timestamp),
    }))
    .filter((item) => item.category === "food" || item.category === "drink");

  const svg = d3.select("#glucose-chart");
  const yAxisSvg = d3.select("#glucose-y-axis");
  const scroll = document.querySelector("#timeline-scroll");
  const rangeButtons = Array.from(document.querySelectorAll(".range-button"));
  const timelineSlider = document.querySelector("#timeline-slider");
  const timelineWindow = document.querySelector("#timeline-window");

  const stats = {
    average: document.querySelector("#stat-average"),
    range: document.querySelector("#stat-range"),
    low: document.querySelector("#stat-low"),
    high: document.querySelector("#stat-high"),
    events: document.querySelector("#stat-events"),
  };
  const detail = {
    time: document.querySelector("#detail-time"),
    primary: document.querySelector("#detail-primary"),
    secondary: document.querySelector("#detail-secondary"),
  };
  const responseTableBody = document.querySelector("#food-response-table-body");
  const responseCount = document.querySelector("#food-response-count");

  const categories = ["food", "drink"];
  const margin = { top: 28, right: 34, bottom: 76, left: 58 };
  let height = 450;
  let chartHeight = 344;
  const minDate = d3.min(readings, (d) => d.date);
  const maxDate = d3.max(readings, (d) => d.date);
  const allHours = Math.max(1, (maxDate - minDate) / 36e5);
  const bisectReading = d3.bisector((d) => d.date).center;

  let selectedHours = window.matchMedia("(max-width: 760px)").matches ? 72 : 168;
  let responseEvent = null;
  let xScale = null;
  let yScale = null;
  let hoverLine = null;
  let hoverDot = null;
  let hoverLayer = null;
  let visibleReadings = readings;
  let sliderIsDragging = false;
  let pendingScrollRatio = 0;
  let lastIsMobile = window.matchMedia("(max-width: 760px)").matches;
  let userSelectedRange = false;

  const viewportWidth = () => Math.max(760, scroll.clientWidth || 760);
  const isMobile = () => window.matchMedia("(max-width: 760px)").matches;

  function formatDateTime(date) {
    return d3.timeFormat("%a, %b %-d at %-I:%M %p")(date);
  }

  function formatCompactDateTime(timestamp) {
    return d3.timeFormat("%b %-d, %-I:%M %p")(new Date(timestamp));
  }

  function formatDateShort(date) {
    return d3.timeFormat("%b %-d, %-I %p")(date);
  }

  function formatRange(start, end) {
    if (d3.timeDay(start).getTime() === d3.timeDay(end).getTime()) {
      return `${d3.timeFormat("%b %-d")(start)}, ${d3.timeFormat("%-I:%M %p")(start)}-${d3.timeFormat(
        "%-I:%M %p"
      )(end)}`;
    }

    return `${formatDateShort(start)} - ${formatDateShort(end)}`;
  }

  function visibleDomain() {
    if (!xScale) return [minDate, maxDate];

    const left = scroll.scrollLeft + margin.left;
    const right = scroll.scrollLeft + scroll.clientWidth - margin.right;
    const start = d3.max([minDate, xScale.invert(left)]);
    const end = d3.min([maxDate, xScale.invert(right)]);

    return start <= end ? [start, end] : [minDate, maxDate];
  }

  function updateStats(windowReadings, windowEvents) {
    if (!windowReadings.length) {
      Object.values(stats).forEach((node) => {
        node.textContent = "--";
      });
      return;
    }

    const values = windowReadings.map((d) => d.glucose);
    const average = d3.mean(values);
    const min = d3.min(values);
    const max = d3.max(values);
    const lowPct = values.filter((value) => value < 70).length / values.length;
    const highPct = values.filter((value) => value > 140).length / values.length;

    stats.average.textContent = `${Math.round(average)} mg/dL`;
    stats.range.textContent = `${min}-${max}`;
    stats.low.textContent = d3.format(".0%")(lowPct);
    stats.high.textContent = d3.format(".0%")(highPct);
    stats.events.textContent = String(windowEvents.length);
  }

  function updateViewportSummary() {
    const [start, end] = visibleDomain();
    visibleReadings = readings.filter((d) => d.date >= start && d.date <= end);
    const visibleEvents = events.filter((d) => d.date >= start && d.date <= end);

    updateStats(visibleReadings, visibleEvents);
    timelineWindow.textContent = formatRange(start, end);

    const maxScroll = Math.max(1, scroll.scrollWidth - scroll.clientWidth);
    if (!sliderIsDragging) {
      timelineSlider.value = Math.round((scroll.scrollLeft / maxScroll) * Number(timelineSlider.max));
    }

    updateCrosshair();
  }

  function describeEvent(event) {
    const eventIndex = bisectReading(readings, event.date);
    const atEvent = readings[eventIndex];
    const responseEnd = new Date(event.date.getTime() + 3 * 36e5);
    const responseReadings = readings.filter((d) => d.date >= event.date && d.date <= responseEnd);
    const peak = responseReadings.length
      ? responseReadings.reduce((best, item) => (item.glucose > best.glucose ? item : best))
      : null;

    detail.time.textContent = formatDateTime(event.date);
    detail.primary.textContent = event.entry;
    const nearestGlucose = atEvent && Number.isFinite(atEvent.glucose) ? atEvent.glucose : "--";
    detail.secondary.textContent = peak
      ? `${event.category} | nearest reading ${nearestGlucose} mg/dL | next 3h peak ${peak.glucose} mg/dL at ${formatDateShort(
          peak.date
        )}`
      : `${event.category} | no glucose readings found after this event`;
  }

  function describeReading(reading) {
    const nearbyEvents = events
      .filter((event) => Math.abs(event.date - reading.date) <= 30 * 60 * 1000)
      .slice(0, 3);

    detail.time.textContent = formatDateTime(reading.date);
    detail.primary.textContent = `${reading.glucose} mg/dL`;
    detail.secondary.textContent = nearbyEvents.length
      ? nearbyEvents.map((event) => `${event.entry} (${event.category})`).join(" | ")
      : "No journal events within 30 minutes.";
  }

  function updateCrosshair(targetDate) {
    if (!xScale || !yScale || !hoverLayer || !visibleReadings.length) return;

    const [start, end] = visibleDomain();
    const midpoint = new Date((start.getTime() + end.getTime()) / 2);
    const pointerDate = targetDate || midpoint;
    const clampedDate = new Date(Math.min(Math.max(pointerDate.getTime(), start.getTime()), end.getTime()));
    const nearest = visibleReadings[bisectReading(visibleReadings, clampedDate)];

    if (!nearest) return;

    hoverLayer.style("display", null);
    hoverLine.attr("x1", xScale(nearest.date)).attr("x2", xScale(nearest.date));
    hoverDot.attr("cx", xScale(nearest.date)).attr("cy", yScale(nearest.glucose));
    describeReading(nearest);
  }

  function updateCrosshairFromClientX(clientX) {
    if (!clientX || !xScale) return;

    const bounds = scroll.getBoundingClientRect();
    const svgX = clientX - bounds.left + scroll.scrollLeft;
    updateCrosshair(xScale.invert(svgX));
  }

  function formatTableValue(value) {
    return value == null ? "--" : String(value);
  }

  function escapeHtml(value) {
    return formatTableValue(value)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function renderFoodResponseTable(rows) {
    responseCount.textContent = `${rows.length} rows`;
    responseTableBody.innerHTML = rows
      .map(
        (row) => `
          <tr>
            <td data-label="Date / Time">${escapeHtml(formatCompactDateTime(row.timestamp))}</td>
            <td data-label="Food Entry">${escapeHtml(row.foodEntry)}</td>
            <td data-label="Start">${escapeHtml(row.startGlucose)}</td>
            <td data-label="Peak 2h">${escapeHtml(row.highestGlucoseNext2Hours)}</td>
            <td data-label="Increase">${escapeHtml(row.glucoseIncrease)}</td>
          </tr>
        `
      )
      .join("");
  }

  function syncActiveRangeButton() {
    rangeButtons.forEach((button) => {
      const value = button.dataset.rangeHours;
      const buttonHours = value === "all" ? "all" : Number(value);
      button.classList.toggle("is-active", buttonHours === selectedHours);
    });
  }

  function setResponsiveDimensions() {
    if (isMobile()) {
      height = 390;
      chartHeight = 292;
    } else {
      height = 450;
      chartHeight = 344;
    }
  }

  function chartWidth() {
    const innerWidth = viewportWidth() - margin.left - margin.right;
    if (selectedHours === "all") return viewportWidth();
    return Math.max(viewportWidth(), Math.round((allHours / selectedHours) * innerWidth) + margin.left + margin.right);
  }

  function draw() {
    setResponsiveDimensions();
    syncActiveRangeButton();
    const maxScrollBefore = Math.max(1, scroll.scrollWidth - scroll.clientWidth);
    const scrollRatio = Number.isFinite(pendingScrollRatio)
      ? pendingScrollRatio
      : scroll.scrollLeft / maxScrollBefore;
    const width = chartWidth();

    svg.attr("width", width).attr("height", height);
    yAxisSvg.attr("width", 72).attr("height", height);
    svg.selectAll("*").remove();
    yAxisSvg.selectAll("*").remove();

    xScale = d3.scaleTime().domain([minDate, maxDate]).range([margin.left, width - margin.right]);
    yScale = d3
      .scaleLinear()
      .domain([
        Math.min(55, d3.min(readings, (d) => d.glucose) - 8),
        Math.max(165, d3.max(readings, (d) => d.glucose) + 8),
      ])
      .nice()
      .range([margin.top + chartHeight, margin.top]);

    const xAxis = d3
      .axisBottom(xScale)
      .ticks(d3.timeHour.every(4))
      .tickSizeOuter(0)
      .tickFormat((date) => {
        const time = d3.timeFormat("%-I %p")(date);
        return date.getHours() === 0 ? `${d3.timeFormat("%b %-d")(date)} ${time}` : time;
      });
    const yAxis = d3.axisLeft(yScale).ticks(7).tickSizeOuter(0);
    const grid = d3
      .axisLeft(yScale)
      .ticks(7)
      .tickSize(-(width - margin.left - margin.right))
      .tickFormat("");

    svg
      .append("rect")
      .attr("class", "threshold-band-low")
      .attr("x", margin.left)
      .attr("y", yScale(70))
      .attr("width", width - margin.left - margin.right)
      .attr("height", Math.max(0, yScale(yScale.domain()[0]) - yScale(70)));
    svg
      .append("rect")
      .attr("class", "threshold-band-target")
      .attr("x", margin.left)
      .attr("y", yScale(140))
      .attr("width", width - margin.left - margin.right)
      .attr("height", Math.max(0, yScale(70) - yScale(140)));
    svg
      .append("rect")
      .attr("class", "threshold-band-high")
      .attr("x", margin.left)
      .attr("y", yScale(yScale.domain()[1]))
      .attr("width", width - margin.left - margin.right)
      .attr("height", Math.max(0, yScale(140) - yScale(yScale.domain()[1])));

    svg.append("g").attr("class", "grid").attr("transform", `translate(${margin.left},0)`).call(grid);
    svg
      .append("g")
      .attr("class", "axis x-axis")
      .attr("transform", `translate(0,${margin.top + chartHeight})`)
      .call(xAxis);
    yAxisSvg.append("rect").attr("class", "sticky-y-axis-bg").attr("width", 72).attr("height", height);
    yAxisSvg.append("g").attr("class", "axis sticky-axis").attr("transform", `translate(${margin.left},0)`).call(yAxis);

    yAxisSvg
      .append("text")
      .attr("x", margin.left)
      .attr("y", 18)
      .attr("fill", "#92a4b8")
      .attr("font-size", 12)
      .attr("font-weight", 700)
      .text("mg/dL");

    const area = d3
      .area()
      .defined((d) => Number.isFinite(d.glucose))
      .x((d) => xScale(d.date))
      .y0(yScale(yScale.domain()[0]))
      .y1((d) => yScale(d.glucose));
    const line = d3
      .line()
      .defined((d) => Number.isFinite(d.glucose))
      .x((d) => xScale(d.date))
      .y((d) => yScale(d.glucose));

    svg.append("path").datum(readings).attr("class", "glucose-area").attr("d", area);
    svg.append("path").datum(readings).attr("class", "glucose-line").attr("d", line);

    const responseLayer = svg.append("g");
    if (responseEvent) {
      const responseStart = responseEvent.date;
      const responseEnd = new Date(responseStart.getTime() + 3 * 36e5);
      responseLayer
        .append("rect")
        .attr("class", "response-window")
        .attr("x", xScale(responseStart))
        .attr("y", margin.top)
        .attr("width", Math.max(0, xScale(responseEnd) - xScale(responseStart)))
        .attr("height", chartHeight);
    }

    const eventGroup = svg.append("g");
    const eventY = {
      food: yScale(135),
      drink: yScale(126),
    };

    eventGroup
      .selectAll(".event-stem")
      .data(events)
      .join("line")
      .attr("class", (d) => `event-stem category-${d.category}`)
      .attr("x1", (d) => xScale(d.date))
      .attr("x2", (d) => xScale(d.date))
      .attr("y1", margin.top)
      .attr("y2", margin.top + chartHeight)
      .on("mouseenter", (_, d) => describeEvent(d))
      .on("click", (_, d) => {
        responseEvent = d;
        describeEvent(d);
        pendingScrollRatio = scroll.scrollLeft / Math.max(1, scroll.scrollWidth - scroll.clientWidth);
        draw();
      });

    eventGroup
      .selectAll(".event-hit")
      .data(events)
      .join("rect")
      .attr("class", "event-hit")
      .attr("x", (d) => xScale(d.date) - 9)
      .attr("y", (d) => eventY[d.category] - 28)
      .attr("width", 18)
      .attr("height", 42)
      .on("mouseenter", (_, d) => describeEvent(d))
      .on("click", (_, d) => {
        responseEvent = d;
        describeEvent(d);
        pendingScrollRatio = scroll.scrollLeft / Math.max(1, scroll.scrollWidth - scroll.clientWidth);
        draw();
      });

    hoverLayer = svg.append("g");
    hoverLine = hoverLayer
      .append("line")
      .attr("class", "hover-line")
      .attr("y1", margin.top)
      .attr("y2", margin.top + chartHeight);
    hoverDot = hoverLayer.append("circle").attr("class", "hover-dot").attr("r", 5);

    svg
      .append("rect")
      .attr("x", margin.left)
      .attr("y", margin.top)
      .attr("width", width - margin.left - margin.right)
      .attr("height", chartHeight)
      .attr("fill", "transparent")
      .on("mouseenter", (event) => {
        updateCrosshairFromClientX(event.clientX);
      })
      .on("mousemove", (event) => {
        updateCrosshairFromClientX(event.clientX);
      });

    const maxScrollAfter = Math.max(0, scroll.scrollWidth - scroll.clientWidth);
    scroll.scrollLeft = Math.round(maxScrollAfter * scrollRatio);
    pendingScrollRatio = scroll.scrollLeft / Math.max(1, maxScrollAfter);
    updateViewportSummary();
  }

  rangeButtons.forEach((button) => {
    button.addEventListener("click", () => {
      const value = button.dataset.rangeHours;
      selectedHours = value === "all" ? "all" : Number(value);
      userSelectedRange = true;
      pendingScrollRatio = scroll.scrollLeft / Math.max(1, scroll.scrollWidth - scroll.clientWidth);
      rangeButtons.forEach((item) => item.classList.toggle("is-active", item === button));
      responseEvent = null;
      draw();
    });
  });

  timelineSlider.addEventListener("input", () => {
    sliderIsDragging = true;
    const maxScroll = Math.max(0, scroll.scrollWidth - scroll.clientWidth);
    scroll.scrollLeft = Math.round((Number(timelineSlider.value) / Number(timelineSlider.max)) * maxScroll);
    updateViewportSummary();
  });
  timelineSlider.addEventListener("change", () => {
    sliderIsDragging = false;
    updateViewportSummary();
  });

  scroll.addEventListener("scroll", updateViewportSummary);
  window.addEventListener("resize", () => {
    const nowIsMobile = isMobile();
    if (nowIsMobile !== lastIsMobile && !userSelectedRange) {
      selectedHours = nowIsMobile ? 72 : 168;
      lastIsMobile = nowIsMobile;
    }
    pendingScrollRatio = scroll.scrollLeft / Math.max(1, scroll.scrollWidth - scroll.clientWidth);
    draw();
  });

  renderFoodResponseTable(foodResponseRows);
  draw();
})();
