const fs = require("fs");
const path = require("path");

const repoRoot = path.resolve(__dirname, "..");
const glucoseSource =
  process.argv[2] ||
  "/Users/jhuck/Documents/work/Glucose/lingo-glucose-data-2026-APR-25.csv";
const journalSource =
  process.argv[3] ||
  "/Users/jhuck/Documents/work/Glucose/food_journal_complete.md";
const outputDir = path.join(repoRoot, "assets", "data");

function parseCsvLine(line) {
  const values = [];
  let current = "";
  let quoted = false;

  for (let index = 0; index < line.length; index += 1) {
    const character = line[index];
    const next = line[index + 1];

    if (character === '"' && quoted && next === '"') {
      current += '"';
      index += 1;
    } else if (character === '"') {
      quoted = !quoted;
    } else if (character === "," && !quoted) {
      values.push(current);
      current = "";
    } else {
      current += character;
    }
  }

  values.push(current);
  return values.map((value) => value.trim());
}

function readGlucoseReadings(source) {
  const [, ...rows] = fs.readFileSync(source, "utf8").trim().split(/\r?\n/);

  return rows
    .map((row) => {
      const [timestamp, measurement] = parseCsvLine(row);
      const glucose = Number(measurement);

      if (!timestamp || !Number.isFinite(glucose)) {
        return null;
      }

      return {
        timestamp,
        glucose,
      };
    })
    .filter(Boolean)
    .sort((left, right) => new Date(left.timestamp) - new Date(right.timestamp));
}

function classifyEntry(entry) {
  const text = entry.toLowerCase();

  if (/\b(light headed|dizzy|nausea|sick|not feeling|headache|symptom)\b/.test(text)) {
    return "note";
  }
  if (/\b(coffee|tea|spindrift|waterloo|gatorade|shake|water|milk|juice)\b/.test(text)) {
    return "drink";
  }

  return "food";
}

function readJournalEvents(source) {
  return fs
    .readFileSync(source, "utf8")
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter((line) => line.startsWith("|") && !line.includes("---"))
    .map((line) => {
      const cells = line
        .split("|")
        .slice(1, -1)
        .map((cell) => cell.trim());

      if (cells[0] === "timestamp" || cells.length < 2) {
        return null;
      }

      const [timestamp, entry] = cells;
      if (!timestamp || !entry || Number.isNaN(new Date(timestamp).getTime())) {
        return null;
      }

      return {
        timestamp,
        entry,
        category: classifyEntry(entry),
      };
    })
    .filter(Boolean)
    .sort((left, right) => new Date(left.timestamp) - new Date(right.timestamp));
}

function summarize(readings, events) {
  return {
    generatedAt: new Date().toISOString(),
    readingCount: readings.length,
    eventCount: events.length,
    glucoseStart: readings.length ? readings[0].timestamp : null,
    glucoseEnd: readings.length ? readings[readings.length - 1].timestamp : null,
    journalStart: events.length ? events[0].timestamp : null,
    journalEnd: events.length ? events[events.length - 1].timestamp : null,
  };
}

function nearestReading(readings, date) {
  let low = 0;
  let high = readings.length - 1;
  const target = date.getTime();

  while (low < high) {
    const middle = Math.floor((low + high) / 2);
    if (new Date(readings[middle].timestamp).getTime() < target) {
      low = middle + 1;
    } else {
      high = middle;
    }
  }

  const candidates = [readings[low], readings[low - 1]].filter(Boolean);
  return candidates.reduce((best, item) => {
    const bestDistance = Math.abs(new Date(best.timestamp).getTime() - target);
    const itemDistance = Math.abs(new Date(item.timestamp).getTime() - target);
    return itemDistance < bestDistance ? item : best;
  }, candidates[0]);
}

function formatDateTime(timestamp) {
  return new Intl.DateTimeFormat("en-US", {
    weekday: "short",
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
    timeZoneName: "short",
  }).format(new Date(timestamp));
}

function groupFoodEvents(events) {
  const foodEvents = events.filter((event) => event.category === "food");
  const groups = [];

  foodEvents.forEach((event) => {
    const current = groups[groups.length - 1];
    if (!current) {
      groups.push([event]);
      return;
    }

    const previousEvent = current[current.length - 1];
    const gapMinutes =
      (new Date(event.timestamp).getTime() - new Date(previousEvent.timestamp).getTime()) / 60000;

    if (gapMinutes <= 60) {
      current.push(event);
    } else {
      groups.push([event]);
    }
  });

  return groups;
}

function buildFoodResponseSummary(readings, events) {
  return groupFoodEvents(events).map((group) => {
    const startEvent = group[0];
    const startTime = new Date(startEvent.timestamp);
    const endTime = new Date(startTime.getTime() + 120 * 60000);
    const baseline = nearestReading(readings, startTime);
    const windowReadings = readings.filter((reading) => {
      const readingTime = new Date(reading.timestamp);
      return readingTime >= startTime && readingTime <= endTime;
    });
    const peak = windowReadings.length
      ? windowReadings.reduce((best, reading) => (reading.glucose > best.glucose ? reading : best))
      : null;
    const startGlucose = baseline ? baseline.glucose : null;
    const peakGlucose = peak ? peak.glucose : null;

    return {
      timestamp: startEvent.timestamp,
      dateTime: formatDateTime(startEvent.timestamp),
      foodEntry: group.map((event) => event.entry).join(" + "),
      startGlucose,
      highestGlucoseNext2Hours: peakGlucose,
      glucoseIncrease:
        startGlucose != null && peakGlucose != null ? peakGlucose - startGlucose : null,
    };
  });
}

fs.mkdirSync(outputDir, { recursive: true });

const readings = readGlucoseReadings(glucoseSource);
const events = readJournalEvents(journalSource);
const summary = summarize(readings, events);
const foodResponseSummary = buildFoodResponseSummary(readings, events);

fs.writeFileSync(
  path.join(outputDir, "glucose-readings.json"),
  `${JSON.stringify(readings, null, 2)}\n`
);
fs.writeFileSync(
  path.join(outputDir, "food-events.json"),
  `${JSON.stringify(events, null, 2)}\n`
);
fs.writeFileSync(
  path.join(outputDir, "glucose-summary.json"),
  `${JSON.stringify(summary, null, 2)}\n`
);
fs.writeFileSync(
  path.join(outputDir, "food-response-summary.json"),
  `${JSON.stringify(foodResponseSummary, null, 2)}\n`
);

console.log(
  `Wrote ${summary.readingCount} glucose readings, ${summary.eventCount} journal events, and ${foodResponseSummary.length} food response rows to ${path.relative(
    repoRoot,
    outputDir
  )}.`
);
