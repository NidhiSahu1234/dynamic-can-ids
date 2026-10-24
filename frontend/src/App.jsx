import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "./api.js";

const SOURCES = ["normal", "dos", "fuzzy", "gear", "rpm"];
const FILTERS = ["all", "normal", "attack", "anomaly"];

function count(value) {
  return Number(value ?? 0).toLocaleString();
}

function metric(value) {
  return Number.isFinite(value) ? (value * 100).toFixed(1) + "%" : "—";
}

function TableEmpty({ children }) {
  return <p className="empty">{children}</p>;
}

export default function App() {
  const [health, setHealth] = useState(null);
  const [stats, setStats] = useState(null);
  const [alerts, setAlerts] = useState([]);
  const [metrics, setMetrics] = useState(null);
  const [rules, setRules] = useState([]);
  const [results, setResults] = useState([]);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [source, setSource] = useState("normal");
  const [simRunning, setSimRunning] = useState(false);
  const [resultFilter, setResultFilter] = useState("all");
  const [ruleSignature, setRuleSignature] = useState("");
  const [ruleType, setRuleType] = useState("custom");
  const [ruleStatus, setRuleStatus] = useState("suspected");
  const [ruleBusy, setRuleBusy] = useState(false);
  const [ruleMessage, setRuleMessage] = useState("");
  const skipRef = useRef(0);
  const timerRef = useRef(null);
  const batchBusyRef = useRef(false);

  const refresh = useCallback(async () => {
    try {
      const [h, s, a, m, r] = await Promise.all([
        api.health(),
        api.stats(),
        api.alerts(40),
        api.metrics().catch(() => null),
        api.rules(),
      ]);
      setHealth(h);
      setStats(s);
      setAlerts(a);
      setMetrics(m);
      setRules(r);
      setError("");
    } catch (e) {
      setError(e.message);
    }
  }, []);

  useEffect(() => {
    refresh();
    const id = setInterval(refresh, 8000);
    return () => clearInterval(id);
  }, [refresh]);

  const runBatch = async () => {
    if (batchBusyRef.current || !health?.models_loaded) return;
    batchBusyRef.current = true;
    setLoading(true);
    setError("");
    try {
      let sample = await api.trafficSample(source, skipRef.current, 20);
      if (sample.count === 0 && skipRef.current > 0) {
        skipRef.current = 0;
        sample = await api.trafficSample(source, 0, 20);
      }
      if (sample.count === 0) {
        throw new Error("No traffic rows are available for " + source + ".");
      }
      skipRef.current += sample.count;
      const frames = sample.frames.map(({ timestamp, can_id, dlc, data }) => ({
        timestamp,
        can_id,
        dlc,
        data,
      }));
      const batch = await api.predictBatch(frames);
      setResults((previous) =>
        [...batch.results.map((result, index) => ({ ...result, ...sample.frames[index] })), ...previous]
          .slice(0, 100)
      );
      await refresh();
    } catch (e) {
      setError(e.message);
    } finally {
      batchBusyRef.current = false;
      setLoading(false);
    }
  };

  useEffect(() => {
    if (!simRunning) {
      if (timerRef.current) clearInterval(timerRef.current);
      return undefined;
    }
    runBatch();
    timerRef.current = setInterval(runBatch, 3000);
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, [simRunning, source, health?.models_loaded]);

  const addRule = async (event) => {
    event.preventDefault();
    setRuleBusy(true);
    setRuleMessage("");
    try {
      await api.addRule({
        signature: ruleSignature,
        attack_type: ruleType,
        status: ruleStatus,
      });
      setRuleSignature("");
      setRuleMessage(ruleStatus === "confirmed"
        ? "Confirmed signature added to the Bloom filter."
        : "Suspected rule saved for review; it does not trigger the Bloom filter.");
      await refresh();
    } catch (e) {
      setRuleMessage(e.message);
    } finally {
      setRuleBusy(false);
    }
  };

  const removeRule = async (ruleId) => {
    setRuleMessage("");
    try {
      await api.deleteRule(ruleId);
      await refresh();
    } catch (e) {
      setRuleMessage(e.message);
    }
  };

  const backendState = !health
    ? "checking"
    : health.models_loaded
      ? "connected"
      : "degraded";

  const visibleResults = resultFilter === "all"
    ? results
    : results.filter((result) => result.predicted_class === resultFilter);
  const dt = metrics?.decision_tree ?? {};
  const dtHeldout = dt.heldout_pretraining_sample;
  const dtUnseen = dt.unseen_attack_categories ?? {};
  const ae = metrics?.autoencoder ?? {};

  return (
    <main className="app">
      <header className="topbar">
        <div>
          <p className="eyebrow">VEHICLE NETWORK MONITOR</p>
          <h1>Dynamic CAN Intrusion Detection</h1>
          <p className="subhead">Decision tree, anomaly scoring, and confirmed signature checks</p>
        </div>
        <div className={"status " + backendState} role="status">
          <span className="status-dot" />
          API {backendState} · {health?.models_loaded ? "models ready" : health ? "models unavailable" : "checking"}
        </div>
      </header>

      {error && <div className="error-banner" role="alert">{error}</div>}
      {health && !health.models_loaded && (
        <div className="notice">
          <strong>Inference is unavailable.</strong>{" "}
          {health.model_error || "Train the models to create the required artifacts."}
        </div>
      )}

      <section className="cards" aria-label="Session summary">
        <article className="card accent-cyan">
          <span className="label">Frames analyzed this session</span>
          <strong className="value">{count(stats?.frames_processed_session)}</strong>
          <span className="card-foot">Current backend session</span>
        </article>
        <article className="card accent-green">
          <span className="label">Normal frames</span>
          <strong className="value">{count(stats?.normal_processed_session)}</strong>
          <span className="card-foot">Classified as normal</span>
        </article>
        <article className="card accent-red">
          <span className="label">Known attacks</span>
          <strong className="value">{count(stats?.attack_processed_session)}</strong>
          <span className="card-foot">Decision tree or confirmed signature</span>
        </article>
        <article className="card accent-amber">
          <span className="label">Suspected anomalies</span>
          <strong className="value">{count(stats?.anomaly_processed_session)}</strong>
          <span className="card-foot">Autoencoder flagged for review</span>
        </article>
        <article className="card accent-violet">
          <span className="label">Stored rules</span>
          <strong className="value">{count(stats?.rules_count)}</strong>
          <span className="card-foot">Confirmed and suspected</span>
        </article>
      </section>

      <section className="panel">
        <div className="panel-heading">
          <div>
            <p className="eyebrow">DATASET PLAYBACK</p>
            <h2>Traffic analysis</h2>
          </div>
          <span className="live-label"><span /> Stored CSV rows</span>
        </div>
        <p className="muted">
          Replays processed dataset records through the API. This is simulated traffic, not a live vehicle connection.
        </p>
        <div className="controls">
          <label>
            Dataset
            <select value={source} onChange={(event) => {
              setSource(event.target.value);
              skipRef.current = 0;
            }}>
              {SOURCES.map((item) => <option key={item} value={item}>{item.toUpperCase()}</option>)}
            </select>
          </label>
          <button className="button primary" onClick={runBatch} disabled={loading || !health?.models_loaded}>
            {loading ? "Analyzing…" : "Analyze next 20"}
          </button>
          <button
            className={"button " + (simRunning ? "stop" : "secondary")}
            onClick={() => setSimRunning((running) => !running)}
            disabled={!health?.models_loaded}
          >
            {simRunning ? "Stop playback" : "Start playback"}
          </button>
          <span className="muted small">Playback reads the selected processed CSV.</span>
        </div>
      </section>

      <section className="panel">
        <div className="panel-heading">
          <div>
            <p className="eyebrow">FRAME LEVEL</p>
            <h2>Detection results</h2>
          </div>
          <label className="filter-control">
            Show
            <select value={resultFilter} onChange={(event) => setResultFilter(event.target.value)}>
              {FILTERS.map((item) => <option key={item} value={item}>{item}</option>)}
            </select>
          </label>
        </div>
        {visibleResults.length === 0 ? (
          <TableEmpty>{results.length ? "No frames match this filter." : "Run a batch to see CAN frame detections."}</TableEmpty>
        ) : (
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>CAN ID</th>
                  <th>Dataset label</th>
                  <th>Detection</th>
                  <th>Method</th>
                  <th>Anomaly score / threshold</th>
                  <th>DT attack probability</th>
                  <th>Severity</th>
                </tr>
              </thead>
              <tbody>
                {visibleResults.map((result, index) => (
                  <tr key={result.can_id + "-" + result.timestamp + "-" + index}>
                    <td className="mono">{result.can_id}</td>
                    <td>{result.label}</td>
                    <td><span className={"badge " + result.predicted_class}>{result.predicted_class}</span></td>
                    <td>{result.detection_method.replaceAll("_", " ")}</td>
                    <td className="mono">
                      {result.anomaly_score != null
                        ? result.anomaly_score.toFixed(5) + " / " + (result.threshold?.toFixed(5) ?? "—")
                        : "—"}
                    </td>
                    <td>{result.dt_probability_attack != null ? metric(result.dt_probability_attack) : "—"}</td>
                    <td><span className={"severity " + result.severity}>{result.severity}</span></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>

      <div className="two-column">
        <section className="panel">
          <div className="panel-heading">
            <div>
              <p className="eyebrow">PERSISTED DETECTIONS</p>
              <h2>Recent alerts</h2>
            </div>
            <span className="count-chip">{count(stats?.alert_count)} total</span>
          </div>
          {alerts.length === 0 ? (
            <TableEmpty>Alerts appear here when a frame is classified as an attack or anomaly.</TableEmpty>
          ) : (
            <div className="table-wrap">
              <table>
                <thead>
                  <tr><th>Time</th><th>CAN ID</th><th>Class</th><th>Method</th><th>Severity</th></tr>
                </thead>
                <tbody>
                  {alerts.map((alert) => (
                    <tr key={alert.alert_id}>
                      <td>{alert.timestamp.slice(11, 19)}</td>
                      <td className="mono">{alert.can_id}</td>
                      <td><span className={"badge " + alert.predicted_class}>{alert.predicted_class}</span></td>
                      <td>{alert.detection_method.replaceAll("_", " ")}</td>
                      <td><span className={"severity " + alert.severity}>{alert.severity}</span></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </section>

        <section className="panel">
          <div className="panel-heading">
            <div>
              <p className="eyebrow">MODEL EVALUATION</p>
              <h2>Measured metrics</h2>
            </div>
          </div>
          {!metrics ? (
            <TableEmpty>Metrics are unavailable until training creates artifacts/metrics.json.</TableEmpty>
          ) : (
            <>
              <p className="muted small">{metrics.evaluation_scope}</p>
              <div className="table-wrap">
                <table>
                  <thead>
                    <tr><th>Evaluation set</th><th>Samples</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1</th></tr>
                  </thead>
                  <tbody>
                    <MetricRow name="DT held-out sample" data={dtHeldout} />
                    <MetricRow name="DT · Fuzzy" data={dtUnseen.Fuzzy} />
                    <MetricRow name="DT · RPM" data={dtUnseen.RPM} />
                    <MetricRow name="AE · Fuzzy" data={ae.Fuzzy} />
                    <MetricRow name="AE · RPM" data={ae.RPM} />
                  </tbody>
                </table>
              </div>
              {ae.normal_validation_sample && (
                <p className="threshold-note">
                  AE threshold: <strong>{ae.normal_validation_sample.threshold?.toFixed(6)}</strong>
                  {" "}· normal validation sample · p{ae.normal_validation_sample.threshold_percentile}
                </p>
              )}
            </>
          )}
        </section>
      </div>

      <section className="panel">
        <div className="panel-heading">
          <div>
            <p className="eyebrow">SIGNATURE CONTROL</p>
            <h2>Dynamic attack rules</h2>
          </div>
          <span className="count-chip">{rules.length} stored</span>
        </div>
        <p className="muted">
          Suspected rules stay in review. Only a confirmed signature enters the Bloom filter and can trigger a signature match.
        </p>
        <form className="rule-form" onSubmit={addRule}>
          <label className="signature-input">
            Composite signature
            <input
              required
              maxLength={128}
              value={ruleSignature}
              onChange={(event) => setRuleSignature(event.target.value)}
              placeholder="0350|dlc=8|05:28:84:66:6d:00:00:a2"
            />
          </label>
          <label>
            Attack type
            <input
              required
              maxLength={32}
              value={ruleType}
              onChange={(event) => setRuleType(event.target.value)}
              placeholder="dos"
            />
          </label>
          <label>
            Status
            <select value={ruleStatus} onChange={(event) => setRuleStatus(event.target.value)}>
              <option value="suspected">Suspected</option>
              <option value="confirmed">Confirmed</option>
            </select>
          </label>
          <button className="button primary" type="submit" disabled={ruleBusy}>
            {ruleBusy ? "Saving…" : "Add rule"}
          </button>
        </form>
        {ruleMessage && <p className="form-message" role="status">{ruleMessage}</p>}
        {rules.length === 0 ? (
          <TableEmpty>No dynamic rules have been added.</TableEmpty>
        ) : (
          <div className="table-wrap">
            <table>
              <thead><tr><th>Type</th><th>Status</th><th>Signature</th><th>Created</th><th></th></tr></thead>
              <tbody>
                {rules.map((rule) => (
                  <tr key={rule.rule_id}>
                    <td>{rule.attack_type}</td>
                    <td><span className={"badge " + rule.status}>{rule.status}</span></td>
                    <td className="mono signature-cell">{rule.signature}</td>
                    <td>{rule.created_at.slice(0, 10)}</td>
                    <td><button className="button compact" onClick={() => removeRule(rule.rule_id)}>Delete</button></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
      <footer>Dynamic CAN IDS · Research prototype · Dataset playback only</footer>
    </main>
  );
}

function MetricRow({ name, data }) {
  if (!data) return <tr><td>{name}</td><td colSpan="5">No evaluation result</td></tr>;
  return (
    <tr>
      <td>{name}</td>
      <td>{count(data.samples)}</td>
      <td>{metric(data.accuracy)}</td>
      <td>{metric(data.precision)}</td>
      <td>{metric(data.recall)}</td>
      <td>{metric(data.f1)}</td>
    </tr>
  );
}
