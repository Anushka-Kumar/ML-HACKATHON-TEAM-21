
import { useEffect, useState } from "react";

import {
  Activity,
  ArrowRight,
  ArrowUpRight,
  CheckCircle2,
  ChevronDown,
  CircleHelp,
  Clock3,
  Compass,
  GitBranch,
  Layers3,
  Map,
  MapPin,
  Navigation,
  Route,
  Settings2,
  ShieldCheck,
  Sparkles,
  Wifi,
  XCircle,
} from "lucide-react";
import RouteMap from "./components/RouteMap";
import { getHealth, getTrips, predictTrip } from "./api";
import "./App.css";

const navigationItems = [
  { id: "overview", label: "Overview", icon: Layers3 },
  { id: "trips", label: "Evaluation Trips", icon: MapPin },
  { id: "routes", label: "Route Explorer", icon: Route },
  { id: "results", label: "Evaluation Results", icon: Activity },
];

function App() {
  const [activePage, setActivePage] = useState("overview");
  const [selectedTrip, setSelectedTrip] = useState("");
  const [requestStatus, setRequestStatus] = useState("idle");
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [trips, setTrips] = useState([]);
  const [health, setHealth] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [errorMessage, setErrorMessage] = useState("");
  const [inferenceMs, setInferenceMs] = useState(null);

  useEffect(() => {
    Promise.all([getTrips(), getHealth()])
      .then(([tripData, healthData]) => {
        setTrips(
          tripData.trips.map((id) => ({
            id,
            name: `Evaluation Trip ${id}`,
          }))
        );
        setHealth(healthData);
      })
      .catch((error) => {
        setErrorMessage(error.message);
        setHealth({ status: "error", model_ready: false });
      });
  }, []);

  const selectedTripData = trips.find((trip) => trip.id === selectedTrip);

  const navigate = (page) => {
    setActivePage(page);
    setSidebarOpen(false);
  };

  const handleTripChange = (event) => {
    setSelectedTrip(event.target.value);
    setRequestStatus("idle");
    setPrediction(null);
    setErrorMessage("");
    setInferenceMs(null);
  };

  const handleReconstruct = async () => {
    if (!selectedTrip) return;

    setRequestStatus("pending");
    setErrorMessage("");
    setPrediction(null);
    const startedAt = performance.now();

    try {
      const result = await predictTrip(selectedTrip);
      setPrediction(result);
      setInferenceMs(Math.round(performance.now() - startedAt));
      setRequestStatus("success");
    } catch (error) {
      setErrorMessage(error.message);
      setRequestStatus("error");
    }
  };

  const pageTitles = {
    overview: "Overview",
    trips: "Evaluation Trips",
    routes: "Route Explorer",
    results: "Evaluation Results",
  };

  return (
    <div className="app-shell">
      {sidebarOpen && (
        <div
          className="mobile-overlay"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      <aside className={`sidebar ${sidebarOpen ? "sidebar-open" : ""}`}>
        <div className="brand">
          <div className="brand-icon">
            <Navigation size={22} />
          </div>

          <div>
            <h2>pathsnap<span>.</span></h2>
            <p>INTELLIGENT MAP MATCHING</p>
          </div>

          <button
            className="mobile-close"
            onClick={() => setSidebarOpen(false)}
            aria-label="Close menu"
          >
            <XCircle size={20} />
          </button>
        </div>

        <div className="workspace-label">WORKSPACE</div>

        <nav className="nav-menu">
          {navigationItems.map((item) => {
            const Icon = item.icon;

            return (
              <button
                key={item.id}
                className={`nav-item ${
                  activePage === item.id ? "active" : ""
                }`}
                onClick={() => navigate(item.id)}
              >
                <Icon size={18} />
                {item.label}
              </button>
            );
          })}
        </nav>

        <div className="sidebar-bottom">
          <div className="help-card">
            <div className="help-icon">
              <CircleHelp size={18} />
            </div>
            <h4>Need assistance?</h4>
            <p>Explore the PathSnap evaluation workflow.</p>
          </div>

          <div className="profile">
            <div className="avatar">PS</div>
            <div className="profile-info">
              <strong>PathSnap Evaluator</strong>
              <span>Evaluation workspace</span>
            </div>
            <Settings2 size={17} />
          </div>
        </div>
      </aside>

      <main className="main-content">
        <header className="topbar">
          <div className="topbar-left">
            <button
              className="mobile-menu"
              onClick={() => setSidebarOpen(true)}
              aria-label="Open menu"
            >
              <Layers3 size={20} />
            </button>

            <div className="breadcrumb">
              Workspace <span>/</span> {pageTitles[activePage]}
            </div>
          </div>

          <div className="topbar-right">
            <div className="environment">
              <span className="status-dot"></span>
              Evaluation environment
            </div>

            <button className="icon-button" aria-label="Help">
              <CircleHelp size={19} />
            </button>
          </div>
        </header>

        <section className="page-content">

          {activePage === "overview" && (
            <>
              <div className="welcome-row">
                <div>
                  <div className="eyebrow">
                    <Sparkles size={14} />
                    GPS INTELLIGENCE PLATFORM
                  </div>

                  <h1>
                    Route reconstruction,
                    <br />
                    <span>reimagined.</span>
                  </h1>

                  <p className="welcome-description">
                    Transform noisy GPS trajectories into meaningful
                    road-network routes with intelligent map matching.
                  </p>
                </div>

                <div className="welcome-decoration">
                  <Compass size={78} strokeWidth={0.8} />
                  <span className="orbit orbit-one"></span>
                  <span className="orbit orbit-two"></span>
                </div>
              </div>

              <div className="section-heading">
                <div>
                  <h2>Evaluation workspace</h2>
                  <p>
                    Select a reserved trip to begin route reconstruction.
                  </p>
                </div>

                <div className="session-tag">
                  <ShieldCheck size={15} />
                  Evaluator interface
                </div>
              </div>

              <TripSelector
                selectedTrip={selectedTrip}
                selectedTripData={selectedTripData}
                requestStatus={requestStatus}
                trips={trips}
                errorMessage={errorMessage}
                onTripChange={handleTripChange}
                onReconstruct={handleReconstruct}
              />

              <RouteVisualization predictions={prediction?.predictions} />

              <div className="bottom-grid">
                <InfoCard
                  icon={<Activity size={20} />}
                  color="blue"
                  label="Prediction status"
                  value={
                    requestStatus === "success"
                      ? "Complete"
                      : requestStatus === "pending"
                        ? "Running"
                        : "Not started"
                  }
                  description={
                    prediction
                      ? `${prediction.predictions.length} observations`
                      : "Waiting for trip selection"
                  }
                />

                <InfoCard
                  icon={<Clock3 size={20} />}
                  color="purple"
                  label="Inference time"
                  value={inferenceMs === null ? "—" : `${inferenceMs} ms`}
                  description="Browser round-trip time"
                />

                <InfoCard
                  icon={<Wifi size={20} />}
                  color="green"
                  label="Backend connection"
                  value={health?.status === "ok" ? "Connected" : health ? "Unavailable" : "Checking"}
                  description={
                    health?.model_ready
                      ? "Model ready"
                      : "Model may require integration"
                  }
                />
              </div>
            </>
          )}

          {activePage === "trips" && (
            <>
              <PageHeading
                eyebrow="EVALUATION WORKSPACE"
                title="Evaluation trips"
                description="Select a reserved GPS trajectory for map matching."
              />

              <TripSelector
                selectedTrip={selectedTrip}
                selectedTripData={selectedTripData}
                requestStatus={requestStatus}
                trips={trips}
                errorMessage={errorMessage}
                onTripChange={handleTripChange}
                onReconstruct={handleReconstruct}
              />

              <div className="section-heading">
                <div>
                  <h2>Available trip previews</h2>
                  <p>
                    Trips are loaded directly from the backend.
                  </p>
                </div>
                <span className="session-tag">
                  {trips.length} available {trips.length === 1 ? "trip" : "trips"}
                </span>
              </div>

              <div className="trip-list">
                {trips.map((trip, index) => (
                  <button
                    key={trip.id}
                    className={`trip-list-item ${
                      selectedTrip === trip.id ? "trip-selected" : ""
                    }`}
                    onClick={() => {
                      setSelectedTrip(trip.id);
                      setRequestStatus("idle");
                    }}
                  >
                    <div className="trip-number">
                      {String(index + 1).padStart(2, "0")}
                    </div>

                    <div className="trip-list-info">
                      <strong>{trip.name}</strong>
                      <span>{trip.id} · Backend dataset trip</span>
                    </div>

                    {selectedTrip === trip.id ? (
                      <CheckCircle2 size={19} className="selected-check" />
                    ) : (
                      <ArrowRight size={18} />
                    )}
                  </button>
                ))}
              </div>
            </>
          )}

          {activePage === "routes" && (
            <>
              <PageHeading
                eyebrow="GEOSPATIAL VISUALIZATION"
                title="Route explorer"
                description="Explore the original GPS trajectory and reconstructed road-network route."
              />

              <div className="trip-panel">
                <div className="panel-heading">
                  <div className="panel-icon">
                    <MapPin size={19} />
                  </div>
                  <div>
                    <h3>Selected trajectory</h3>
                    <p>
                      {selectedTripData
                        ? selectedTripData.name
                        : "No evaluation trip selected"}
                    </p>
                  </div>
                </div>
              </div>

              <RouteVisualization predictions={prediction?.predictions} />
            </>
          )}

          {activePage === "results" && (
            <>
              <PageHeading
                eyebrow="MODEL EVALUATION"
                title="Evaluation results"
                description="Review route reconstruction and model performance."
              />

              <div className={prediction ? "results-empty results-ready" : "results-empty"}>
                <div className="results-empty-icon">
                  <Activity size={30} />
                </div>

                <h2>
                  {prediction
                    ? `Results for ${prediction.trip_id}`
                    : "Evaluation results will appear here"}
                </h2>

                <p>
                  {prediction?.evaluation?.status === "available"
                    ? "The backend returned observation-level evaluation metrics."
                    : prediction?.evaluation?.reason ||
                      "Actual metrics will be displayed after the backend returns model predictions and evaluation results."}
                </p>

                <button
                  className="secondary-button"
                  onClick={() => navigate("trips")}
                >
                  Select an evaluation trip
                  <ArrowUpRight size={16} />
                </button>
              </div>

              <div className="bottom-grid">
                <InfoCard
                  icon={<Activity size={20} />}
                  color="blue"
                  label="Top-1 accuracy"
                  value={
                    prediction?.evaluation?.classifier_top1?.pointwise_accuracy != null
                      ? `${(prediction.evaluation.classifier_top1.pointwise_accuracy * 100).toFixed(1)}%`
                      : "—"
                  }
                  description="Classifier pointwise accuracy"
                />

                <InfoCard
                  icon={<GitBranch size={20} />}
                  color="purple"
                  label="Route continuity"
                  value={
                    prediction?.evaluation?.viterbi?.route_continuity != null
                      ? `${(prediction.evaluation.viterbi.route_continuity * 100).toFixed(1)}%`
                      : "—"
                  }
                  description="Viterbi route continuity"
                />

                <InfoCard
                  icon={<Clock3 size={20} />}
                  color="green"
                  label="Inference time"
                  value={inferenceMs === null ? "—" : `${inferenceMs} ms`}
                  description="Browser round-trip time"
                />
              </div>
            </>
          )}

          <footer className="footer">
            <span>PATHSNAP © 2026</span>
            <span>
              Distributed Map Matching & Route Reconstruction
            </span>
          </footer>
        </section>
      </main>
    </div>
  );
}

function PageHeading({ eyebrow, title, description }) {
  return (
    <div className="page-heading">
      <div className="eyebrow">
        <Sparkles size={14} />
        {eyebrow}
      </div>

      <h1>{title}</h1>
      <p>{description}</p>
    </div>
  );
}

function TripSelector({
  selectedTrip,
  selectedTripData,
  requestStatus,
  trips,
  errorMessage,
  onTripChange,
  onReconstruct,
}) {
  return (
    <section className="trip-panel" id="trips">
      <div className="panel-heading">
        <div className="panel-icon">
          <MapPin size={19} />
        </div>

        <div>
          <h3>Select an evaluation trip</h3>
          <p>Choose an evaluation trip loaded from the backend dataset.</p>
        </div>
      </div>

      <div className="selection-area">
        <div className="select-wrapper">
          <Map size={19} />

          <select value={selectedTrip} onChange={onTripChange}>
            <option value="">Choose an evaluation trip</option>

            {trips.map((trip) => (
              <option key={trip.id} value={trip.id}>
                {trip.name} ({trip.id})
              </option>
            ))}
          </select>

          <ChevronDown size={17} className="select-chevron" />
        </div>

        <button
          className="process-button"
          disabled={!selectedTrip}
          onClick={onReconstruct}
        >
          <Route size={17} />
          Reconstruct route
          <ArrowUpRight size={17} />
        </button>
      </div>

      <div className="notice">
        <Activity size={16} />

        {selectedTripData
          ? `Selected: ${selectedTripData.name} (${selectedTripData.id}).`
          : "Select a trip to prepare a reconstruction request."}
      </div>

      {requestStatus === "pending" && (
        <div className="pending-message">
          <Clock3 size={17} />
          Requesting route reconstruction from the backend...
        </div>
      )}

      {requestStatus === "success" && (
        <div className="pending-message">
          <CheckCircle2 size={17} />
          Prediction returned successfully.
        </div>
      )}

      {requestStatus === "error" && (
        <div className="pending-message">
          <XCircle size={17} />
          {errorMessage || "The prediction request failed."}
        </div>
      )}
    </section>
  );
}
function RouteVisualization({ predictions = [] }) {
  return (
    <>
      <div className="section-heading map-heading">
        <div>
          <h2>Route visualization</h2>
          <p>
            Geographical representation of the selected trajectory.
          </p>
        </div>

        <div className="map-label">
          <span className={`status-dot ${predictions.length ? "" : "muted"}`}></span>
          {predictions.length ? `${predictions.length} observations loaded` : "Awaiting prediction"}
        </div>
      </div>

      <RouteMap predictions={predictions} />
    </>
  );
}

function InfoCard({ icon, color, label, value, description }) {
  return (
    <div className="info-card">
      <div className={`info-card-icon ${color}`}>
        {icon}
      </div>

      <div>
        <span>{label}</span>
        <strong>{value}</strong>
        <p>{description}</p>
      </div>
    </div>
  );
}

export default App;
