
import { useState } from "react";

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
import "./App.css";

const demoTrips = Array.from({ length: 10 }, (_, index) => ({
  id: `DEMO-${String(index + 1).padStart(2, "0")}`,
  name: `Evaluation Trip ${String(index + 1).padStart(2, "0")}`,
}));

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

  const selectedTripData = demoTrips.find(
    (trip) => trip.id === selectedTrip
  );

  const navigate = (page) => {
    setActivePage(page);
    setSidebarOpen(false);
  };

  const handleTripChange = (event) => {
    setSelectedTrip(event.target.value);
    setRequestStatus("idle");
  };

  const handleReconstruct = () => {
    if (!selectedTrip) return;

    // UI-only action. No backend request is made yet.
    setRequestStatus("pending");
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
                onTripChange={handleTripChange}
                onReconstruct={handleReconstruct}
              />

              <RouteVisualization />

              <div className="bottom-grid">
                <InfoCard
                  icon={<Activity size={20} />}
                  color="blue"
                  label="Prediction status"
                  value="Not started"
                  description="Waiting for trip selection"
                />

                <InfoCard
                  icon={<Clock3 size={20} />}
                  color="purple"
                  label="Inference time"
                  value="—"
                  description="Available after prediction"
                />

                <InfoCard
                  icon={<Wifi size={20} />}
                  color="green"
                  label="Backend connection"
                  value="Not connected"
                  description="Integration pending"
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
                onTripChange={handleTripChange}
                onReconstruct={handleReconstruct}
              />

              <div className="section-heading">
                <div>
                  <h2>Available trip previews</h2>
                  <p>
                    These are temporary interface placeholders.
                  </p>
                </div>
                <span className="session-tag">
                  10 demo entries
                </span>
              </div>

              <div className="trip-list">
                {demoTrips.map((trip, index) => (
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
                      <span>{trip.id} · Demo placeholder</span>
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

              <RouteVisualization />
            </>
          )}

          {activePage === "results" && (
            <>
              <PageHeading
                eyebrow="MODEL EVALUATION"
                title="Evaluation results"
                description="Review route reconstruction and model performance."
              />

              <div className="results-empty">
                <div className="results-empty-icon">
                  <Activity size={30} />
                </div>

                <h2>Evaluation results will appear here</h2>

                <p>
                  Actual metrics will be displayed after the backend
                  returns the model predictions and evaluation results.
                  No demonstration scores are being presented as real results.
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
                  value="—"
                  description="Awaiting evaluation"
                />

                <InfoCard
                  icon={<GitBranch size={20} />}
                  color="purple"
                  label="Route continuity"
                  value="—"
                  description="Awaiting evaluation"
                />

                <InfoCard
                  icon={<Clock3 size={20} />}
                  color="green"
                  label="Inference time"
                  value="—"
                  description="Awaiting evaluation"
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
          <p>Choose one of the ten reserved evaluation trips.</p>
        </div>
      </div>

      <div className="selection-area">
        <div className="select-wrapper">
          <Map size={19} />

          <select value={selectedTrip} onChange={onTripChange}>
            <option value="">Choose an evaluation trip</option>

            {demoTrips.map((trip) => (
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
          ? `Selected: ${selectedTripData.name}. Demo ID only; API integration is pending.`
          : "Select a trip to prepare a reconstruction request."}
      </div>

      {requestStatus === "pending" && (
        <div className="pending-message">
          <Clock3 size={17} />
          Trip selected successfully. Backend integration is pending;
          no prediction has been requested.
        </div>
      )}
    </section>
  );
}
function RouteVisualization() {
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
          <span className="status-dot muted"></span>
          Awaiting prediction
        </div>
      </div>

      <RouteMap />
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
