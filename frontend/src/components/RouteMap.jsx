
import { MapContainer, TileLayer, ZoomControl } from "react-leaflet";
import { Layers, MapPin, Route } from "lucide-react";
import "leaflet/dist/leaflet.css";

function RouteMap() {
  return (
    <div className="real-map-container">
      <div className="real-map-toolbar">
        <div className="real-map-heading">
          <MapPin size={17} />
          <span>OpenStreetMap · Geospatial view</span>
        </div>

        <div className="real-map-layer-label">
          <Layers size={15} />
          Map layers
        </div>
      </div>

      <div className="leaflet-map-wrapper">
        <MapContainer
          center={[12.9716, 77.5946]}
          zoom={13}
          zoomControl={false}
          scrollWheelZoom={true}
          className="leaflet-map"
        >
          <TileLayer
            attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
            url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          />
          <ZoomControl position="bottomright" />
        </MapContainer>

        <div className="map-empty-overlay">
          <div className="map-empty-card">
            <div className="map-center-icon">
              <Route size={25} />
            </div>

            <h3>Ready for route reconstruction</h3>

            <p>
              Select an evaluation trip to display its GPS trajectory
              and predicted road-network route.
            </p>

            <span className="map-awaiting">
              Awaiting actual trip coordinates
            </span>
          </div>
        </div>
      </div>

      <div className="real-map-footer">
        <span>
          <i className="legend-blue"></i>
          GPS observations
        </span>
        <span>
          <i className="legend-green"></i>
          Predicted route
        </span>
        <span className="map-data-note">
          No route data loaded
        </span>
      </div>
    </div>
  );
}

export default RouteMap;
