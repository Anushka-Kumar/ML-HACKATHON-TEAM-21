const configuredBaseUrl = import.meta.env.VITE_API_BASE_URL?.trim();

export const API_BASE_URL = configuredBaseUrl
  ? configuredBaseUrl.replace(/\/$/, "")
  : "/api";

async function request(path, options) {
  const response = await fetch(`${API_BASE_URL}${path}`, options);
  const contentType = response.headers.get("content-type") || "";
  const data = contentType.includes("application/json")
    ? await response.json()
    : null;

  if (!response.ok) {
    const detail =
      typeof data?.detail === "string"
        ? data.detail
        : `Request failed with status ${response.status}`;
    throw new Error(detail);
  }

  return data;
}

export function getHealth() {
  return request("/health");
}

export function getTrips() {
  return request("/trips");
}

export function predictTrip(tripId) {
  return request(`/trips/${encodeURIComponent(tripId)}/predict`, {
    method: "POST",
  });
}
