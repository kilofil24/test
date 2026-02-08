"""Deterministic weather signal engine for Polymarket (METAR-based)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, Iterable, List, Optional, Tuple

import logging
import math
import os

import json
import urllib.error
import urllib.request


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class Thresholds:
    temp_delta_c: float
    visibility_delta_m: int
    ceiling_delta_ft: int


@dataclass(frozen=True)
class StationMeta:
    icao: str
    lat: float
    lon: float


@dataclass
class MetarSnapshot:
    station: str
    observed: datetime
    wind_dir: Optional[int]
    wind_speed_kt: Optional[float]
    temperature_c: Optional[float]
    visibility_m: Optional[int]
    ceiling_ft: Optional[int]


@dataclass
class DeltaSnapshot:
    temp_delta_c: Optional[float]
    visibility_delta_m: Optional[int]
    ceiling_delta_ft: Optional[int]


@dataclass
class TrendInfo:
    temp: Optional[str]
    visibility: Optional[str]
    ceiling: Optional[str]


@dataclass
class SignalState:
    last_alerted: Dict[str, datetime]
    last_deltas: Dict[Tuple[str, str], DeltaSnapshot]
    last_sector: Optional[str]


DEFAULT_THRESHOLDS = Thresholds(
    temp_delta_c=2.0,
    visibility_delta_m=2000,
    ceiling_delta_ft=500,
)

DEFAULT_STATIONS: Dict[str, StationMeta] = {
    "EGLC": StationMeta("EGLC", 51.5053, 0.0553),
    "EGKK": StationMeta("EGKK", 51.1481, -0.1903),
    "EGLL": StationMeta("EGLL", 51.4700, -0.4543),
    "EGHI": StationMeta("EGHI", 50.9503, -1.3568),
    "EGMC": StationMeta("EGMC", 51.5725, 0.6956),
    "EGSS": StationMeta("EGSS", 51.8850, 0.2350),
    "EGGW": StationMeta("EGGW", 51.8747, -0.3683),
    "EGKA": StationMeta("EGKA", 50.8356, -0.2972),
    "EGKB": StationMeta("EGKB", 51.3308, 0.0325),
}

DEFAULT_UPWIND_STATIONS: Dict[str, List[str]] = {
    "N": ["EGSS", "EGMC", "EGGW"],
    "NE": ["EGMC", "EGSS", "EGGW"],
    "E": ["EGMC", "EGKB", "EGGW"],
    "SE": ["EGKB", "EGKA", "EGHI"],
    "S": ["EGKA", "EGHI", "EGKK"],
    "SW": ["EGKK", "EGLL", "EGHI"],
    "W": ["EGLL", "EGGW", "EGKK"],
    "NW": ["EGGW", "EGSS", "EGLL"],
}


class SignalEngine:
    def __init__(
        self,
        checkwx_api_key: str,
        telegram_token: Optional[str] = None,
        telegram_chat_id: Optional[str] = None,
        thresholds: Thresholds = DEFAULT_THRESHOLDS,
        upwind_stations: Dict[str, List[str]] = DEFAULT_UPWIND_STATIONS,
        stations: Dict[str, StationMeta] = DEFAULT_STATIONS,
        hysteresis_deg: int = 20,
        max_age_minutes: int = 45,
        cooldown_minutes: int = 60,
        request_timeout_s: int = 10,
    ) -> None:
        self.checkwx_api_key = checkwx_api_key
        self.telegram_token = telegram_token
        self.telegram_chat_id = telegram_chat_id
        self.thresholds = thresholds
        self.upwind_stations = upwind_stations
        self.stations = stations
        self.hysteresis_deg = hysteresis_deg
        self.max_age = timedelta(minutes=max_age_minutes)
        self.cooldown = timedelta(minutes=cooldown_minutes)
        self.request_timeout_s = request_timeout_s
        self.state = SignalState(last_alerted={}, last_deltas={}, last_sector=None)

    def run_once(self) -> Optional[str]:
        eglc = self.fetch_decoded_metar("EGLC")
        if eglc is None:
            LOGGER.warning("EGLC METAR unavailable")
            return None
        if self.is_stale(eglc):
            LOGGER.warning("EGLC METAR stale")
            return None

        sector = determine_sector(
            eglc.wind_dir, self.state.last_sector, self.hysteresis_deg
        )
        if sector is None:
            LOGGER.warning("Unable to determine sector from EGLC wind")
            return None
        self.state.last_sector = sector

        upwind = self.select_upwind_station(sector)
        if upwind is None:
            LOGGER.warning("No upwind station available for %s", sector)
            return None

        deltas = compare_metars(eglc, upwind)
        reasons = reasons_for_alert(deltas, self.thresholds)
        if not reasons:
            return None

        signal_id = signal_key(sector, upwind.station, reasons)
        if not self.should_alert(signal_id):
            return None

        trend = self.compute_trend(sector, upwind.station, deltas)
        lead_time = estimate_lead_time_minutes(
            self.stations, eglc.station, upwind.station, eglc.wind_speed_kt
        )

        message = format_alert(
            sector=sector,
            upwind=upwind,
            eglc=eglc,
            reasons=reasons,
            deltas=deltas,
            lead_time_minutes=lead_time,
            trend=trend,
        )

        self.send_telegram(message)
        self.state.last_alerted[signal_id] = datetime.now(timezone.utc)
        self.state.last_deltas[(sector, upwind.station)] = deltas
        return message

    def select_upwind_station(self, sector: str) -> Optional[MetarSnapshot]:
        for station in self.upwind_stations.get(sector, []):
            metar = self.fetch_decoded_metar(station)
            if metar is None:
                continue
            if self.is_stale(metar):
                continue
            return metar
        return None

    def fetch_decoded_metar(self, station: str) -> Optional[MetarSnapshot]:
        url = f"https://api.checkwx.com/metar/{station}/decoded"
        headers = {"X-API-Key": self.checkwx_api_key}
        try:
            payload = get_json(url, headers=headers, timeout_s=self.request_timeout_s)
        except (urllib.error.URLError, ValueError) as exc:
            LOGGER.warning("CheckWX request failed for %s: %s", station, exc)
            return None

        data = payload.get("data")
        if not data:
            return None
        return parse_metar_payload(data[0], station)

    def is_stale(self, metar: MetarSnapshot) -> bool:
        return datetime.now(timezone.utc) - metar.observed > self.max_age

    def should_alert(self, signal_id: str) -> bool:
        last = self.state.last_alerted.get(signal_id)
        if last is None:
            return True
        return datetime.now(timezone.utc) - last >= self.cooldown

    def compute_trend(
        self, sector: str, station: str, deltas: DeltaSnapshot
    ) -> TrendInfo:
        previous = self.state.last_deltas.get((sector, station))
        if previous is None:
            return TrendInfo(temp=None, visibility=None, ceiling=None)
        return TrendInfo(
            temp=trend_arrow(deltas.temp_delta_c, previous.temp_delta_c),
            visibility=trend_arrow(
                deltas.visibility_delta_m, previous.visibility_delta_m
            ),
            ceiling=trend_arrow(deltas.ceiling_delta_ft, previous.ceiling_delta_ft),
        )

    def send_telegram(self, message: str) -> None:
        if not self.telegram_token or not self.telegram_chat_id:
            LOGGER.info("Telegram disabled; message:\n%s", message)
            return
        url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
        payload = {"chat_id": self.telegram_chat_id, "text": message}
        try:
            post_json(url, payload, timeout_s=self.request_timeout_s)
        except (urllib.error.URLError, ValueError) as exc:
            LOGGER.warning("Telegram send failed: %s", exc)


def parse_metar_payload(raw: dict, station: str) -> Optional[MetarSnapshot]:
    try:
        observed = raw.get("observed")
        if not observed:
            return None
        observed_dt = datetime.fromisoformat(observed.replace("Z", "+00:00"))
        wind = raw.get("wind") or {}
        temperature = raw.get("temperature") or {}
        visibility = raw.get("visibility") or {}
        clouds = raw.get("clouds") or []

        wind_dir = wind.get("degrees")
        wind_speed = wind.get("speed_kts")
        temp_c = temperature.get("celsius")
        vis_m = visibility.get("meters")
        ceiling = lowest_ceiling(clouds)

        return MetarSnapshot(
            station=station,
            observed=observed_dt.astimezone(timezone.utc),
            wind_dir=wind_dir,
            wind_speed_kt=wind_speed,
            temperature_c=temp_c,
            visibility_m=vis_m,
            ceiling_ft=ceiling,
        )
    except (TypeError, ValueError) as exc:
        LOGGER.warning("Failed to parse METAR payload for %s: %s", station, exc)
        return None


def lowest_ceiling(clouds: Iterable[dict]) -> Optional[int]:
    layers = [layer for layer in clouds if layer.get("code") in {"BKN", "OVC"}]
    ceilings = [layer.get("feet") for layer in layers if layer.get("feet") is not None]
    return min(ceilings) if ceilings else None


def get_json(url: str, headers: Dict[str, str], timeout_s: int) -> dict:
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        payload = response.read().decode("utf-8")
        return json.loads(payload)


def post_json(url: str, payload: dict, timeout_s: int) -> None:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        response.read()


def determine_sector(
    wind_dir: Optional[int],
    previous: Optional[str],
    hysteresis: int,
) -> Optional[str]:
    if wind_dir is None:
        return None
    wind_dir = wind_dir % 360

    if previous:
        center = sector_center(previous)
        if center is not None:
            delta = angular_delta(wind_dir, center)
            if delta <= 22.5 + hysteresis:
                return previous

    return sector_from_degrees(wind_dir)


def sector_from_degrees(degrees: int) -> str:
    boundaries = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    index = int(((degrees + 22.5) % 360) // 45)
    return boundaries[index]


def sector_center(sector: str) -> Optional[int]:
    centers = {
        "N": 0,
        "NE": 45,
        "E": 90,
        "SE": 135,
        "S": 180,
        "SW": 225,
        "W": 270,
        "NW": 315,
    }
    return centers.get(sector)


def angular_delta(a: int, b: int) -> float:
    diff = abs(a - b) % 360
    return min(diff, 360 - diff)


def compare_metars(eglc: MetarSnapshot, upwind: MetarSnapshot) -> DeltaSnapshot:
    def maybe_delta(a: Optional[float], b: Optional[float]) -> Optional[float]:
        if a is None or b is None:
            return None
        return b - a

    def maybe_delta_int(a: Optional[int], b: Optional[int]) -> Optional[int]:
        if a is None or b is None:
            return None
        return b - a

    return DeltaSnapshot(
        temp_delta_c=maybe_delta(eglc.temperature_c, upwind.temperature_c),
        visibility_delta_m=maybe_delta_int(eglc.visibility_m, upwind.visibility_m),
        ceiling_delta_ft=maybe_delta_int(eglc.ceiling_ft, upwind.ceiling_ft),
    )


def reasons_for_alert(deltas: DeltaSnapshot, thresholds: Thresholds) -> List[str]:
    reasons: List[str] = []
    if deltas.temp_delta_c is not None and abs(deltas.temp_delta_c) >= thresholds.temp_delta_c:
        reasons.append("temperature")
    if (
        deltas.visibility_delta_m is not None
        and abs(deltas.visibility_delta_m) >= thresholds.visibility_delta_m
    ):
        reasons.append("visibility")
    if (
        deltas.ceiling_delta_ft is not None
        and abs(deltas.ceiling_delta_ft) >= thresholds.ceiling_delta_ft
    ):
        reasons.append("ceiling")
    return reasons


def trend_arrow(current: Optional[float], previous: Optional[float]) -> Optional[str]:
    if current is None or previous is None:
        return None
    if current > previous:
        return "⬆️"
    if current < previous:
        return "⬇️"
    return "➡️"


def estimate_lead_time_minutes(
    stations: Dict[str, StationMeta],
    eglc_station: str,
    upwind_station: str,
    wind_speed_kt: Optional[float],
) -> Optional[int]:
    if wind_speed_kt is None or wind_speed_kt <= 0:
        return None
    eglc_meta = stations.get(eglc_station)
    upwind_meta = stations.get(upwind_station)
    if not eglc_meta or not upwind_meta:
        return None
    distance_km = haversine_km(
        eglc_meta.lat, eglc_meta.lon, upwind_meta.lat, upwind_meta.lon
    )
    speed_kmh = wind_speed_kt * 1.852
    hours = distance_km / speed_kmh
    return max(1, int(round(hours * 60)))


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = (
        math.sin(d_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    )
    return 2 * radius * math.asin(math.sqrt(a))


def format_alert(
    sector: str,
    upwind: MetarSnapshot,
    eglc: MetarSnapshot,
    reasons: List[str],
    deltas: DeltaSnapshot,
    lead_time_minutes: Optional[int],
    trend: TrendInfo,
) -> str:
    lead = f"{lead_time_minutes} min" if lead_time_minutes else "n/a"
    temp_delta = format_delta(deltas.temp_delta_c, "°C")
    vis_delta = format_delta(deltas.visibility_delta_m, "m")
    ceiling_delta = format_delta(deltas.ceiling_delta_ft, "ft")
    trend_text = format_trend(trend)

    return (
        "Polymarket Weather Alert\n"
        f"Sector: {sector}\n"
        f"Upwind station: {upwind.station}\n"
        f"Reasons: {', '.join(reasons)}\n"
        "\n"
        f"EGLC Temp: {format_value(eglc.temperature_c, '°C')} | "
        f"Upwind Temp: {format_value(upwind.temperature_c, '°C')} | "
        f"Δ {temp_delta}\n"
        f"EGLC Vis: {format_value(eglc.visibility_m, 'm')} | "
        f"Upwind Vis: {format_value(upwind.visibility_m, 'm')} | "
        f"Δ {vis_delta}\n"
        f"EGLC Ceiling: {format_value(eglc.ceiling_ft, 'ft')} | "
        f"Upwind Ceiling: {format_value(upwind.ceiling_ft, 'ft')} | "
        f"Δ {ceiling_delta}\n"
        f"Lead time estimate: {lead}\n"
        f"Trend: {trend_text}"
    )


def format_value(value: Optional[float], unit: str) -> str:
    if value is None:
        return "n/a"
    return f"{value}{unit}"


def format_delta(value: Optional[float], unit: str) -> str:
    if value is None:
        return "n/a"
    return f"{value:+}{unit}"


def format_trend(trend: TrendInfo) -> str:
    parts = []
    if trend.temp:
        parts.append(f"Temp {trend.temp}")
    if trend.visibility:
        parts.append(f"Vis {trend.visibility}")
    if trend.ceiling:
        parts.append(f"Ceiling {trend.ceiling}")
    return ", ".join(parts) if parts else "n/a"


def signal_key(sector: str, station: str, reasons: List[str]) -> str:
    joined = ":".join(sorted(reasons))
    return f"{sector}:{station}:{joined}"


def load_engine_from_env() -> SignalEngine:
    return SignalEngine(
        checkwx_api_key="a30731f563514dda84a3149d9a2e770b",
        telegram_token=None,
        telegram_chat_id=None,
    )



def main() -> None:
    logging.basicConfig(level=logging.INFO)
    engine = load_engine_from_env()
    engine.run_once()


if __name__ == "__main__":
    main()
