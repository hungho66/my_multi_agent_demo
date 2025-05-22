import httpx
from langchain_core.tools import tool
from typing import Optional, Dict, Any
from ..utils.progress import progress_tracker

OPEN_METEO_API_URL = "https://api.open-meteo.com/v1/forecast"
GEOCODING_API_URL = "https://geocoding-api.open-meteo.com/v1/search"

def get_coordinates(city_name: str) -> Optional[tuple[float, float]]:
    """Fetches coordinates (latitude, longitude) for a given city name."""
    tool_name = "WeatherToolInternal (Geocoding)"
    progress_tracker.update_status(tool_name, optional_data={"city": city_name}, status_message="Đang lấy tọa độ...")
    try:
        params = {"name": city_name, "count": 1, "language": "en", "format": "json"}
        with httpx.Client(timeout=10.0) as client:
            response = client.get(GEOCODING_API_URL, params=params)
            response.raise_for_status()
            data = response.json()
            if data.get("results"):
                location = data["results"][0]
                lat, lon = location["latitude"], location["longitude"]
                progress_tracker.update_status(tool_name, optional_data={"city": city_name}, status_message=f"Tọa độ cho {city_name}: ({lat}, {lon}).")
                return lat, lon
        progress_tracker.update_status(tool_name, optional_data={"city": city_name}, status_message=f"LỖI: Không tìm thấy kết quả tọa độ cho {city_name}.")
        return None
    except Exception as e:
        error_msg = f"Lỗi khi lấy tọa độ cho {city_name}: {e}"
        progress_tracker.update_status(tool_name, optional_data={"city": city_name}, status_message=f"LỖI: {error_msg}")
        print(error_msg)
        return None

@tool
def get_current_weather(city: str) -> str:
    """
    Fetches the current weather for a specified city using the Open-Meteo API.
    Input should be the city name (e.g., "Hanoi", "London").
    """
    tool_name = "WeatherTool"
    progress_tracker.update_status(tool_name, optional_data={"city": city}, status_message=f"Bắt đầu lấy thời tiết cho {city}...")

    coordinates = get_coordinates(city)
    if not coordinates:
        error_msg = f"Không thể tìm thấy tọa độ cho thành phố: {city}. Vui lòng cung cấp tên thành phố hợp lệ."
        progress_tracker.update_status(tool_name, optional_data={"city": city}, status_message=f"LỖI: {error_msg}")
        return error_msg

    latitude, longitude = coordinates
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "current_weather": "true",
        "temperature_unit": "celsius",
        "wind_speed_unit": "kmh",
        "precipitation_unit": "mm",
        "timezone": "auto"
    }
    try:
        with httpx.Client(timeout=10.0) as client:
            progress_tracker.update_status(tool_name, optional_data={"city": city}, status_message=f"Đang gọi API thời tiết cho {city}...")
            response = client.get(OPEN_METEO_API_URL, params=params)
            response.raise_for_status()
            data = response.json()

        if "current_weather" in data:
            current = data["current_weather"]
            weather_code = current.get('weather_code')
            weather_condition_map = {
                0: "Trời quang", 1: "Trời quang, ít mây", 2: "Trời nhiều mây", 3: "Trời u ám",
                45: "Sương mù", 48: "Sương mù",
                51: "Mưa phùn nhẹ", 53: "Mưa phùn vừa", 55: "Mưa phùn dày",
                56: "Mưa phùn đông lạnh nhẹ", 57: "Mưa phùn đông lạnh dày",
                61: "Mưa nhẹ", 63: "Mưa vừa", 65: "Mưa to",
                66: "Mưa đông lạnh nhẹ", 67: "Mưa đông lạnh to",
                71: "Tuyết rơi nhẹ", 73: "Tuyết rơi vừa", 75: "Tuyết rơi dày",
                77: "Hạt tuyết",
                80: "Mưa rào nhẹ", 81: "Mưa rào vừa", 82: "Mưa rào dữ dội",
                85: "Mưa tuyết nhẹ", 86: "Mưa tuyết nặng",
                95: "Dông (nhẹ hoặc vừa)", 96: "Dông kèm mưa đá nhẹ", 99: "Dông kèm mưa đá nặng"
            }
            weather_condition = weather_condition_map.get(weather_code, f"Không xác định (Mã: {weather_code})")

            result_str = (
                f"Thời tiết hiện tại ở {city.title()}: "
                f"Nhiệt độ: {current.get('temperature')}°C, "
                f"Tốc độ gió: {current.get('windspeed')} km/h. "
                f"Tình trạng: {weather_condition}. "
                f"Thời gian ghi nhận (giờ địa phương): {current.get('time')}."
            )
            progress_tracker.update_status(tool_name, optional_data={"city": city}, status_message=f"Lấy thời tiết cho {city} thành công.")
            return result_str
        else:
            error_msg = f"Không thể lấy dữ liệu thời tiết hiện tại cho {city.title()} từ API response."
            progress_tracker.update_status(tool_name, optional_data={"city": city}, status_message=f"LỖI: {error_msg}")
            return error_msg
    except httpx.HTTPStatusError as e:
        error_msg = f"Lỗi API cho {city.title()}: {e.response.status_code} - {e.response.text}"
        progress_tracker.update_status(tool_name, optional_data={"city": city}, status_message=f"LỖI API: {e.response.status_code}")
        return error_msg
    except Exception as e:
        error_msg = f"Đã xảy ra lỗi khi lấy thông tin thời tiết cho {city.title()}: {str(e)}"
        progress_tracker.update_status(tool_name, optional_data={"city": city}, status_message=f"LỖI: {str(e)}")
        return error_msg 