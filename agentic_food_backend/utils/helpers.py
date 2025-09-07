# agentic_food_backend/utils/helpers.py

from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import re
import hashlib
import secrets
import string

class ValidationUtils:
    """Utility functions for data validation"""
    
    @staticmethod
    def is_valid_email(email: str) -> bool:
        """Validate email format"""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(pattern, email) is not None
    
    @staticmethod
    def is_valid_phone(phone: str) -> bool:
        """Validate phone number format"""
        # Simple validation - adjust regex based on requirements
        pattern = r'^\+?1?-?\.?\s?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}$'
        return re.match(pattern, phone) is not None
    
    @staticmethod
    def validate_password_strength(password: str) -> Dict[str, bool]:
        """Validate password strength"""
        checks = {
            "min_length": len(password) >= 8,
            "has_uppercase": re.search(r'[A-Z]', password) is not None,
            "has_lowercase": re.search(r'[a-z]', password) is not None,
            "has_digit": re.search(r'\d', password) is not None,
            "has_special": re.search(r'[!@#$%^&*(),.?":{}|<>]', password) is not None
        }
        checks["is_strong"] = all(checks.values())
        return checks

class StringUtils:
    """Utility functions for string manipulation"""
    
    @staticmethod
    def generate_session_id(prefix: str = "session") -> str:
        """Generate a unique session ID"""
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        random_string = ''.join(secrets.choice(string.ascii_letters + string.digits) for _ in range(8))
        return f"{prefix}_{timestamp}_{random_string}"
    
    @staticmethod
    def generate_order_id(prefix: str = "ORD") -> str:
        """Generate a unique order ID"""
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        random_string = ''.join(secrets.choice(string.digits) for _ in range(4))
        return f"{prefix}{timestamp}{random_string}"
    
    @staticmethod
    def clean_text_for_search(text: str) -> str:
        """Clean text for search operations"""
        if not text:
            return ""
        # Remove extra whitespace and convert to lowercase
        cleaned = re.sub(r'\s+', ' ', text.strip().lower())
        # Remove special characters except spaces and basic punctuation
        cleaned = re.sub(r'[^\w\s\-.,!?]', '', cleaned)
        return cleaned
    
    @staticmethod
    def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
        """Truncate text to specified length"""
        if not text or len(text) <= max_length:
            return text
        return text[:max_length - len(suffix)] + suffix

class DataUtils:
    """Utility functions for data processing"""
    
    @staticmethod
    def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance between two coordinates using Haversine formula"""
        from math import radians, sin, cos, sqrt, atan2
        
        R = 6371  # Earth's radius in kilometers
        
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        distance = R * c
        
        return distance
    
    @staticmethod
    def paginate_results(items: List, page: int = 1, per_page: int = 10) -> Dict:
        """Paginate a list of items"""
        total = len(items)
        start = (page - 1) * per_page
        end = start + per_page
        
        return {
            "items": items[start:end],
            "total": total,
            "page": page,
            "per_page": per_page,
            "pages": (total + per_page - 1) // per_page,
            "has_prev": page > 1,
            "has_next": end < total
        }
    
    @staticmethod
    def merge_preferences(existing: Dict, new: Dict) -> Dict:
        """Merge user preferences, with new values overriding existing ones"""
        if not existing:
            return new or {}
        if not new:
            return existing
        
        merged = existing.copy()
        merged.update(new)
        return merged
    
    @staticmethod
    def calculate_average_rating(ratings: List[int]) -> float:
        """Calculate average rating from a list of ratings"""
        if not ratings:
            return 0.0
        return sum(ratings) / len(ratings)

class TimeUtils:
    """Utility functions for time-related operations"""
    
    @staticmethod
    def is_restaurant_open(operating_hours: Dict, current_time: Optional[datetime] = None) -> bool:
        """Check if restaurant is currently open based on operating hours"""
        if not operating_hours:
            return True  # Assume open if no hours specified
        
        if current_time is None:
            current_time = datetime.now()
        
        day_name = current_time.strftime('%A').lower()
        current_hour = current_time.hour
        current_minute = current_time.minute
        current_time_minutes = current_hour * 60 + current_minute
        
        day_hours = operating_hours.get(day_name)
        if not day_hours:
            return False
        
        # Handle different formats: "09:00-22:00" or {"open": "09:00", "close": "22:00"}
        if isinstance(day_hours, str):
            if day_hours.lower() == 'closed':
                return False
            open_time, close_time = day_hours.split('-')
        elif isinstance(day_hours, dict):
            open_time = day_hours.get('open')
            close_time = day_hours.get('close')
        else:
            return False
        
        try:
            open_hour, open_min = map(int, open_time.split(':'))
            close_hour, close_min = map(int, close_time.split(':'))
            
            open_minutes = open_hour * 60 + open_min
            close_minutes = close_hour * 60 + close_min
            
            # Handle overnight hours (e.g., 22:00-02:00)
            if close_minutes < open_minutes:
                return current_time_minutes >= open_minutes or current_time_minutes <= close_minutes
            else:
                return open_minutes <= current_time_minutes <= close_minutes
        except (ValueError, AttributeError):
            return True  # Default to open if parsing fails
    
    @staticmethod
    def estimate_delivery_time(distance_km: float, base_time_minutes: int = 30) -> timedelta:
        """Estimate delivery time based on distance"""
        # Simple estimation: base time + 2 minutes per km
        additional_time = min(distance_km * 2, 60)  # Cap at 60 additional minutes
        total_minutes = base_time_minutes + additional_time
        return timedelta(minutes=total_minutes)
    
    @staticmethod
    def format_time_ago(created_at: datetime) -> str:
        """Format datetime as 'time ago' string"""
        now = datetime.utcnow()
        diff = now - created_at
        
        if diff.days > 0:
            return f"{diff.days} day{'s' if diff.days != 1 else ''} ago"
        elif diff.seconds > 3600:
            hours = diff.seconds // 3600
            return f"{hours} hour{'s' if hours != 1 else ''} ago"
        elif diff.seconds > 60:
            minutes = diff.seconds // 60
            return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
        else:
            return "Just now"

class HashUtils:
    """Utility functions for hashing operations"""
    
    @staticmethod
    def generate_file_hash(content: bytes) -> str:
        """Generate SHA-256 hash of file content"""
        return hashlib.sha256(content).hexdigest()
    
    @staticmethod
    def generate_api_key(prefix: str = "ak") -> str:
        """Generate a secure API key"""
        random_bytes = secrets.token_bytes(32)
        key = hashlib.sha256(random_bytes).hexdigest()
        return f"{prefix}_{key[:32]}"
    
    @staticmethod
    def hash_user_data_for_privacy(data: str) -> str:
        """Hash sensitive user data for privacy"""
        return hashlib.sha256(data.encode()).hexdigest()

# Create instances for easy importing
validation_utils = ValidationUtils()
string_utils = StringUtils()
data_utils = DataUtils()
time_utils = TimeUtils()
hash_utils = HashUtils()