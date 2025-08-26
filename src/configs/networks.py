import ipaddress

def is_valid_ip(ip_string: str) -> bool:
    """
    Check if a string is a valid IP address (IPv4 or IPv6).
    
    Args:
        ip_string (str): The IP address string to validate
        
    Returns:
        bool: True if valid IP address, False otherwise
    """
    try:
        # Use Python's built-in ipaddress module for validation
        ipaddress.ip_address(ip_string)
        return True
    except ValueError:
        return False
    
def is_valid_port(port:int|str) -> bool:
    """
    Check if a port number is valid.
    
    Args:
        port (int, str): The port number to validate
        
    Returns:
        bool: True if valid port number, False otherwise
    """
    try:
        # Convert to integer if string
        port_num = int(port)
        # Valid port range is 1-65535
        return 1 <= port_num <= 65535
    except (ValueError, TypeError):
        return False

def is_well_known_port(port:int|str) -> bool:
    """
    Check if a port is in the well-known ports range (1-1023).
    
    Args:
        port (int, str): The port number to check
        
    Returns:
        bool: True if port is in well-known range, False otherwise
    """
    try:
        port_num = int(port)
        return 1 <= port_num <= 1023
    except (ValueError, TypeError):
        return False

def is_registered_port(port:int|str) -> bool:
    """
    Check if a port is in the registered ports range (1024-49151).
    
    Args:
        port (int, str): The port number to check
        
    Returns:
        bool: True if port is in registered range, False otherwise
    """
    try:
        port_num = int(port)
        return 1024 <= port_num <= 49151
    except (ValueError, TypeError):
        return False

def is_dynamic_port(port:int|str) -> bool:
    """
    Check if a port is in the dynamic/private ports range (49152-65535).
    
    Args:
        port (int, str): The port number to check
        
    Returns:
        bool: True if port is in dynamic range, False otherwise
    """
    try:
        port_num = int(port)
        return 49152 <= port_num <= 65535
    except (ValueError, TypeError):
        return False

def get_port_category(port:int|str) -> str:
    """
    Get the category of a port number.
    
    Args:
        port (int, str): The port number to categorize
        
    Returns:
        str: Port category or "Invalid" if not a valid port
    """
    if not is_valid_port(port):
        return "Invalid"
    
    port_num = int(port)
    
    if is_well_known_port(port_num):
        return "Well-known (1-1023)"
    elif is_registered_port(port_num):
        return "Registered (1024-49151)"
    elif is_dynamic_port(port_num):
        return "Dynamic/Private (49152-65535)"
    else:
        return "Invalid"

def validate_port_range(start_port: int|str, end_port: int|str) -> bool:
    """
    Validate a range of ports.
    
    Args:
        start_port (int, str): Starting port number
        end_port (int, str): Ending port number
        
    Returns:
        bool: True if both ports are valid and start <= end, False otherwise
    """
    try:
        start = int(start_port)
        end = int(end_port)
        return (is_valid_port(start) and 
                is_valid_port(end) and 
                start <= end)
    except (ValueError, TypeError):
        return False
    

def check_networking_args(urls: dict):

    assert is_valid_ip(urls["stream_ip"]), f"Invalid 'stream_ip' {urls["stream_ip"]}"
    assert is_registered_port(urls["stream_port"]), f"Invalid 'stream_port' {urls["stream_port"]}, expected value in [1024-49151]"
    assert is_valid_ip(urls["telemetry_ip"]), f"Invalid 'telemetry_ip' {urls["telemetry_ip"]}"
    assert is_registered_port(urls["telemetry_port"]), f"Invalid 'telemetry_port' {urls["telemetry_port"]}, expected value in [1024-49151]"
    assert is_valid_ip(urls["annotations_ip"]), f"Invalid 'annotations_ip' {urls["annotations_ip"]}"
    assert is_registered_port(urls["annotations_port"]), f"Invalid 'annotations_port' {urls["annotations_port"]}, expected value in [1024-49151]"
    assert is_valid_ip(urls["alerts_ip"]), f"Invalid 'alerts_ip' {urls["alerts_ip"]}"
    assert is_registered_port(urls["alerts_port"]), f"Invalid 'alerts_port' {urls["alerts_port"]}, expected value in [1024-49151]"