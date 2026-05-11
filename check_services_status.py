#!/usr/bin/env python3
"""
Service Status Checker for EC2 Deployed Services

Usage:
    python3 check_services_status.py [--host IP] [--user USERNAME] [--key-file PATH]
    
Examples:
    python3 check_services_status.py --host 52.1.2.3 --user ec2-user --key-file ~/my-key.pem
    python3 check_services_status.py --host localhost
    python3 check_services_status.py  # Uses local systemd
"""

import subprocess
import json
import sys
import argparse
from datetime import datetime
from typing import Dict, List, Tuple
from dataclasses import dataclass
import os

# ANSI color codes
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'

@dataclass
class ServiceStatus:
    name: str
    active: bool
    running: bool
    last_exit_code: int
    last_start_time: str
    success: bool
    error_message: str = ""

SERVICES = [
    "extract-orders-previous-day.service",
    "generate-metrics-presets.service",
    "prediction-training.service",
]

def execute_command(cmd: str, host: str = None, user: str = None, key_file: str = None) -> Tuple[str, int]:
    """Execute command locally or on remote EC2 instance"""
    try:
        if host and host not in ["localhost", "127.0.0.1"]:
            ssh_cmd = ["ssh", "-o", "StrictHostKeyChecking=no"]
            if key_file:
                ssh_cmd.extend(["-i", key_file])
            ssh_cmd.append(f"{user}@{host}")
            ssh_cmd.append(cmd)
        else:
            ssh_cmd = ["bash", "-c", cmd]
        
        result = subprocess.run(ssh_cmd, capture_output=True, text=True, timeout=30)
        return result.stdout, result.returncode
    except subprocess.TimeoutExpired:
        return "", -1
    except Exception as e:
        return str(e), -1

def get_service_status(service: str, host: str = None, user: str = None, key_file: str = None) -> ServiceStatus:
    """Get detailed status of a systemd service"""
    
    # Check if service is active
    cmd_status = f"systemctl is-active {service} 2>&1"
    status_output, _ = execute_command(cmd_status, host, user, key_file)
    is_active = status_output.strip() in ["active", "activating"]
    
    # Check if service exists
    if "Unit" in status_output and "could not be found" in status_output:
        return ServiceStatus(
            name=service,
            active=False,
            running=False,
            last_exit_code=-1,
            last_start_time="N/A",
            success=False,
            error_message="Service not found"
        )
    
    # Get exit code
    cmd_exit = f"systemctl show -p ExecMainStatus --value {service} 2>&1"
    exit_code_output, _ = execute_command(cmd_exit, host, user, key_file)
    try:
        exit_code = int(exit_code_output.strip() or "-1")
    except ValueError:
        exit_code = -1
    
    # Get last start time
    cmd_time = f"systemctl show -p ExecMainStartTimestamp --value {service} 2>&1"
    time_output, _ = execute_command(cmd_time, host, user, key_file)
    last_start_time = time_output.strip() or "N/A"
    
    # Get recent logs
    cmd_journal = f"journalctl -u {service} -n 20 --no-pager 2>&1"
    journal_output, _ = execute_command(cmd_journal, host, user, key_file)
    
    # Determine success
    success = exit_code == 0 and is_active
    error_message = ""
    
    # Check for error patterns in logs
    if journal_output:
        for line in journal_output.split('\n'):
            if any(keyword in line.lower() for keyword in ['error', 'failed', 'exception', 'traceback']):
                error_message = line[:100]
                success = False
                break
    
    return ServiceStatus(
        name=service,
        active=is_active,
        running="running" in status_output.lower(),
        last_exit_code=exit_code,
        last_start_time=last_start_time,
        success=success,
        error_message=error_message
    )

def print_service_status(status: ServiceStatus) -> None:
    """Pretty print service status"""
    print(f"\n{Colors.BLUE}{'━' * 50}{Colors.END}")
    print(f"{Colors.BOLD}Service: {status.name}{Colors.END}")
    print(f"{Colors.BLUE}{'━' * 50}{Colors.END}")
    
    # Status indicator
    if status.success:
        status_indicator = f"{Colors.GREEN}✓ Success{Colors.END}"
    else:
        status_indicator = f"{Colors.RED}✗ Failed{Colors.END}"
    
    print(f"Overall Status: {status_indicator}")
    
    # Active status
    active_status = f"{Colors.GREEN}Active{Colors.END}" if status.active else f"{Colors.RED}Inactive{Colors.END}"
    print(f"Is Active: {active_status}")
    
    # Exit code
    if status.last_exit_code == 0:
        exit_status = f"{Colors.GREEN}0 (Success){Colors.END}"
    elif status.last_exit_code == -1:
        exit_status = f"{Colors.YELLOW}N/A{Colors.END}"
    else:
        exit_status = f"{Colors.RED}{status.last_exit_code} (Failed){Colors.END}"
    print(f"Last Exit Code: {exit_status}")
    
    # Last run time
    print(f"Last Start Time: {Colors.BOLD}{status.last_start_time}{Colors.END}")
    
    # Error message
    if status.error_message:
        print(f"Latest Error: {Colors.RED}{status.error_message}{Colors.END}")

def print_summary(statuses: List[ServiceStatus]) -> int:
    """Print summary and return exit code"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'=' * 50}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}SUMMARY{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'=' * 50}{Colors.END}\n")
    
    successful = [s for s in statuses if s.success]
    failed = [s for s in statuses if not s.success]
    
    print(f"Total Services: {len(statuses)}")
    print(f"✓ Successful: {Colors.GREEN}{len(successful)}/{len(statuses)}{Colors.END}")
    print(f"✗ Failed:    {Colors.RED}{len(failed)}/{len(statuses)}{Colors.END}")
    
    if failed:
        print(f"\n{Colors.RED}Failed Services:{Colors.END}")
        for status in failed:
            print(f"  - {status.name}")
            if status.error_message:
                print(f"    Error: {status.error_message}")
    else:
        print(f"\n{Colors.GREEN}{Colors.BOLD}All services running successfully!{Colors.END}")
    
    return 1 if failed else 0

def main():
    parser = argparse.ArgumentParser(
        description="Check status of systemd services on EC2 instance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 check_services_status.py
  python3 check_services_status.py --host 52.1.2.3 --user ec2-user --key-file ~/.ssh/my-key.pem
  python3 check_services_status.py --host localhost
        """
    )
    parser.add_argument("--host", default="localhost", help="EC2 instance IP or hostname (default: localhost)")
    parser.add_argument("--user", default=os.getenv("USER"), help="SSH username (default: $USER)")
    parser.add_argument("--key-file", default=None, help="Path to SSH private key file")
    parser.add_argument("--json", action="store_true", help="Output results as JSON")
    
    args = parser.parse_args()
    
    print(f"{Colors.BOLD}{Colors.BLUE}=== Service Status Check ==={Colors.END}")
    location = f"EC2 instance ({args.host})" if args.host not in ["localhost", "127.0.0.1"] else "local machine"
    print(f"Checking on: {Colors.BOLD}{location}{Colors.END}\n")
    
    # Check all services
    statuses = []
    for service in SERVICES:
        print(f"Checking {service}...", end=" ", flush=True)
        status = get_service_status(service, args.host, args.user, args.key_file)
        statuses.append(status)
        
        indicator = f"{Colors.GREEN}✓{Colors.END}" if status.success else f"{Colors.RED}✗{Colors.END}"
        print(indicator)
    
    # Print detailed status
    if not args.json:
        for status in statuses:
            print_service_status(status)
    
    # Print summary
    if args.json:
        # Output as JSON
        data = {
            "timestamp": datetime.now().isoformat(),
            "location": args.host,
            "services": [
                {
                    "name": s.name,
                    "success": s.success,
                    "active": s.active,
                    "last_exit_code": s.last_exit_code,
                    "last_start_time": s.last_start_time,
                    "error": s.error_message
                }
                for s in statuses
            ]
        }
        print(json.dumps(data, indent=2))
    else:
        exit_code = print_summary(statuses)
        sys.exit(exit_code)

if __name__ == "__main__":
    main()
