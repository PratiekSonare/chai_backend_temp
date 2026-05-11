#!/bin/bash

##############################################################################
# Service Status Checker for EC2 Deployed Services
# 
# Usage:
#   ./check_services_status.sh [EC2_INSTANCE_IP] [SSH_USER] [SSH_KEY]
#
# Examples:
#   ./check_services_status.sh ec2-user
#   ./check_services_status.sh 52.1.2.3 ec2-user ~/my-key.pem
#   ./check_services_status.sh (uses local systemd if on same machine)
##############################################################################

set -o pipefail

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Services to check
SERVICES=(
    "extract-orders-previous-day.service"
    "generate-metrics-presets.service"
    "prediction-training.service"
)

# Parse arguments
EC2_IP="${1:-localhost}"
SSH_USER="${2:-$USER}"
SSH_KEY="${3:-}"

# Build SSH command
if [ "$EC2_IP" == "localhost" ] || [ "$EC2_IP" == "127.0.0.1" ]; then
    SSH_CMD=""
    LOCATION="local machine"
else
    if [ -z "$SSH_KEY" ]; then
        SSH_CMD="ssh -o StrictHostKeyChecking=no $SSH_USER@$EC2_IP"
    else
        SSH_CMD="ssh -o StrictHostKeyChecking=no -i $SSH_KEY $SSH_USER@$EC2_IP"
    fi
    LOCATION="EC2 instance ($EC2_IP)"
fi

echo -e "${BOLD}${BLUE}=== Service Status Check ===${NC}"
echo -e "Checking on: ${BOLD}$LOCATION${NC}\n"

# Function to execute commands (local or remote)
execute_cmd() {
    local cmd="$1"
    if [ -z "$SSH_CMD" ]; then
        eval "$cmd"
    else
        $SSH_CMD "$cmd"
    fi
}

# Function to get service status
check_service_status() {
    local service="$1"
    echo -e "${BOLD}${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BOLD}Service: $service${NC}"
    echo -e "${BOLD}${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    
    # Get service status
    status_output=$(execute_cmd "systemctl status $service 2>&1" || true)
    
    # Check if service exists
    if echo "$status_output" | grep -q "Unit.*could not be found"; then
        echo -e "${RED}❌ Service not found${NC}"
        return 1
    fi
    
    # Extract status info
    if echo "$status_output" | grep -q "active (exited)"; then
        echo -e "${GREEN}✓ Status: Active (Last run completed)${NC}"
    elif echo "$status_output" | grep -q "active (running)"; then
        echo -e "${YELLOW}⟳ Status: Active (Currently running)${NC}"
    elif echo "$status_output" | grep -q "inactive"; then
        echo -e "${RED}✗ Status: Inactive${NC}"
    else
        echo -e "${YELLOW}⚠ Status: Unknown${NC}"
    fi
    
    # Get last run time
    last_run=$(execute_cmd "systemctl show -p ExecMainStartTimestamp --value $service 2>&1" || true)
    if [ -n "$last_run" ] && [ "$last_run" != "ExecMainStartTimestamp=" ]; then
        echo -e "Last run: ${BOLD}$last_run${NC}"
    fi
    
    # Get last exit code
    exit_code=$(execute_cmd "systemctl show -p ExecMainStatus --value $service 2>&1" || true)
    if [ -n "$exit_code" ] && [ "$exit_code" != "0" ]; then
        echo -e "Last exit code: ${RED}$exit_code${NC}"
    elif [ "$exit_code" = "0" ]; then
        echo -e "Last exit code: ${GREEN}$exit_code (Success)${NC}"
    fi
    
    # Get detailed journal log (last 10 lines)
    echo -e "\n${BOLD}Latest logs:${NC}"
    journal_output=$(execute_cmd "journalctl -u $service -n 10 --no-pager 2>&1" || true)
    
    if [ -n "$journal_output" ]; then
        # Check for error patterns in logs
        if echo "$journal_output" | grep -qi "error\|failed\|exception"; then
            echo -e "${RED}⚠ Errors detected in logs:${NC}"
            echo "$journal_output" | grep -i "error\|failed\|exception" | tail -5 | sed 's/^/  /'
        else
            echo "$journal_output" | tail -3 | sed 's/^/  /'
        fi
    fi
    
    echo ""
}

# Check all services
FAILED_SERVICES=()
SUCCESS_SERVICES=()

for service in "${SERVICES[@]}"; do
    if check_service_status "$service"; then
        SUCCESS_SERVICES+=("$service")
    else
        FAILED_SERVICES+=("$service")
    fi
done

# Summary
echo -e "${BOLD}${BLUE}=== SUMMARY ===${NC}"
echo -e "✓ Successful: ${GREEN}${#SUCCESS_SERVICES[@]}/${#SERVICES[@]}${NC}"
echo -e "✗ Failed:    ${RED}${#FAILED_SERVICES[@]}/${#SERVICES[@]}${NC}"

if [ ${#FAILED_SERVICES[@]} -gt 0 ]; then
    echo -e "\n${RED}Failed services:${NC}"
    for service in "${FAILED_SERVICES[@]}"; do
        echo -e "  - $service"
    done
    exit 1
fi

echo -e "\n${GREEN}All services running successfully!${NC}"
exit 0
