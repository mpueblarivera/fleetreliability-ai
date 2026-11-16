#!/usr/bin/env bash
# ==============================================================================
# FleetReliability-AI — Automated Cloud & Local E2E Validation Suite
# ==============================================================================
set -euo pipefail

RAW_TARGET_URL="${1:-https://fleetreliability-ai.onrender.com}"
TARGET_URL="${RAW_TARGET_URL%/}"

echo "========================================================================"
echo "Starting E2E Validation Suite against: ${TARGET_URL}"
echo "========================================================================"

# Helper function for timed HTTP assertions against canonical Pydantic v2 schemas
execute_test() {
    local test_name="$1"
    local endpoint="$2"
    local method="$3"
    local payload="$4"
    local expected_code="$5"
    local jq_check="${6:-}"

    echo ""
    echo ">> Running: ${test_name}"

    local response_file
    response_file=$(mktemp)

    local curl_meta
    local http_code
    local latency

    if [ "${method}" = "GET" ]; then
        curl_meta=$(curl -s --noproxy '127.0.0.1,localhost' -o "${response_file}" -w "%{http_code} %{time_total}s" -X GET "${TARGET_URL}${endpoint}")
    else
        curl_meta=$(curl -s --noproxy '127.0.0.1,localhost' -o "${response_file}" -w "%{http_code} %{time_total}s" -X POST "${TARGET_URL}${endpoint}" \
            -H "Content-Type: application/json" -d "${payload}")
    fi

    http_code="${curl_meta%% *}"
    latency="${curl_meta#* }"

    echo "   HTTP Status Code: ${http_code} (Expected: ${expected_code})"
    echo "   Total Latency:    ${latency}"
    echo "   Server Response:"
    if command -v jq &> /dev/null; then
        jq -C . "${response_file}" || cat "${response_file}"
    else
        cat "${response_file}"
    fi
    echo ""

    if [ "${http_code}" != "${expected_code}" ]; then
        echo "[FAIL] HTTP status (${http_code}) does not match expected (${expected_code})."
        rm -f "${response_file}"
        exit 1
    fi

    if [ -n "${jq_check}" ] && command -v jq &> /dev/null; then
        if ! jq -e "${jq_check}" "${response_file}" > /dev/null; then
            echo "[FAIL] Schema/contract assertion failed: ${jq_check}"
            rm -f "${response_file}"
            exit 1
        fi
    fi

    echo "[PASS] ${test_name} passed."
    rm -f "${response_file}"
}

# 1. Healthcheck (HealthResponse)
execute_test "Health Check (/health)" "/health" "GET" "" "200" \
    '.status == "HEALTHY" and .model_loaded == true and (.thresholds | has("theta_warning") and has("theta_critical"))'

# 2. Normal condition (PredictionResponse - GOLDEN-NORMAL-01)
NORMAL_PAYLOAD='{"machine_id":"PUMP-SAG-GOLDEN-01","machine_type":"M","air_temperature_k":298.2,"process_temperature_k":308.7,"rotational_speed_rpm":1500.0,"torque_nm":42.8,"tool_wear_min":45.0}'
execute_test "Predict: Normal Condition" "/predict" "POST" "${NORMAL_PAYLOAD}" "200" \
    '.operational_status == "NORMAL" and (.failure_risk_score >= 0 and .failure_risk_score <= 1) and (.derived_physics | has("mechanical_power_kw") and has("temperature_difference_k") and has("strain_index_min_nm") and has("heat_stress_ratio"))'

# 3. CBM Inspection warning (PredictionResponse - GOLDEN-INSPECTION-02)
INSPECTION_PAYLOAD='{"machine_id":"COMPRESSOR-GOLDEN-02","machine_type":"H","air_temperature_k":300.7,"process_temperature_k":310.8,"rotational_speed_rpm":1296.0,"torque_nm":58.2,"tool_wear_min":204.0}'
execute_test "Predict: Inspection Required" "/predict" "POST" "${INSPECTION_PAYLOAD}" "200" \
    '.operational_status == "INSPECTION_REQUIRED" and (.failure_risk_score >= 0 and .failure_risk_score <= 1) and (.derived_physics | has("mechanical_power_kw"))'

# 4. Critical intervention (PredictionResponse - GOLDEN-CRITICAL-03)
CRITICAL_PAYLOAD='{"machine_id":"CRUSHER-GOLDEN-03","machine_type":"H","air_temperature_k":303.0,"process_temperature_k":313.5,"rotational_speed_rpm":1250.0,"torque_nm":88.0,"tool_wear_min":245.0}'
execute_test "Predict: Critical Intervention" "/predict" "POST" "${CRITICAL_PAYLOAD}" "200" \
    '.operational_status == "CRITICAL_INTERVENTION" and (.failure_risk_score >= 0 and .failure_risk_score <= 1)'

# 5. Physical validation error (HTTP 422)
INVALID_PAYLOAD='{"machine_id":"TEST-ERR","machine_type":"L","air_temperature_k":100.0,"process_temperature_k":308.2,"rotational_speed_rpm":1550.0,"torque_nm":32.0,"tool_wear_min":15.0}'
execute_test "Physical Integrity Validation (HTTP 422)" "/predict" "POST" "${INVALID_PAYLOAD}" "422" \
    '.detail | length > 0'

echo "========================================================================"
echo "All E2E tests completed successfully with exit code 0."
echo "========================================================================"
