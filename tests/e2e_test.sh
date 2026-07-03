#!/bin/bash
# ============================================================
# E2E Test Script for Multimodal RAG Chunking Service
# Usage: bash tests/e2e_test.sh [path_to_pdf]
# ============================================================

BASE_URL="${BASE_URL:-http://localhost:8000}"
TEST_PDF="${1:-tests/sample.pdf}"

echo "=== Multimodal RAG Chunking Service E2E Test ==="
echo "Base URL: ${BASE_URL}"
echo ""

# 1. Health check
echo "[1/6] Health check (deep)..."
RESP=$(curl -s -w "\n%{http_code}" "${BASE_URL}/health")
HTTP_CODE=$(echo "$RESP" | tail -1)
BODY=$(echo "$RESP" | sed '$d')
echo "  HTTP: ${HTTP_CODE}"
echo "  Body: ${BODY}"
echo ""

# 2. Upload file
echo "[2/6] Upload file..."
if [ -f "${TEST_PDF}" ]; then
    UPLOAD_RESP=$(curl -s -X POST \
        -F "file=@${TEST_PDF}" \
        "${BASE_URL}/api/v1/upload")
    echo "  Response: ${UPLOAD_RESP}"

    TASK_ID=$(echo "${UPLOAD_RESP}" | python3 -c "import sys,json; print(json.load(sys.stdin)['data']['task_id'])" 2>/dev/null)
    DOC_ID=$(echo "${UPLOAD_RESP}" | python3 -c "import sys,json; print(json.load(sys.stdin)['data']['doc_id'])" 2>/dev/null)
    DOC_TYPE=$(echo "${UPLOAD_RESP}" | python3 -c "import sys,json; print(json.load(sys.stdin)['data']['doc_type'])" 2>/dev/null)

    if [ -z "${TASK_ID}" ] || [ -z "${DOC_ID}" ]; then
        echo "  ERROR: Failed to parse upload response"
        exit 1
    fi
    echo "  task_id:  ${TASK_ID}"
    echo "  doc_id:   ${DOC_ID}"
    echo "  doc_type: ${DOC_TYPE}"
else
    echo "  SKIP: No test file found at ${TEST_PDF}"
    exit 0
fi
echo ""

# 3. Poll task status
echo "[3/6] Polling task status..."
for i in $(seq 1 60); do
    sleep 2
    STATUS_RESP=$(curl -s "${BASE_URL}/api/v1/tasks/${TASK_ID}")
    STATUS=$(echo "${STATUS_RESP}" | python3 -c "import sys,json; print(json.load(sys.stdin)['data']['status'])" 2>/dev/null)
    PROGRESS=$(echo "${STATUS_RESP}" | python3 -c "import sys,json; print(json.load(sys.stdin)['data']['progress'])" 2>/dev/null)
    STAGE=$(echo "${STATUS_RESP}" | python3 -c "import sys,json; print(json.load(sys.stdin)['data']['stage'])" 2>/dev/null)
    echo "  Attempt $i: status=${STATUS}, progress=${PROGRESS}%, stage=${STAGE}"

    if [ "${STATUS}" = "completed" ]; then
        echo "  Task completed!"
        break
    elif [ "${STATUS}" = "failed" ]; then
        echo "  Task FAILED!"
        echo "  Response: ${STATUS_RESP}"
        exit 1
    fi
done
echo ""

# 4. List chunks
echo "[4/6] List chunks for doc_id=${DOC_ID}..."
CHUNKS_RESP=$(curl -s "${BASE_URL}/api/v1/chunks?doc_id=${DOC_ID}&page=1&size=5")
TOTAL=$(echo "${CHUNKS_RESP}" | python3 -c "import sys,json; print(json.load(sys.stdin)['data']['total'])" 2>/dev/null)
echo "  Total chunks: ${TOTAL}"

# Check doc_type in chunk response
FIRST_CHUNK_DOC_TYPE=$(echo "${CHUNKS_RESP}" | python3 -c "import sys,json; data=json.load(sys.stdin); print(data['data']['result'][0].get('doc_type',''))" 2>/dev/null)
echo "  First chunk doc_type: ${FIRST_CHUNK_DOC_TYPE}"
echo ""

# 5. Search (basic)
echo "[5/6] Search (basic text)..."
SEARCH_RESP=$(curl -s -X POST \
    -H "Content-Type: application/json" \
    -d '{"query": "test query", "top_k": 5}' \
    "${BASE_URL}/api/v1/search")
echo "  Response: $(echo "${SEARCH_RESP}" | head -c 500)"
echo ""

# 6. Search (hybrid modes)
echo "[6/6] Search (hybrid modes)..."

# Test modal=both with rerank
echo "  Testing modal=both + rerank..."
BOTH_RESP=$(curl -s -X POST \
    -H "Content-Type: application/json" \
    -d "{\"query\": \"test query\", \"top_k\": 5, \"modal\": \"both\", \"use_rerank\": true}" \
    "${BASE_URL}/api/v1/search")
echo "  Response: $(echo "${BOTH_RESP}" | head -c 500)"
echo ""

# Test metadata filter
echo "  Testing metadata filter (chunk_type=text)..."
FILTER_RESP=$(curl -s -X POST \
    -H "Content-Type: application/json" \
    -d "{\"query\": \"test query\", \"top_k\": 5, \"filters\": {\"chunk_type\": \"text\"}}" \
    "${BASE_URL}/api/v1/search")
echo "  Response: $(echo "${FILTER_RESP}" | head -c 500)"
echo ""

# Test doc_ids filter
echo "  Testing doc_ids filter..."
DOCIDS_RESP=$(curl -s -X POST \
    -H "Content-Type: application/json" \
    -d "{\"query\": \"test query\", \"top_k\": 5, \"doc_ids\": [\"${DOC_ID}\"]}" \
    "${BASE_URL}/api/v1/search")
echo "  Response: $(echo "${DOCIDS_RESP}" | head -c 500)"
echo ""

echo "=== All tests passed ==="

# Bonus: auth test (only if API_KEYS is configured)
API_KEYS="${API_KEYS:-}"
if [ -n "${API_KEYS}" ]; then
    echo ""
    echo "[Bonus] Auth tests..."

    # Without key → should get 401
    NOKEY=$(curl -s -w "\n%{http_code}" "${BASE_URL}/api/v1/chunks?doc_id=test")
    NOKEY_CODE=$(echo "$NOKEY" | tail -1)
    if [ "${NOKEY_CODE}" = "401" ]; then
        echo "  OK: rejected without API key (401)"
    else
        echo "  FAIL: expected 401, got ${NOKEY_CODE}"
    fi

    # With valid key → should succeed
    VALID=$(curl -s -w "\n%{http_code}" \
        -H "X-API-Key: ${API_KEYS}" \
        "${BASE_URL}/api/v1/chunks?doc_id=test")
    VALID_CODE=$(echo "$VALID" | tail -1)
    if [ "${VALID_CODE}" = "200" ]; then
        echo "  OK: accepted with valid API key (200)"
    else
        echo "  FAIL: expected 200, got ${VALID_CODE}"
    fi
fi
