# Urdu Voicebot Testing Plan

## 1. Environment Setup

### 1.1 Docker Environment
- Verify Docker containers are running: `docker-compose ps`
- Check MongoDB connection: `docker-compose exec mongo mongosh --eval "db.adminCommand('ping')"`
- Verify dashboard is accessible at http://localhost:8000

### 1.2 Environment Variables
- Verify all required environment variables are set in `.env.local`
- Confirm PTCL SIP credentials are properly configured
- Verify RunPod API key is set for GPU pod management

## 2. Core Functionality Testing

### 2.1 LLM Client Testing
- **Token Streaming**:
  - Send a test query and verify tokens are streamed incrementally
  - Check latency metrics for first token time (TTFT)
  - Verify proper error handling for LLM failures

- **Prompt Construction**:
  - Test with and without RAG content
  - Verify conversation history is properly bounded
  - Check system prompt is correctly included

### 2.2 RAG Implementation Testing
- **Filtered Lookup**:
  - Query with terms that should match RAG content
  - Query with terms that shouldn't match any RAG content
  - Verify confidence scores are reasonable
  - Check that returned content is under 500 tokens

- **Prompt Injection**:
  - Verify RAG content is properly included in the prompt
  - Test with multiple relevant RAG sections
  - Check that irrelevant RAG content is filtered out

### 2.3 Telephony Integration Testing
- **Inbound Call Flow**:
  - Simulate an inbound call from an allowed PTCL number
  - Verify call is properly routed to a LiveKit room
  - Check that the calling agent is auto-dispatched

- **Outbound Call Flow**:
  - Initiate an outbound call via the API
  - Verify call is properly connected through PTCL
  - Check that the LiveKit room is created and configured

### 2.4 GPU Pod Scheduling Testing
- **Pod Switching**:
  - Verify pods switch at the correct times
  - Check that the handoff buffer period is respected
  - Test manual pod switching via API

- **Health Monitoring**:
  - Simulate a pod failure and verify automatic restart
  - Check that the scheduler properly handles API failures

## 3. Performance Testing

### 3.1 Latency Metrics
- Run `benchmark_turn_latency.py` with the new components
- Verify p95 latency ≤ 650ms
- Check that all latency stages are properly logged

### 3.2 Concurrent Call Handling
- Simulate multiple concurrent calls
- Verify each call maintains proper latency metrics
- Check resource utilization doesn't exceed limits

### 3.3 Load Testing
- Gradually increase call volume to test system limits
- Monitor CPU, RAM, and GPU utilization
- Verify system stability under load

## 4. Integration Testing

### 4.1 Dashboard Integration
- Verify call summaries are properly received
- Check lead data is correctly stored in MongoDB
- Verify dashboard UI displays call information correctly

### 4.2 Call Recording
- Initiate a test call and verify audio is recorded
- Check recordings are properly stored in Cloudflare R2
- Verify recordings can be accessed via dashboard links

## 5. Regression Testing

- Verify existing features not directly modified still work
- Test edge cases and error conditions
- Check logging and monitoring are properly configured

## 6. User Acceptance Testing

- Conduct a small-scale user test with actual callers
- Gather feedback on call quality and agent responses
- Verify the system meets the business requirements

## Test Execution

1. Set up the testing environment according to the plan
2. Execute tests in the specified order
3. Document any issues or failures
4. Verify fixes and retest affected areas

## Test Reporting

- Create a test report summarizing results
- Document any issues found and their resolution status
- Provide recommendations for further improvements

This testing plan covers the critical components of the modified codebase. The tests should be executed in the specified order to ensure proper system functionality and performance.