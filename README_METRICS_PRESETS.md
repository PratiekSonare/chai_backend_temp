# Metrics Presets - Pre-Calculated Daily Metrics Pipeline

## Overview

This system pre-calculates metrics for 7-day, 30-day, and all-time presets daily and uploads them to S3. The frontend then fetches these pre-calculated metrics directly from S3 for instant access, reducing API load and improving performance.

**Schedule**: Daily at **00:10:30 UTC** (10 minutes after orders extraction completes)

**S3 Output**: `s3://chupps-data-portal/metrics-presets/YYYY-MM-DD/all.json`

---

## Architecture

### Data Flow

```
DynamoDB (history-orders-final)
    ↓
[generate_metrics_presets.py] (00:10:30 UTC)
    ↓
[Metric Calculations: 7d, 30d, all-time]
    ↓
[S3 Upload: metrics-presets/YYYY-MM-DD/all.json]
    ↓
Frontend (useFetchMetric.js)
    ↓
[Detect Preset Query + No Filters?]
    ↓ YES
[Fetch from S3 (fetchMetricsFromS3)]
    ↓
[Process & Render]
    ↓ NO or S3 Failure
[Fallback to API: /history/batch/all-metrics]
```

### Components

| Component             | Purpose                              | Location                                                  |
| --------------------- | ------------------------------------ | --------------------------------------------------------- |
| **Backend Script**    | Calculates and uploads metrics       | `backend/generate_metrics_presets.py`                     |
| **Wrapper Script**    | Shell wrapper for EC2 execution      | `backend/run_generate_metrics_daily.sh`                   |
| **Systemd Service**   | Service unit for cron-like execution | `backend/deploy/systemd/generate-metrics-presets.service` |
| **Systemd Timer**     | Scheduler (runs daily at 00:10:30)   | `backend/deploy/systemd/generate-metrics-presets.timer`   |
| **Frontend Function** | Fetch metrics from S3                | `frontend/src/lib/api.js::fetchMetricsFromS3()`           |
| **Frontend Hook**     | Integration with useFetchMetric      | `frontend/src/app/orders/hooks/useFetchMetric.js`         |

### Metrics Calculated

- **Primary KPIs**: Total orders, units sold, gross revenue, AOV, unique SKUs, cancellation rate, RTO rate, COD share, delivered rate
- **Product Metrics**: SKU diversity, top SKUs by revenue/units, avg units per order, size mix, SKU performance
- **Performance Metrics**: Fulfillment rate, order value distribution, order velocity, units velocity
- **Geographic Metrics**: Top states by revenue/orders, geographic concentration, state cancellation rates
- **Channel & Payment Metrics**: Marketplace performance, courier performance, warehouse efficiency, payment mode breakdown
- **Order Type Metrics**: B2B vs B2C analysis
- **Quality & Risk Metrics**: Overall fulfillment, issue rate, payment risk, marketplace risk score
- **Advanced Metrics**: Revenue per channel, seasonal trends, product-payment correlation

---

## Deployment

### 1. EC2 Systemd Setup

Copy systemd files to the EC2 instance:

```bash
# SSH into EC2
ssh ubuntu@<ec2-instance-ip>

# Copy systemd files
sudo cp backend/deploy/systemd/generate-metrics-presets.service /etc/systemd/system/
sudo cp backend/deploy/systemd/generate-metrics-presets.timer /etc/systemd/system/

# Reload daemon
sudo systemctl daemon-reload

# Enable and start timer
sudo systemctl enable generate-metrics-presets.timer
sudo systemctl start generate-metrics-presets.timer

# Verify
sudo systemctl status generate-metrics-presets.timer
sudo systemctl list-timers generate-metrics-presets.timer
```

### 2. Verify Timer Schedule

```bash
# Check next scheduled run
sudo systemctl list-timers generate-metrics-presets.timer

# Check past executions
sudo journalctl -u generate-metrics-presets.service | head -20
```

### 3. Make Wrapper Executable

```bash
chmod +x backend/run_generate_metrics_daily.sh
```

---

## Configuration

### Environment Variables

Add these to `.env` (backend or EC2 `/home/ubuntu/chupps/.env`):

```bash
# S3 Configuration
METRICS_PRESETS_BUCKET=chupps-data-portal          # S3 bucket name
METRICS_PRESETS_PREFIX=metrics-presets             # S3 key prefix
METRICS_S3_REGION=ap-south-1                       # AWS region

# DynamoDB Configuration
HISTORY_ORDERS_DYNAMODB_TABLE=history-orders-final   # Source table
HISTORY_CACHE_ALL_TIME_START=2025-09-01 00:00:00   # All-time preset start date

# AWS Credentials (if not using IAM role)
AWS_ACCESS_KEY_ID=<your-key>
AWS_SECRET_ACCESS_KEY=<your-secret>
AWS_REGION=ap-south-1

# Optional
METRICS_PRESETS_ENABLE=true                        # Toggle on/off
METRICS_PRESETS_LOG_LEVEL=info                     # info, debug, error
```

### Frontend Environment Variables

Add these to `frontend/.env` or `frontend/.env.local`:

```bash
# S3 Bucket URL for pre-calculated metrics (signed URL or public bucket)
NEXT_PUBLIC_METRICS_S3_BUCKET_URL=https://chupps-data-portal.s3.ap-south-1.amazonaws.com

# S3 key prefix (default: metrics-presets)
NEXT_PUBLIC_METRICS_S3_PREFIX=metrics-presets

# Existing API URLs (unchanged)
NEXT_PUBLIC_QUERY_API_URL=http://localhost:5001
NEXT_PUBLIC_METRICS_API_URL=http://localhost:5002
```

---

## Manual Execution

### Test Locally

```bash
cd backend

# Generate for yesterday
python generate_metrics_presets.py --execution-date 2026-05-06

# Generate for specific date
python generate_metrics_presets.py --execution-date 2026-05-05

# With custom S3 bucket
python generate_metrics_presets.py \
    --execution-date 2026-05-06 \
    --bucket my-bucket \
    --prefix my-prefix \
    --aws-region us-east-1 \
    --ddb-table my-table
```

### Manual EC2 Execution

```bash
cd /home/ubuntu/chupps/backend

# Run wrapper script
./run_generate_metrics_daily.sh

# Or run Python directly
python generate_metrics_presets.py --execution-date 2026-05-06
```

### Trigger Systemd Service

```bash
# Run service immediately (don't wait for 00:10:30)
sudo systemctl start generate-metrics-presets.service

# Check execution
sudo journalctl -u generate-metrics-presets.service -f
```

---

## Monitoring & Logs

### View Logs

```bash
# Systemd journal (last 50 lines)
sudo journalctl -u generate-metrics-presets.service | tail -50

# Follow live logs
sudo journalctl -u generate-metrics-presets.service -f

# Log file location
tail -100 /var/log/generate-metrics.log
```

### Check S3 Upload

```bash
# List metrics by date
aws s3 ls s3://chupps-data-portal/metrics-presets/ --recursive --region ap-south-1

# Download and inspect latest
aws s3 cp s3://chupps-data-portal/metrics-presets/2026-05-07/all.json - --region ap-south-1 | jq .

# Validate JSON
aws s3 cp s3://chupps-data-portal/metrics-presets/2026-05-07/all.json - --region ap-south-1 | jq '.["7d"].data.primaryKpis' | head -20
```

### Frontend Debugging

Open browser DevTools → Network tab:

1. Apply "Last 7 Days" preset (no filters)
2. Look for S3 request in Network tab:
   - **S3 Hit**: URL will be `https://chupps-data-portal.s3.ap-south-1.amazonaws.com/metrics-presets/...`
   - **API Fallback**: Will show `/history/batch/all-metrics` request
3. Check console logs:
   - `✅ Metrics loaded from S3` → S3 hit
   - `🔴 Metrics loaded from API (live)` → API fallback

---

## Troubleshooting

### Symptom: S3 fetch fails, but API works

**Cause**: S3 bucket URL not public or signed URL expired

**Solution**:

1. Ensure S3 bucket allows public read access (for `metrics-presets/` prefix)
2. Or: Set up backend proxy endpoint to generate signed URLs
3. Check `NEXT_PUBLIC_METRICS_S3_BUCKET_URL` is correct

### Symptom: Metrics generation fails every day

**Check logs**:

```bash
sudo journalctl -u generate-metrics-presets.service -n 30
```

**Common issues**:

- DynamoDB table not reachable → Check security groups, table name, region
- S3 bucket doesn't exist → Create bucket or update `METRICS_PRESETS_BUCKET`
- AWS credentials missing → Set `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`
- Out of memory → Increase EC2 instance size

**Fallback behavior**: If calculation fails, previous day's metrics are uploaded with `_fallback_date: true` flag.

### Symptom: Timer not triggering

**Check timer status**:

```bash
sudo systemctl status generate-metrics-presets.timer
sudo systemctl list-timers
```

**Restart timer**:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now generate-metrics-presets.timer
```

### Symptom: No metrics in S3 for yesterday

**Possible causes**:

1. Script failed silently → Check logs
2. Wrong execution date → Script uses yesterday, not today
3. S3 upload permissions → Check IAM policy

**Verify**:

```bash
# List all metrics files
aws s3 ls s3://chupps-data-portal/metrics-presets/ --recursive

# Check last run
aws s3 cp s3://chupps-data-portal/metrics-presets/2026-05-06/all.json - | jq '._execution_timestamp'
```

### Symptom: Frontend always hits API, never S3

**Check**:

1. `NEXT_PUBLIC_METRICS_S3_BUCKET_URL` is set correctly
2. S3 metrics file exists for yesterday
3. Console logs show which preset detected
4. DevTools Network tab shows failed S3 request details

**Force S3 test**:

```javascript
// In browser console
import { fetchMetricsFromS3 } from "@/lib/api";
const data = await fetchMetricsFromS3();
console.log(data);
```

---

## Performance Impact

### Before (All queries hit API)

- API load: 100% of metric requests
- Latency: ~500-2000ms per request
- Server CPU: High during metric calculations
- Database load: Continuous

### After (Preset queries use S3)

- API load: ~70% of metric requests (filtered queries, custom date ranges)
- Latency: ~100-300ms for S3 (cached + CDN)
- Server CPU: Spike at 00:10:30 only
- Database load: Once daily at 00:10:30

### S3 Storage Cost

- **Per file**: ~50-200 KB (3 presets + metadata)
- **Daily**: ~200 KB
- **Monthly**: ~6 MB
- **Cost**: < $0.01/month (at standard S3 rates)

---

## S3 Output Format

```json
{
  "_execution_timestamp": "2026-05-07T00:10:30Z",
  "_execution_date": "2026-05-07",
  "_fallback_date": false,
  "_fallback_from_date": null,
  "7d": {
    "success": true,
    "data": {
      "primaryKpis": { ... },
      "productMetrics": { ... },
      "performanceMetrics": { ... },
      "geographicMetrics": { ... },
      "channelPaymentMetrics": { ... },
      "orderTypeMetrics": { ... },
      "qualityRiskMetrics": { ... },
      "advancedMetrics": { ... }
    },
    "timestamp": "2026-05-07T00:10:30Z"
  },
  "30d": { ... },
  "all": { ... }
}
```

---

## Monitoring & Alerts

### CloudWatch Integration (Optional)

```bash
# Create custom metric for metrics generation time
aws cloudwatch put-metric-data \
  --namespace ChuppMetrics \
  --metric-name MetricsGenerationDuration \
  --value 120 \
  --unit Seconds
```

### Email Alerts

To get email alerts on failure:

1. Add SNS topic to systemd service:

   ```ini
   OnFailure=send-email [email protected]
   ```

2. Or use Lambda + CloudWatch Events to trigger on systemd failure logs

---

## Upgrade & Rollback

### Disable Metrics Presets (revert to live API only)

```bash
# Stop timer
sudo systemctl stop generate-metrics-presets.timer
sudo systemctl disable generate-metrics-presets.timer

# Frontend will auto-fallback to API on S3 404
```

### Update Script

```bash
cd /home/ubuntu/chupps
git pull origin main
# Systemd service will use new script on next 00:10:30 trigger
```

---

## References

- [Systemd Timers Documentation](https://www.freedesktop.org/software/systemd/man/systemd.timer.html)
- [AWS S3 Documentation](https://docs.aws.amazon.com/s3/)
- [DynamoDB Scan API](https://docs.aws.amazon.com/amazondynamodb/latest/developerguide/Scan.html)
- [Metrics Calculation Implementation](../routes/historyOrders.py#L4219)

---

## Support

For issues or questions:

1. Check logs: `sudo journalctl -u generate-metrics-presets.service`
2. Test manually: `python backend/generate_metrics_presets.py --execution-date 2026-05-06`
3. Verify S3: `aws s3 ls s3://chupps-data-portal/metrics-presets/`
4. Check frontend console: Browser DevTools → Console tab
