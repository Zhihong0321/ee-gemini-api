# Railway Cookie Management Guide

## 🔄 Updating Session Tokens

### **Automatic Refresh (Recommended)**
Your Railway deployment has proactive cookie refresh enabled:
```python
# In cookie_refresh_scheduler.py
# Dedicated background task runs every 9 minutes
# Proactively refreshes all client sessions
```

**Benefits:**
- ⚡ Runs every 9 minutes in background (dedicated scheduler)
- ⚡ Works even without API requests
- ⚡ Railway service stays available overnight
- ⚡ Multiple accounts supported

### **When You Need Manual Updates**

**Signs cookies need refresh:**
- 🚨 API returns 401/403 errors
- 🚨 Health check shows `"client_ready": false`
- 🚨 Railway logs show authentication failures
- 🚨 You can't login to gemini.google.com anymore

**Quick Fix Process:**
1. **Test locally first:**
   ```bash
   cd E:\geminiapi
   python cookie_monitor.py
   ```

2. **Get fresh cookies:**
   - Go to https://gemini.google.com
   - Login again (refresh session)
   - DevTools → Network → Refresh
   - Copy new `__Secure-1PSID` and `__Secure-1PSIDTS`

3. **Update Railway:**
   - Railway → Your Project → Settings → Variables
   - Update `SECURE_1PSID` and `SECURE_1PSIDTS`
   - Railway auto-redeploys (takes ~1-2 minutes)

4. **Verify deployment:**
   ```bash
   curl https://your-app.up.railway.app/health
   ```

## 📊 Cookie Monitoring

### **Built-in Health Check**
```bash
# Check your Railway deployment health
curl https://your-app.up.railway.app/health

# Expected healthy response:
{"status":"healthy","client_ready":true,"active_sessions":0}

# Check detailed status with scheduler info:
curl https://your-app.up.railway.app/status

# Look for:
# - "client_initialized": true
# - "api_test": "passed"
# - "cookie_scheduler": {"running": true, "refresh_interval_minutes": 9}
```

### **Monitoring Cookie Refresh**
The scheduler logs all refresh attempts. Check Railway logs for:
```
Started cookie refresh scheduler (every 9 minutes)
Successfully refreshed 1 account(s) at 2024-01-01T12:00:00.000Z
Refreshed cookies for account: primary
```

## 🕰️ Cookie Lifecycle

### **Typical Expiration Patterns:**
- **1PSIDTS**: 2-7 days (expires frequently)
- **1PSIDTSCC**: 2-7 days (backup cookie)
- **1PSID**: 6 months (primary cookie, stable)

### **Automatic Refresh Handling:**
```
Background Scheduler (every 9 min) → Lightweight API call → Triggers auto_refresh → All clients refreshed → Service continues
```

## 🛡️ Backup Strategies

### **Option 1: Multiple Accounts**
- Set up 2+ Google accounts with Gemini
- Keep spare cookies ready
- Quick switch if primary fails

### **Option 2: Cookie Rotation Script**
```python
# Create backup cookies script
backup_cookies = {
    "primary": {"1psid": "...", "1psidts": "..."},
    "backup": {"1psid": "...", "1psidts": "..."}
}

def rotate_cookies():
    if not test_primary_cookies():
        switch_to_backup_cookies()
        notify_admin()
```

### **Option 3: Health Monitoring Webhook**
```python
# Add to production_server.py
@app.get("/health")
async def health_check():
    if not test_gemini_connection():
        # Send alert to monitoring service
        await send_webhook("Cookie refresh needed!")
    return status_response
```

## 🚨 Troubleshooting

### **Problem:** 401 Unauthorized Errors
**Solution:** 
1. Check Railway variables are correct
2. Get fresh cookies from gemini.google.com
3. Update Railway variables
4. Wait for redeploy + test again

### **Problem:** 403 Forbidden (IP Blocked)
**Solution:**
1. Wait 30-60 minutes (unblocks automatically)
2. Use VPN with different IP
3. Consider proxy service for reliability

### **Problem:** 429 Rate Limit Exceeded
**Solution:**
1. Wait 5-10 minutes
2. Reduce request frequency in your apps
3. Implement retry logic with exponential backoff

### **Problem:** Service Stays Down After Update
**Solution:**
1. Check Railway deployment logs
2. Verify cookie format (no extra spaces/quotes)
3. Restart Railway service manually
4. Test cookies locally first

### **Problem:** Cookies Still Expire Overnight
**Solution:**
1. Check `/status` endpoint for scheduler status
2. Look for "cookie_scheduler": {"running": true} in response
3. Check Railway logs for refresh attempts
4. If scheduler not running, redeploy to restart service

## 📞 Monitoring & Alerts

### **Health Check Frequency**
```python
# Add to your monitoring setup
import requests

def check_gemini_api():
    try:
        response = requests.get("https://your-app.up.railway.app/health", timeout=10)
        data = response.json()
        
        if data.get("client_ready") != True:
            send_alert("Gemini API authentication failed!")
            return False
            
        return True
        
    except Exception as e:
        send_alert(f"Gemini API down: {e}")
        return False

# Run every 5 minutes
schedule.every(5).minutes.do(check_gemini_api)
```

### **Slack/Discord Webhook**
```python
import webhook

def send_alert(message):
    webhook.send(
        username="Gemini Monitor",
        text=f"🚨 Gemini API Alert: {message}",
        channel="#alerts"
    )
```

## ✅ Best Practices

1. **Test locally before updating Railway**
2. **Keep backup cookies ready**
3. **Monitor for 401/403 errors**
4. **Set up health check alerts**
5. **Use proactive refresh scheduler (runs every 9 minutes)**
6. **Check /status endpoint for scheduler health**
7. **Document your refresh process**

## 🔄 Quick Reference Commands

```bash
# Test cookies locally
python cookie_monitor.py

# Check Railway health
curl https://your-app.up.railway.app/health

# Check detailed status
curl https://your-app.up.railway.app/status

# Test API manually
curl -X POST https://your-app.up.railway.app/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"test","model":"gemini-2.5-flash"}'
```

---

**Pro Tip:** The automatic refresh handles 95% of cases. Only intervene when you see persistent 401/403 errors! 🚀
