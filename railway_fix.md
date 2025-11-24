# Railway Deployment Fix - Health Check Failed

## 🚨 Issue: Health Check Failed

The Railway build succeeded but health check is failing, which means the service can't start properly.

## 🔍 Most Likely Causes:

### 1. Missing Environment Variables (Most Common)
Your Google cookies are not set in Railway.

**Solution:**
1. Go to your Railway project → Settings → Variables
2. Add these **exact** variables:
   ```
   SECURE_1PSID=g.a0003ghjUxBx1PBzZiPo5g2zzwwFUyxGN3gtsu8hXZzQTMTRMyuSb6jFyAyZq_lrkO_pLBus-gACgYKAW8SARQSFQHGX2Mi4M56oRV31xknPEO1q1NrZxoVAUF8yKrc1oWJHd-GreuXGn54GUg30076
   SECURE_1PSIDTS=sidts-CjEBwQ9iI5gH7XWWJzf4b434kSUbkQl2UJ7ZmEPDr93TJNpUwqEAKVI680C3sK2BWR8AEAA
   ```

### 2. Startup Issues
The server might be failing during initialization.

**Debug steps:**
1. Check Railway logs (View Logs tab)
2. Look for authentication errors
3. Check if cookies are being read correctly

## 🛠️ Quick Fix Process:

### Step 1: Add Environment Variables
1. Go to Railway Dashboard → Your Project → Settings → Variables
2. Click "New Variable" → Add `SECURE_1PSID` with your full cookie
3. Click "New Variable" → Add `SECURE_1PSIDTS` with your short cookie
4. Railway will auto-redeploy

### Step 2: Verify Deployment
```bash
# After variables are set and redeployed:
curl https://your-app-name.up.railway.app/health

# Should return:
{"status":"healthy","client_ready":true,"active_sessions":0}
```

### Step 3: Test the API
```bash
curl -X POST https://your-app-name.up.railway.app/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"Hello! Tell me a joke.","model":"gemini-2.5-flash"}'
```

## 📋 Environment Variables Checklist:

✅ **SECURE_1PSID**: The long cookie starting with `g.a0003ghj...`  
✅ **SECURE_1PSIDTS**: The short cookie starting with `sidts-CjEB...`  
✅ **PORT**: 8000 (auto-set by Railway)  
✅ **HOST**: 0.0.0.0 (auto-set)  

## 🔍 If Still Failing:

### Check Railway Logs:
1. Go to Railway Dashboard → Your Project
2. Click the "Logs" tab
3. Look for error messages like:
   - "SECURE_1PSID environment variable is required"
   - "Gemini client initialization failed"
   - "Authentication failed"

### Common Log Messages:
```
# This means cookies missing:
ERROR: SECURE_1PSID environment variable is required!

# This means cookies invalid:
Failed to initialize Gemini client: Authentication failed

# This means working:
INFO: Gemini client initialized successfully
```

### Temporarily Disable Health Check:
If you want to check logs first, temporarily disable health check:

1. Go to `railway.toml`
2. Comment out the healthcheck line:
   ```toml
   # healthcheckPath = "/health"
   ```
3. Deploy and check logs
4. Fix variables then re-enable health check

## 🚀 Once Fixed:

Your API endpoints will be available at:
- **Swagger UI**: `https://your-app-name.up.railway.app/docs`
- **Health Check**: `https://your-app-name.up.railway.app/health`
- **Chat API**: `https://your-app-name.up.railway.app/chat`

## 📞 Quick Support:

If still having issues:
1. Copy/paste the error logs from Railway
2. Confirm the exact cookies you entered
3. Check if you can access https://gemini.google.com

The most common issue is just missing the environment variables! 🎯
