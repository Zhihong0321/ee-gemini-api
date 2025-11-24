# Railway Deployment Guide for Gemini API Server

## Overview

Deploy the Gemini API server to Railway as a production web service that your other applications can consume via REST API.

## 🚀 Quick Deployment Steps

### 1. Prepare Your Code
```bash
# Make sure you have all files:
# - production_server.py
# - Dockerfile  
# - railway.toml
# - requirements.txt
# - .env (don't commit this)
```

### 2. Install Railway CLI (Optional)
```bash
# Using npm
npm install -g @railway/cli

# Or download from railway.app
```

### 3. Create Railway Account & Project
1. Go to [railway.app](https://railway.app)
2. Sign up with GitHub
3. Create new project: "gemini-api-server"

### 4. Deploy via GitHub
```bash
# 1. Push your code to GitHub
git add .
git commit -m "Deploy Gemini API Server"
git push origin main

# 2. Connect Railway to your GitHub repo
# 3. Railway will auto-deploy
```

## 🔧 Configuration Required

### Environment Variables in Railway

Go to your Railway project → Settings → Variables and add:

```bash
# REQUIRED: Your Google cookies
SECURE_1PSID=g.a0003ghj...your_full_cookie_here
SECURE_1PSIDTS=sidts-CjEBw...your_full_cookie_here

# Optional: Railway automatically sets PORT
PORT=8000
HOST=0.0.0.0
RAILWAY_ENVIRONMENT=production
```

### Getting Your Cookies

1. Go to [https://gemini.google.com](https://gemini.google.com) and login
2. Open DevTools (F12) → Network → Refresh
3. Click any request → Find cookies:
   - `__Secure-1PSID` (long cookie starting with g.a...)
   - `__Secure-1PSIDTS` (shorter cookie starting with sidts-...)
4. Copy the full values to Railway variables

## 🌐 Accessing Your Deployed API

Once deployed, Railway will give you a URL like:
```
https://your-app-name.up.railway.app
```

### API Documentation Available

📚 **Swagger UI**: `https://your-app-name.up.railway.app/docs`  
📖 **ReDoc**: `https://your-app-name.up.railway.app/redoc`

These provide interactive documentation where you can:
- View all endpoints with parameters
- Test API calls directly in browser
- See example requests/responses
- Download OpenAPI spec

### Test Your Deployment

```bash
# Health check
curl https://your-app-name.up.railway.app/health

# Send message
curl -X POST https://your-app-name.up.railway.app/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Hello! Tell me a joke.",
    "model": "gemini-2.5-flash"
  }'

# Get available models
curl https://your-app-name.up.railway.app/models
```

### Interactive Testing

1. Open `https://your-app-name.up.railway.app/docs`
2. Click on any endpoint (like `/chat`)
3. Click "Try it out"
4. Fill in the parameters
5. Click "Execute"
6. See the response live!

## 🔍 Important Considerations

### ⚠️ Potential Issues & Solutions

1. **IP Blocking**: Google may block Railway IPs
   - **Solution**: Use a proxy service or rotate IPs
   - **Monitor**: Health checks will detect failures

2. **Cookie Expiration**: Cookies expire periodically
   - **Solution**: API has auto-refresh built-in
   - **Monitor**: Set up alerts for failures

3. **Rate Limiting**: Google limits API calls
   - **Solution**: Implement rate limiting in your apps
   - **Monitor**: Track usage and errors

4. **Account Restrictions**: Some features need Pro subscription
   - **Solution**: Test with gemini-2.5-flash first

### 🔒 Security Best Practices

1. **Never commit cookies** to Git
2. **Use Railway secrets** for sensitive data
3. **Monitor access**: Add authentication if needed
4. **Set CORS policies** for your domains only

## 📊 Monitoring Your Deployment

### Railway Dashboard
- Automatic health checks on `/health`
- Logs and metrics in Railway dashboard
- Automatic restarts on failures

### Custom Monitoring
```bash
# Status endpoint with detailed info
curl https://your-app-name.up.railway.app/status

# Response includes:
# - API test results
# - Active sessions  
# - Error status
```

## 🔄 Integrating with Your Web Apps

### Frontend integration (JavaScript/React/etc.)
```javascript
const API_URL = 'https://your-app-name.up.railway.app';

async function sendMessage(message) {
  const response = await fetch(`${API_URL}/chat`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      message: message,
      model: 'gemini-2.5-flash'
    })
  });
  
  const data = await response.json();
  return data.response;
}

// Usage
sendMessage('Hello! Tell me about Python')
  .then(reply => console.log(reply))
  .catch(err => console.error(err));
```

### Backend integration (Node.js/etc.)
```javascript
const API_URL = 'https://your-app-name.up.railway.app';

app.post('/api/ai-chat', async (req, res) => {
  try {
    const response = await fetch(`${API_URL}/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        message: req.body.message,
        model: req.body.model || 'gemini-2.5-flash'
      })
    });
    
    const data = await response.json();
    res.json(data);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});
```

## 🛠️ Production Optimizations

### Built-in Features
- ✅ Auto cookie refresh
- ✅ Error handling and retries  
- ✅ Health checks
- ✅ CORS support
- ✅ Security headers
- ✅ Rate limiting awareness

### Recommended Additions
1. **API Key Authentication**: Add your own auth layer
2. **Request Rate Limiting**: Prevent abuse
3. **Usage Analytics**: Track API calls
4. **Cache Responses**: For common queries
5. **Backup Deployment**: Multiple Railway regions

## 📈 Scaling Considerations

### When to Scale
- More than 100 requests per minute
- Multiple applications using the API
- Geographic distribution needs

### Scaling Options
1. **Vertical**: Upgrade Railway plan
2. **Horizontal**: Multiple Railway instances
3. **Geographic**: Deploy in multiple regions
4. **Caching**: Add Redis layer

## 🆘 Troubleshooting

### Common Issues

**5xx Errors:**
```bash
# Check Railway logs in dashboard
# Verify environment variables
# Test cookies locally first
```

**Authentication Failures:**
```bash
# Refresh cookies from Gemini
# Verify cookies not expired
# Test with SECURE_1PSID only first
```

**Rate Limiting:**
```bash
# Wait 5-10 minutes
# Reduce request frequency
# Use different message patterns
```

### Support Resources
- Railway dashboard: [railway.app](https://railway.app)
- Gemini API repo: [GitHub](https://github.com/HanaokaYuzu/Gemini-API)  
- This deployment: Check Railway logs

## ✅ Pre-Deployment Checklist

- [ ] Cookies tested locally and work
- [ ] All code pushed to GitHub  
- [ ] Railway environment variables set
- [ ] Health check endpoint responding
- [ ] Test basic API call works
- [ ] CORS configured for your domains
- [ ] Error monitoring enabled
- [ ] Backup plan for cookie refresh

---

**Ready to deploy?** Go to railway.app → New Project → Connect your repo → Set environment variables → Deploy! 🚀
