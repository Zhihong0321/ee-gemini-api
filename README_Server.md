# Gemini API Server

A FastAPI-based REST server that provides easy access to Google's Gemini AI through the reverse-engineered Gemini-API library.

## Features

- **REST API Interface**: Clean HTTP endpoints for Gemini AI functionality
- **Multiple Models**: Support for Gemini-3.0-Pro, Gemini-2.5-Pro, Gemini-2.5-Flash
- **Chat Sessions**: Persistent conversations with session management
- **File Upload**: Support for images and documents
- **Async Processing**: High-performance async request handling
- **Error Handling**: Comprehensive error responses and rate limiting
- **CORS Support**: Ready for frontend integration

## Quick Start

### 1. Installation

```bash
# Clone or download the project
cd geminiapi

# Install dependencies
pip install -r requirements.txt
```

### 2. Authentication

Get your Gemini cookies:

1. Go to [https://gemini.google.com](https://gemini.google.com) and login
2. Open browser dev tools (F12)
3. Go to Network tab and refresh the page
4. Click any request and copy cookie values:
   - `__Secure-1PSID` (required)
   - `__Secure-1PSIDTS` (optional)

### 3. Configure Environment

```bash
# Windows
set SECURE_1PSID=your_cookie_value_here
set SECURE_1PSIDTS=your_cookie_value_here

# Linux/Mac
export SECURE_1PSID=your_cookie_value_here
export SECURE_1PSIDTS=your_cookie_value_here
```

Or create `.env` file:
```bash
cp .env.example .env
# Edit .env with your cookie values
```

### 4. Start Server

```bash
python api_server.py
```

Server will start at `http://localhost:8000`

## API Endpoints

### Health & Models
```
GET  /                    # Basic info
GET  /health              # Health check
GET  /models              # List available models
```

### Single Messages
```
POST /chat                # Send one-time message
```

### Chat Sessions
```
POST /chat/new            # Create new chat session
GET  /chat/{session_id}   # Get session info
POST /chat/{session_id}   # Send message in session
DEL  /chat/{session_id}   # Delete session
GET  /chat                # List all sessions
```

### File Upload
```
POST /upload              # Upload file (returns path)
POST /chat/{id}/files     # Message with file attachments
```

## Usage Examples

### Basic Chat
```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Hello! Tell me about Python programming.",
    "model": "gemini-2.5-flash"
  }'
```

### Chat Session
```bash
# Create session
curl -X POST "http://localhost:8000/chat/new"

# Send message
curl -X POST "http://localhost:8000/chat/session-id-here" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What is machine learning?",
    "model": "gemini-2.5-pro"
  }'
```

### File Upload
```bash
curl -X POST "http://localhost:8000/chat/session-id/files" \
  -F "message=Analyze this image" \
  -F "model=gemini-2.5-flash" \
  -F "files=@image.jpg"
```

## Response Format

```json
{
  "response": "The AI's text response",
  "model": "gemini-2.5-flash",
  "session_id": "uuid-here",
  "candidates_count": 1,
  "thoughts": "Internal reasoning (for thinking models)",
  "images": [
    {
      "url": "https://...",
      "title": "Image title",
      "alt": "Description",
      "type": "web|generated"
    }
  ],
  "metadata": {
    "rcid": "reply-candidate-id",
    "metadata": ["chat-id", "reply-id", "rcid"]
  }
}
```

## Available Models

- `gemini-2.5-flash` - Fast, efficient model
- `gemini-2.5-pro` - More capable model
- `gemini-3.0-pro` - Latest advanced model
- `unspecified` - Let Gemini choose

## Error Handling

The API returns proper HTTP status codes and detailed error messages:

- `400` - Bad request (invalid model, etc.)
- `401` - Authentication failed
- `403` - Access blocked/temporarily unavailable
- `429` - Rate limit exceeded
- `500` - Internal server error
- `503` - Service unavailable

## Development

### Running in Development Mode
```bash
python api_server.py
```

### Docker Support
You can containerize the server:

```dockerfile
FROM python:3.11
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "api_server.py"]
```

## Security Notes

- **Never commit your cookies to version control**
- **Use environment variables** for sensitive data
- **Consider Docker secrets** for production部署
- **Rotate cookies periodically** as they expire

## Limitations

- Relies on reverse-engineered web API (may break)
- Subject to Google's rate limiting
- Some features may require advanced subscription
- Cookies expire and need refreshing

## Support

For issues with:
- **Server code**: Check this repository
- **Gemini-API**: Visit https://github.com/HanaokaYuzu/Gemini-API
- **Google Gemini**: Visit https://gemini.google.com
