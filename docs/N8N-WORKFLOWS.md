# CAAL n8n Workflow Integration

This guide explains how CAAL discovers and uses n8n workflows as voice-callable tools.

## Overview

CAAL transforms n8n workflows into LLM tools via the Model Context Protocol (MCP). Any n8n workflow with a webhook trigger and MCP enabled becomes a tool the voice assistant can call.

```
┌─────────────────────────────────────────────────────────────────┐
│                        Voice Request                            │
│                    "What's the weather?"                        │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                          CAAL Agent                             │
│                                                                 │
│  1. LLM sees tool: weather_get_forecast                         │
│     Description: "Get weather. Params: location (required)"     │
│                                                                 │
│  2. LLM decides to call tool with: {"location": "Seattle"}      │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                         n8n Workflow                            │
│                                                                 │
│  POST http://n8n:5678/webhook/weather_get_forecast              │
│  Body: {"location": "Seattle"}                                  │
│                                                                 │
│  Returns: {                                                     │
│    "message": "Seattle is 52 degrees and cloudy",               │
│    "weather": {"temp": 52, "condition": "cloudy"}               │
│  }                                                              │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        Voice Response                           │
│                "Seattle is 52 degrees and cloudy"               │
└─────────────────────────────────────────────────────────────────┘
```

---

## How It Works

### 1. Discovery

When CAAL starts (or when `/reload-tools` is called), it discovers workflows via n8n's MCP server:

1. Calls `search_workflows` to get list of MCP-enabled workflows
2. Calls `get_workflow_details` for each to extract the webhook node's notes
3. Creates tool definitions with the workflow name and description

### 2. Tool Description

The **webhook node's `notes` field** becomes the tool description the LLM sees. This is how the LLM knows what the tool does and what parameters to pass.

Example webhook node notes:
```
Get current weather for a location.
Parameters:
- location (required): City name or zip code
- units (optional): "metric" or "imperial", defaults to imperial
```

The LLM reads this and knows to call the tool with `{"location": "Seattle", "units": "metric"}`.

### 3. Execution

When the LLM decides to call a tool:

1. CAAL receives the tool call with arguments (e.g., `weather_get_forecast` with `{"location": "Seattle"}`)
2. CAAL POSTs to `{n8n_url}/webhook/{workflow_name}` with the arguments as JSON body
3. The n8n workflow executes and returns a response
4. CAAL speaks the `message` field to the user

### 4. Response Format

Workflows must return both a voice message and structured data:

```javascript
return {
  message: "Brief voice response here",  // Spoken aloud
  data: [...]  // Enables follow-up questions
};
```

**Why both?**
- `message` - Read aloud by the voice assistant
- `data` - Cached in conversation context for follow-up questions

Example:
```javascript
return {
  message: "Bills 20, Browns 10.",
  games: [
    { away: "BUF", awayScore: 20, home: "CLE", homeScore: 10, status: "live" },
    { away: "SEA", awayScore: 38, home: "LAR", homeScore: 37, status: "final" }
  ]
};
```

Now if the user asks "What about the Seahawks game?", the LLM can check the cached `games` array without re-calling the tool.

### 5. Naming Convention

Workflow names follow the pattern: `service_action_object` (snake_case)

| Workflow Name | Description |
|---------------|-------------|
| `weather_get_forecast` | Get weather forecast |
| `calendar_get_events` | Get calendar events |
| `radarr_search_movies` | Search Radarr library |
| `espn_get_nfl_scores` | Get NFL scores from ESPN |

**Important:** The workflow name becomes both:
- The webhook path: `/webhook/weather_get_forecast`
- The tool name the LLM sees: `weather_get_forecast`

---

## Creating a Workflow

### Step 1: Create Webhook Trigger

1. Add a **Webhook** node as the trigger
2. Set **HTTP Method** to `POST`
3. Set **Path** to your workflow name (e.g., `weather_get_forecast`)
4. Set **Response Mode** to "Using 'Respond to Webhook' Node"

### Step 2: Add Tool Description

In the webhook node's **Notes** field, describe:
- What the tool does
- Required parameters
- Optional parameters with defaults

```
Get current weather for a location.

Parameters:
- location (required): City name or zip code
- units (optional): "metric" or "imperial", defaults to imperial

Returns current temperature and conditions.
```

### Step 3: Build Your Logic

Add nodes to:
1. Call external APIs (HTTP Request node)
2. Process/filter data (Code node)
3. Format response for voice (Code node)

### Step 4: Format Voice Response

In your final Code node before responding:

```javascript
const data = $input.item.json;

// Format for voice - keep it brief and natural
const message = `It's ${data.temp} degrees and ${data.condition} in ${data.location}`;

return {
  message: message,
  weather: {
    temp: data.temp,
    condition: data.condition,
    location: data.location
  }
};
```

### Step 5: Respond to Webhook

Add a **Respond to Webhook** node at the end to return the response.

### Step 6: Enable MCP

1. Go to **Workflow Settings** (gear icon)
2. Enable **"Available in MCP"**
3. Save and activate the workflow

---

## Voice Output Guidelines

All responses are read aloud. Follow these rules:

| Do | Don't |
|----|-------|
| Brief and conversational | Long detailed explanations |
| Plain text | Markdown, JSON, symbols |
| Natural language | Technical formats |
| Short names ("Falcons") | Full names ("Atlanta Falcons") |
| Limit lists (3-5 items) | Long enumerated lists |

**Good:** "Falcons 29, Bucs 28. Seahawks 16, Bears 7."

**Bad:** "Game 1: Atlanta Falcons versus Tampa Bay Buccaneers, final score 29 to 28..."

---

## Common Patterns

### Pattern 1: Simple API Query

```
Webhook → HTTP Request → Code (format) → Respond
```

### Pattern 2: Multi-Step Processing

```
Webhook → HTTP Request → Code (filter) → HTTP Request 2 → Code (format) → Respond
```

### Pattern 3: Async/Long-Running

For tasks taking >5 seconds, respond immediately then announce when done:

```
Webhook → Respond ("On it, I'll let you know")
    ↓
 [Long work]
    ↓
 HTTP Request → CAAL /announce endpoint
```

The announce call:
```json
{
  "method": "POST",
  "url": "http://caal-agent:8889/announce",
  "body": {"message": "Your task is complete."}
}
```

---

## n8n Setup

### Enable MCP Server

1. Open n8n **Settings**
2. Go to **MCP Access**
3. Enable MCP
4. Set connection method to **Access Token**
5. Copy the token

### Configure CAAL

In CAAL's Settings Panel → Integrations → n8n:

1. Enable n8n
2. Enter your n8n host URL (e.g., `http://192.168.1.100:5678`)
3. Paste the MCP access token
4. Test the connection

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Tool not appearing | Check workflow is active and "Available in MCP" is enabled |
| Wrong parameters | Update webhook node notes with clear parameter docs |
| Empty response | Ensure Code node returns `{message: "...", data: ...}` |
| Webhook 404 | Verify workflow name matches webhook path exactly |
| Timeout | Use async pattern for long-running tasks |

### Reload Tools

After creating or modifying workflows, reload CAAL's tool cache:

```bash
curl -X POST http://localhost:8889/reload-tools
```

Or say "reload tools" to the voice assistant.

---

## Example Workflow

Here's a complete weather workflow:

```json
{
  "name": "weather_get_forecast",
  "nodes": [
    {
      "parameters": {
        "httpMethod": "POST",
        "path": "weather_get_forecast",
        "responseMode": "responseNode"
      },
      "type": "n8n-nodes-base.webhook",
      "typeVersion": 2,
      "position": [0, 0],
      "webhookId": "weather_get_forecast",
      "notes": "Get current weather.\n\nParameters:\n- location (required): City name"
    },
    {
      "parameters": {
        "url": "=https://wttr.in/{{ $json.location }}?format=j1"
      },
      "type": "n8n-nodes-base.httpRequest",
      "typeVersion": 4.2,
      "position": [200, 0]
    },
    {
      "parameters": {
        "jsCode": "const weather = $input.item.json.current_condition[0];\nconst location = $input.item.json.nearest_area[0].areaName[0].value;\n\nreturn {\n  message: `${location} is ${weather.temp_F} degrees and ${weather.weatherDesc[0].value.toLowerCase()}`,\n  weather: {\n    location: location,\n    temp: parseInt(weather.temp_F),\n    condition: weather.weatherDesc[0].value\n  }\n};"
      },
      "type": "n8n-nodes-base.code",
      "typeVersion": 2,
      "position": [400, 0]
    },
    {
      "parameters": {},
      "type": "n8n-nodes-base.respondToWebhook",
      "typeVersion": 1.1,
      "position": [600, 0]
    }
  ],
  "connections": {
    "Webhook": { "main": [[{ "node": "HTTP Request", "index": 0 }]] },
    "HTTP Request": { "main": [[{ "node": "Code", "index": 0 }]] },
    "Code": { "main": [[{ "node": "Respond to Webhook", "index": 0 }]] }
  },
  "settings": {
    "availableInMCP": true
  }
}
```

---

## Further Reading

- [n8n-workflows/README.md](../n8n-workflows/README.md) - Pre-built workflows and setup script
- [n8n-workflows/caal-workflow-builder-seed.md](../n8n-workflows/caal-workflow-builder-seed.md) - AI prompt for building workflows
