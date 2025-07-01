# Product Questions MVP - Slack Bot

A Slack bot that helps teams find relevant past discussions by searching through channel history and linking to existing threads. The bot uses ChromaDB for vector storage and Google's Gemini API for semantic search.

## 🚀 Features

- **Channel-Specific Search**: Only searches within the channel where the bot is tagged
- **Thread Linking**: Finds and links to relevant past discussions
- **Eddie Personality**: Enthusiastic responses inspired by The Hitchhiker's Guide to the Galaxy
- **Vector Search**: Uses semantic embeddings to find relevant content
- **No AI Responses**: Focuses purely on linking to existing discussions

## 📋 Prerequisites

Before setting up the bot, ensure you have:

- **Python 3.8+** installed on your machine
- **Slack Workspace** with admin permissions to create apps
- **Google Cloud Project** with Gemini API access
- **Git** for version control

## 🛠️ Installation & Setup

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd product-questions-ai-bot
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

If `requirements.txt` doesn't exist, install these packages manually:

```bash
pip install slack-bolt python-dotenv google-generativeai chromadb slack-sdk
```

### 3. Environment Configuration

Create a `.env` file in the project root with the following variables:

```env
# Slack Configuration
SLACK_BOT_TOKEN=xoxb-your-bot-token-here
SLACK_SIGNING_SECRET=your-signing-secret-here
SLACK_APP_TOKEN=xapp-your-app-token-here

# Google Gemini API
GOOGLE_API_KEY=your-gemini-api-key-here
```

#### How to Get These Tokens:

**Slack Bot Token (`SLACK_BOT_TOKEN`):**
1. Go to [api.slack.com/apps](https://api.slack.com/apps)
2. Create a new app or select existing one
3. Navigate to "OAuth & Permissions"
4. Copy the "Bot User OAuth Token" (starts with `xoxb-`)

**Slack Signing Secret (`SLACK_SIGNING_SECRET`):**
1. In your Slack app settings, go to "Basic Information"
2. Copy the "Signing Secret"

**Slack App Token (`SLACK_APP_TOKEN`):**
1. In your Slack app settings, go to "Basic Information"
2. Scroll to "App-Level Tokens"
3. Create a new token with `connections:write` scope
4. Copy the token (starts with `xapp-`)

**Google API Key (`GOOGLE_API_KEY`):**
1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create a new API key
3. Copy the key

### 4. Slack App Configuration

Your Slack app needs these OAuth scopes:
- `app_mentions:read`
- `channels:history`
- `channels:read`
- `chat:write`
- `links:read`

And these event subscriptions:
- `app_mention`

## 🏃‍♂️ Running the Bot

### 1. Populate Channel History

First, you need to fetch and store your channel's message history:

```bash
# Update the channel ID in slack_history_loader.py
python slack_history_loader.py
```

This creates a JSON file with your channel's message history.

### 2. Set Up Vector Database

```bash
# Update the JSON filename in vector_db_populator.py
python vector_db_populator.py
```

This creates embeddings and stores them in ChromaDB.

### 3. Start the Bot

```bash
python bot.py
```

The bot will connect to Slack and start listening for mentions.

## 🔧 Configuration Files

### `slack_history_loader.py`
- **Purpose**: Fetches message history from a specific Slack channel
- **Key Settings**:
  - `TARGET_CHANNEL_ID`: The channel ID to fetch history from
  - `MESSAGES_TO_FETCH_LIMIT`: Number of messages to fetch (default: 75)

### `vector_db_populator.py`
- **Purpose**: Creates embeddings and populates ChromaDB
- **Key Settings**:
  - `INPUT_JSON_FILE`: The JSON file created by the loader
  - `COLLECTION_NAME`: ChromaDB collection name (default: "slack_history")

### `bot.py`
- **Purpose**: Main bot application
- **Key Settings**:
  - `MODEL_NAME`: Gemini model version (default: "gemini-2.0-flash-exp")
  - `COLLECTION_NAME`: Must match the populator's collection name

## 🧹 Database Management

### Clear ChromaDB Collection

If you need to start fresh:

```bash
python clear_chroma.py
```

### Re-populate with New Data

```bash
# 1. Clear the database
python clear_chroma.py

# 2. Fetch new history
python slack_history_loader.py

# 3. Populate the database
python vector_db_populator.py
```

## 🧪 Testing

### Test the Bot

1. **Invite the bot to a channel**:
   ```
   /invite @YourBotName
   ```

2. **Tag the bot with a question**:
   ```
   @YourBotName Do we have a lease buy tool available for clients?
   ```

3. **Expected Response**:
   - Eddie's enthusiastic greeting
   - Links to relevant past discussions (if found)
   - No AI-generated answers

### Debug Mode

Enable debug logging by modifying the logging level in any script:

```python
logging.basicConfig(level=logging.DEBUG, ...)
```

## 📁 Project Structure

```
product-questions-ai-bot/
├── bot.py                    # Main bot application
├── slack_history_loader.py   # Fetches channel history
├── vector_db_populator.py    # Creates embeddings and populates ChromaDB
├── clear_chroma.py          # Utility to clear ChromaDB
├── requirements.txt         # Python dependencies
├── .env                     # Environment variables (create this)
├── chroma_data/            # ChromaDB storage (created automatically)
└── slack_history_*.json    # Channel history files (created by loader)
```

## 🔄 Development Workflow

### Making Changes

1. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** and test locally

3. **Commit your changes**:
   ```bash
   git add .
   git commit -m "feat: Description of your changes"
   ```

4. **Push and create a pull request**:
   ```bash
   git push origin feature/your-feature-name
   ```

### Common Development Tasks

**Adding a New Channel:**
1. Update `TARGET_CHANNEL_ID` in `slack_history_loader.py`
2. Run the loader to fetch new history
3. Update `INPUT_JSON_FILE` in `vector_db_populator.py`
4. Re-populate the database

**Modifying Bot Personality:**
- Edit the greeting message in `bot.py` around line 231

**Changing Search Behavior:**
- Modify the `search_slack_history` function in `bot.py`
- Adjust `top_n` parameter for number of results

## 🐛 Troubleshooting

### Common Issues

**"Channel not found" Error:**
- Ensure the bot is invited to the channel
- Verify the channel ID is correct
- Check bot permissions

**"No relevant results found":**
- Verify ChromaDB is populated with data
- Check that the JSON file contains the expected channel's messages
- Ensure `channel_id` is included in message metadata

**Bot not responding:**
- Check that all environment variables are set
- Verify the bot is running and connected
- Check Slack app configuration and permissions

**Rate Limiting Issues:**
- The loader includes delays to respect Slack's rate limits
- If you encounter issues, increase the sleep times in `slack_history_loader.py`

### Logs and Debugging

All scripts include comprehensive logging. Check the console output for:
- Connection status
- API call results
- Error messages
- Processing progress

## 🤝 Contributing

### Code Style

- Follow PEP 8 guidelines
- Use descriptive variable names
- Add comments for complex logic
- Keep functions focused and small

### Testing Your Changes

1. **Test locally** before pushing
2. **Verify the bot responds** correctly
3. **Check that thread linking** works as expected
4. **Ensure no breaking changes** to existing functionality

### Pull Request Guidelines

- **Clear description** of changes
- **Test results** included
- **Screenshots** if UI changes
- **Update documentation** if needed

## 📞 Support

If you encounter issues:

1. **Check the troubleshooting section** above
2. **Review the logs** for error messages
3. **Verify your configuration** matches the setup guide
4. **Create an issue** with detailed error information

## 📄 License

[Add your license information here]

---

**Happy coding! 🚀**
