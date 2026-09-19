use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Lenient OpenAI-compatible chat request. MinusPod always sends a single
/// system + single user message with string content; accept arrays too.
#[derive(Debug, Deserialize)]
pub struct ChatRequest {
    #[serde(default)]
    pub model: String,
    #[serde(default)]
    pub messages: Vec<ChatMessage>,
    #[serde(default)]
    #[allow(dead_code)]
    pub temperature: Option<f64>,
    #[serde(default)]
    pub max_tokens: Option<u32>,
    #[serde(default)]
    pub max_completion_tokens: Option<u32>,
    #[serde(default)]
    pub response_format: Option<Value>,
    #[serde(default)]
    pub stream: Option<bool>,
}

#[derive(Debug, Deserialize)]
pub struct ChatMessage {
    #[serde(default)]
    pub role: String,
    #[serde(default)]
    pub content: Value,
}

impl ChatMessage {
    pub fn text(&self) -> String {
        match &self.content {
            Value::String(s) => s.clone(),
            Value::Array(parts) => parts
                .iter()
                .filter_map(|p| {
                    p.get("text")
                        .and_then(Value::as_str)
                        .map(str::to_string)
                })
                .collect::<Vec<_>>()
                .join("\n"),
            _ => String::new(),
        }
    }
}

impl ChatRequest {
    pub fn system_text(&self) -> String {
        self.messages
            .iter()
            .find(|m| m.role == "system")
            .map(|m| m.text())
            .unwrap_or_default()
    }

    pub fn user_text(&self) -> String {
        let mut out: Vec<String> = self
            .messages
            .iter()
            .filter(|m| m.role == "user")
            .map(|m| m.text())
            .collect();
        if out.is_empty() {
            // Fall back to any non-system message.
            out = self
                .messages
                .iter()
                .filter(|m| m.role != "system")
                .map(|m| m.text())
                .collect();
        }
        out.join("\n")
    }

    /// response_format.json_schema.name, e.g. ad_detection / ad_review /
    /// trim_recovery / segment_categories.
    pub fn schema_name(&self) -> Option<String> {
        self.response_format
            .as_ref()?
            .get("json_schema")?
            .get("name")?
            .as_str()
            .map(str::to_string)
    }

    pub fn token_budget(&self) -> Option<u32> {
        self.max_tokens.or(self.max_completion_tokens)
    }
}

#[derive(Debug, Serialize)]
pub struct ChatResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatChoice>,
    pub usage: Usage,
}

#[derive(Debug, Serialize)]
pub struct ChatChoice {
    pub index: u32,
    pub message: ChatMessageOut,
    pub finish_reason: String,
}

#[derive(Debug, Serialize)]
pub struct ChatMessageOut {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Serialize)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

pub fn chat_response(model: &str, content: String) -> ChatResponse {
    ChatResponse {
        id: format!("chatcmpl-jev-{}", now_secs()),
        object: "chat.completion".to_string(),
        created: now_secs(),
        model: model.to_string(),
        choices: vec![ChatChoice {
            index: 0,
            message: ChatMessageOut {
                role: "assistant".to_string(),
                content,
            },
            finish_reason: "stop".to_string(),
        }],
        usage: Usage {
            prompt_tokens: 0,
            completion_tokens: 0,
            total_tokens: 0,
        },
    }
}

fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}
