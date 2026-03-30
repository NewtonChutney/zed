use std::path::PathBuf;

use anyhow::{Context as _, Result, anyhow};
use futures::{AsyncBufReadExt, AsyncReadExt, StreamExt, io::BufReader, stream::BoxStream};
use http_client::{AsyncBody, HttpClient, Method, Request as HttpRequest};
use serde::{Deserialize, Serialize};
use strum::EnumIter;

pub const DEFAULT_API_URL: &str = "https://us-east5-aiplatform.googleapis.com";
const TOKEN_ENDPOINT: &str = "https://oauth2.googleapis.com/token";

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AdcCredentials {
    pub client_id: String,
    pub client_secret: String,
    pub refresh_token: String,
    #[serde(rename = "type")]
    pub credential_type: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct TokenResponse {
    access_token: String,
    expires_in: Option<u64>,
    token_type: Option<String>,
}

pub fn gcloud_config_dir() -> Option<PathBuf> {
    if let Ok(config_dir) = std::env::var("CLOUDSDK_CONFIG") {
        return Some(PathBuf::from(config_dir));
    }
    dirs::home_dir().map(|home| home.join(".config").join("gcloud"))
}

pub fn read_adc_credentials() -> Result<AdcCredentials> {
    let config_dir = gcloud_config_dir()
        .context("Could not determine gcloud config directory")?;
    let adc_path = config_dir.join("application_default_credentials.json");
    let contents = std::fs::read_to_string(&adc_path)
        .with_context(|| format!("Failed to read ADC file at {}", adc_path.display()))?;
    serde_json::from_str(&contents).context("Failed to parse ADC credentials")
}

pub fn read_default_project() -> Result<String> {
    let config_dir = gcloud_config_dir()
        .context("Could not determine gcloud config directory")?;

    let active_config = config_dir.join("active_config");
    let config_name = if active_config.exists() {
        std::fs::read_to_string(&active_config)
            .unwrap_or_else(|_| "default".to_string())
            .trim()
            .to_string()
    } else {
        "default".to_string()
    };

    let config_path = config_dir
        .join("configurations")
        .join(format!("config_{config_name}"));
    let contents = std::fs::read_to_string(&config_path)
        .with_context(|| format!("Failed to read gcloud config at {}", config_path.display()))?;

    for line in contents.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("project") {
            if let Some((_key, value)) = trimmed.split_once('=') {
                return Ok(value.trim().to_string());
            }
        }
    }
    Err(anyhow!("No project found in gcloud config"))
}

pub async fn refresh_access_token(
    client: &dyn HttpClient,
    credentials: &AdcCredentials,
) -> Result<String> {
    let body = serde_json::json!({
        "client_id": credentials.client_id,
        "client_secret": credentials.client_secret,
        "refresh_token": credentials.refresh_token,
        "grant_type": "refresh_token",
    });

    let request = HttpRequest::builder()
        .method(Method::POST)
        .uri(TOKEN_ENDPOINT)
        .header("Content-Type", "application/json")
        .body(AsyncBody::from(serde_json::to_string(&body)?))?;

    let mut response = client.send(request).await?;
    let mut text = String::new();
    response.body_mut().read_to_string(&mut text).await?;

    anyhow::ensure!(
        response.status().is_success(),
        "Failed to refresh access token: {} {}",
        response.status(),
        text
    );

    let token_response: TokenResponse =
        serde_json::from_str(&text).context("Failed to parse token response")?;
    Ok(token_response.access_token)
}

/// Which publisher a model belongs to on Vertex AI.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Publisher {
    Google,
    Anthropic,
}

impl Publisher {
    pub fn as_str(&self) -> &str {
        match self {
            Self::Google => "google",
            Self::Anthropic => "anthropic",
        }
    }
}

#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, EnumIter)]
pub enum Model {
    // Google models
    #[serde(rename = "gemini-2.5-pro")]
    Gemini25Pro,

    // Anthropic models
    #[default]
    #[serde(rename = "claude-sonnet-4-6")]
    ClaudeSonnet4_6,
    #[serde(rename = "claude-sonnet-4-5")]
    ClaudeSonnet4_5,
    #[serde(rename = "claude-sonnet-4")]
    ClaudeSonnet4,
    #[serde(rename = "claude-opus-4-6")]
    ClaudeOpus4_6,
    #[serde(rename = "claude-opus-4-5")]
    ClaudeOpus4_5,
    #[serde(rename = "claude-haiku-4-5")]
    ClaudeHaiku4_5,
    #[serde(rename = "claude-3-5-haiku")]
    Claude35Haiku,

    #[serde(rename = "custom")]
    Custom {
        name: String,
        display_name: Option<String>,
        max_tokens: u64,
        max_output_tokens: Option<u64>,
        publisher: String,
    },
}

impl Model {
    pub fn default_fast() -> Self {
        Self::Claude35Haiku
    }

    pub fn publisher(&self) -> Publisher {
        match self {
            Self::Gemini25Pro => Publisher::Google,
            Self::ClaudeSonnet4_6
            | Self::ClaudeSonnet4_5
            | Self::ClaudeSonnet4
            | Self::ClaudeOpus4_6
            | Self::ClaudeOpus4_5
            | Self::ClaudeHaiku4_5
            | Self::Claude35Haiku => Publisher::Anthropic,
            Self::Custom { publisher, .. } => {
                if publisher == "google" {
                    Publisher::Google
                } else {
                    Publisher::Anthropic
                }
            }
        }
    }

    pub fn id(&self) -> &str {
        match self {
            Self::Gemini25Pro => "gemini-2.5-pro",
            Self::ClaudeSonnet4_6 => "claude-sonnet-4-6",
            Self::ClaudeSonnet4_5 => "claude-sonnet-4-5",
            Self::ClaudeSonnet4 => "claude-sonnet-4",
            Self::ClaudeOpus4_6 => "claude-opus-4-6",
            Self::ClaudeOpus4_5 => "claude-opus-4-5",
            Self::ClaudeHaiku4_5 => "claude-haiku-4-5",
            Self::Claude35Haiku => "claude-3-5-haiku",
            Self::Custom { name, .. } => name,
        }
    }

    pub fn display_name(&self) -> &str {
        match self {
            Self::Gemini25Pro => "Gemini 2.5 Pro (Vertex)",
            Self::ClaudeSonnet4_6 => "Claude Sonnet 4.6 (Vertex)",
            Self::ClaudeSonnet4_5 => "Claude Sonnet 4.5 (Vertex)",
            Self::ClaudeSonnet4 => "Claude Sonnet 4 (Vertex)",
            Self::ClaudeOpus4_6 => "Claude Opus 4.6 (Vertex)",
            Self::ClaudeOpus4_5 => "Claude Opus 4.5 (Vertex)",
            Self::ClaudeHaiku4_5 => "Claude Haiku 4.5 (Vertex)",
            Self::Claude35Haiku => "Claude 3.5 Haiku (Vertex)",
            Self::Custom {
                name, display_name, ..
            } => display_name.as_ref().unwrap_or(name),
        }
    }

    pub fn max_token_count(&self) -> u64 {
        match self {
            Self::Gemini25Pro => 1_048_576,
            Self::ClaudeSonnet4_6 | Self::ClaudeOpus4_6 => 1_000_000,
            Self::ClaudeSonnet4_5
            | Self::ClaudeSonnet4
            | Self::ClaudeOpus4_5
            | Self::ClaudeHaiku4_5
            | Self::Claude35Haiku => 200_000,
            Self::Custom { max_tokens, .. } => *max_tokens,
        }
    }

    pub fn max_output_tokens(&self) -> Option<u64> {
        match self {
            Self::Gemini25Pro => Some(65_536),
            Self::ClaudeOpus4_6 => Some(128_000),
            Self::ClaudeOpus4_5 => Some(32_000),
            Self::ClaudeSonnet4_6
            | Self::ClaudeSonnet4_5
            | Self::ClaudeSonnet4
            | Self::ClaudeHaiku4_5 => Some(64_000),
            Self::Claude35Haiku => Some(8_192),
            Self::Custom {
                max_output_tokens, ..
            } => *max_output_tokens,
        }
    }

    pub fn supports_tools(&self) -> bool {
        true
    }

    pub fn supports_images(&self) -> bool {
        true
    }

    pub fn supports_thinking(&self) -> bool {
        !matches!(self, Self::Claude35Haiku)
    }

    /// The model ID to use in the Vertex AI API request URL.
    /// Uses the bare model name without @date suffix for broadest compatibility.
    pub fn vertex_model_id(&self) -> &str {
        self.id()
    }
}

fn vertex_base_url(api_url: &str, project_id: &str, location_id: &str) -> String {
    let host = if api_url == DEFAULT_API_URL {
        format!("https://{location_id}-aiplatform.googleapis.com")
    } else {
        api_url.to_string()
    };
    format!(
        "{host}/v1/projects/{project_id}/locations/{location_id}"
    )
}

/// Stream a completion for a Google publisher model (Gemini) on Vertex AI.
pub async fn stream_generate_content(
    client: &dyn HttpClient,
    api_url: &str,
    access_token: &str,
    project_id: &str,
    location_id: &str,
    model_id: &str,
    request: google_ai::GenerateContentRequest,
) -> Result<BoxStream<'static, Result<google_ai::GenerateContentResponse>>> {
    let base_url = vertex_base_url(api_url, project_id, location_id);
    let uri = format!(
        "{base_url}/publishers/google/models/{model_id}:streamGenerateContent?alt=sse"
    );

    let request_builder = HttpRequest::builder()
        .method(Method::POST)
        .uri(&uri)
        .header("Content-Type", "application/json")
        .header("Authorization", format!("Bearer {access_token}"));

    let body = serde_json::to_string(&request)?;
    let http_request = request_builder.body(AsyncBody::from(body))?;
    let mut response = client.send(http_request).await?;

    if response.status().is_success() {
        let reader = BufReader::new(response.into_body());
        Ok(reader
            .lines()
            .filter_map(|line| async move {
                match line {
                    Ok(line) => {
                        if let Some(line) = line.strip_prefix("data: ") {
                            match serde_json::from_str(line) {
                                Ok(response) => Some(Ok(response)),
                                Err(error) => Some(Err(anyhow!(
                                    "Error parsing JSON: {error:?}\n{line:?}"
                                ))),
                            }
                        } else {
                            None
                        }
                    }
                    Err(error) => Some(Err(anyhow!(error))),
                }
            })
            .boxed())
    } else {
        let mut text = String::new();
        response.body_mut().read_to_string(&mut text).await?;
        Err(anyhow!(
            "Vertex AI streamGenerateContent error, status: {:?}, body: {}",
            response.status(),
            text
        ))
    }
}

/// Stream a completion for an Anthropic publisher model (Claude) on Vertex AI.
pub async fn stream_raw_predict(
    client: &dyn HttpClient,
    api_url: &str,
    access_token: &str,
    project_id: &str,
    location_id: &str,
    model_id: &str,
    request: anthropic::Request,
) -> Result<
    BoxStream<'static, Result<anthropic::Event, anthropic::AnthropicError>>,
    anthropic::AnthropicError,
> {
    let base_url = vertex_base_url(api_url, project_id, location_id);
    let uri = format!(
        "{base_url}/publishers/anthropic/models/{model_id}:streamRawPredict"
    );

    let streaming_request = anthropic::StreamingRequest {
        base: request,
        stream: true,
    };

    let mut body = serde_json::to_value(&streaming_request)
        .map_err(anthropic::AnthropicError::SerializeRequest)?;

    // Vertex AI provides the model in the URL, not the body.
    // The API rejects extra fields like "model".
    if let Some(object) = body.as_object_mut() {
        object.remove("model");
        object.insert(
            "anthropic_version".to_string(),
            serde_json::Value::String("vertex-2023-10-16".to_string()),
        );
    }

    let serialized = serde_json::to_string(&body)
        .map_err(anthropic::AnthropicError::SerializeRequest)?;

    let http_request = HttpRequest::builder()
        .method(Method::POST)
        .uri(&uri)
        .header("Content-Type", "application/json")
        .header("Authorization", format!("Bearer {access_token}"))
        .body(AsyncBody::from(serialized))
        .map_err(anthropic::AnthropicError::BuildRequestBody)?;

    let response = client
        .send(http_request)
        .await
        .map_err(anthropic::AnthropicError::HttpSend)?;

    if response.status().is_success() {
        let reader = BufReader::new(response.into_body());
        Ok(reader
            .lines()
            .filter_map(|line| async move {
                match line {
                    Ok(line) => {
                        let line = line
                            .strip_prefix("data: ")
                            .or_else(|| line.strip_prefix("data:"))?;
                        match serde_json::from_str(line) {
                            Ok(response) => Some(Ok(response)),
                            Err(error) => {
                                Some(Err(anthropic::AnthropicError::DeserializeResponse(error)))
                            }
                        }
                    }
                    Err(error) => Some(Err(anthropic::AnthropicError::ReadResponse(error))),
                }
            })
            .boxed())
    } else {
        let status_code = response.status();
        let mut body = String::new();
        let _ = response
            .into_body()
            .read_to_string(&mut body)
            .await;
        Err(anthropic::AnthropicError::HttpResponseError {
            status_code,
            message: format!("Vertex AI streamRawPredict error: {body}"),
        })
    }
}
