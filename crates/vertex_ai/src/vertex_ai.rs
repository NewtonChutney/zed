use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{Context as _, Result, anyhow};
use futures::{AsyncBufReadExt, AsyncReadExt, StreamExt, io::BufReader, stream::BoxStream};
use http_client::{AsyncBody, HttpClient, Method, Request as HttpRequest};
use jsonwebtoken::{Algorithm, EncodingKey, Header};
use serde::{Deserialize, Serialize};
use strum::EnumIter;

/// Metadata for a well-known Vertex AI model used to populate capabilities
/// when a model is discovered at runtime.
struct KnownModelMetadata {
    id: &'static str,
    display_name: &'static str,
    publisher: Publisher,
    max_tokens: u64,
    max_output_tokens: u64,
    supports_thinking: bool,
    /// Whether the model accepts Anthropic's `thinking: {type: "adaptive"}` mode.
    /// Models before Claude 4.6 (and all Haiku models) only support the legacy
    /// `enabled`/`disabled` thinking config and reject `adaptive` with a 400.
    supports_adaptive_thinking: bool,
}

const KNOWN_MODELS: &[KnownModelMetadata] = &[
    KnownModelMetadata {
        id: "gemini-2.5-pro",
        display_name: "Gemini 2.5 Pro (Vertex)",
        publisher: Publisher::Google,
        max_tokens: 1_048_576,
        max_output_tokens: 65_536,
        supports_thinking: true,
        supports_adaptive_thinking: false,
    },
    KnownModelMetadata {
        id: "gemini-2.5-flash",
        display_name: "Gemini 2.5 Flash (Vertex)",
        publisher: Publisher::Google,
        max_tokens: 1_048_576,
        max_output_tokens: 65_536,
        supports_thinking: true,
        supports_adaptive_thinking: false,
    },
    KnownModelMetadata {
        id: "claude-sonnet-5",
        display_name: "Claude Sonnet 5 (Vertex)",
        publisher: Publisher::Anthropic,
        max_tokens: 1_000_000,
        max_output_tokens: 128_000,
        supports_thinking: true,
        supports_adaptive_thinking: true,
    },
    KnownModelMetadata {
        id: "claude-sonnet-4-6",
        display_name: "Claude Sonnet 4.6 (Vertex)",
        publisher: Publisher::Anthropic,
        max_tokens: 1_000_000,
        max_output_tokens: 64_000,
        supports_thinking: true,
        supports_adaptive_thinking: true,
    },
    KnownModelMetadata {
        id: "claude-sonnet-4-5",
        display_name: "Claude Sonnet 4.5 (Vertex)",
        publisher: Publisher::Anthropic,
        max_tokens: 200_000,
        max_output_tokens: 64_000,
        supports_thinking: true,
        supports_adaptive_thinking: false,
    },
    KnownModelMetadata {
        id: "claude-sonnet-4",
        display_name: "Claude Sonnet 4 (Vertex)",
        publisher: Publisher::Anthropic,
        max_tokens: 200_000,
        max_output_tokens: 64_000,
        supports_thinking: true,
        supports_adaptive_thinking: false,
    },
    KnownModelMetadata {
        id: "claude-opus-5",
        display_name: "Claude Opus 5 (Vertex)",
        publisher: Publisher::Anthropic,
        max_tokens: 1_000_000,
        max_output_tokens: 128_000,
        supports_thinking: true,
        supports_adaptive_thinking: true,
    },
    KnownModelMetadata {
        id: "claude-opus-4-8",
        display_name: "Claude Opus 4.8 (Vertex)",
        publisher: Publisher::Anthropic,
        max_tokens: 1_000_000,
        max_output_tokens: 128_000,
        supports_thinking: true,
        supports_adaptive_thinking: true,
    },
    KnownModelMetadata {
        id: "claude-opus-4-7",
        display_name: "Claude Opus 4.7 (Vertex)",
        publisher: Publisher::Anthropic,
        max_tokens: 1_000_000,
        max_output_tokens: 128_000,
        supports_thinking: true,
        supports_adaptive_thinking: true,
    },
    KnownModelMetadata {
        id: "claude-opus-4-6",
        display_name: "Claude Opus 4.6 (Vertex)",
        publisher: Publisher::Anthropic,
        max_tokens: 1_000_000,
        max_output_tokens: 128_000,
        supports_thinking: true,
        supports_adaptive_thinking: true,
    },
    KnownModelMetadata {
        id: "claude-opus-4-5",
        display_name: "Claude Opus 4.5 (Vertex)",
        publisher: Publisher::Anthropic,
        max_tokens: 200_000,
        max_output_tokens: 32_000,
        supports_thinking: true,
        supports_adaptive_thinking: false,
    },
    KnownModelMetadata {
        id: "claude-haiku-4-5",
        display_name: "Claude Haiku 4.5 (Vertex)",
        publisher: Publisher::Anthropic,
        max_tokens: 200_000,
        max_output_tokens: 64_000,
        supports_thinking: true,
        supports_adaptive_thinking: false,
    },
    KnownModelMetadata {
        id: "claude-3-5-haiku",
        display_name: "Claude 3.5 Haiku (Vertex)",
        publisher: Publisher::Anthropic,
        max_tokens: 200_000,
        max_output_tokens: 8_192,
        supports_thinking: false,
        supports_adaptive_thinking: false,
    },
    KnownModelMetadata {
        id: "claude-fable-5",
        display_name: "Claude Fable 5 (Vertex)",
        publisher: Publisher::Anthropic,
        max_tokens: 1_000_000,
        max_output_tokens: 128_000,
        supports_thinking: true,
        supports_adaptive_thinking: true,
    },
    KnownModelMetadata {
        id: "claude-mythos-5",
        display_name: "Claude Mythos 5 (Vertex)",
        publisher: Publisher::Anthropic,
        max_tokens: 1_000_000,
        max_output_tokens: 128_000,
        supports_thinking: true,
        supports_adaptive_thinking: true,
    },
];

/// Anthropic model IDs on Vertex AI that have been confirmed to support
/// Claude's `thinking: {type: "adaptive"}` mode. Used to classify Anthropic
/// models discovered dynamically (e.g. via org policy) that aren't in
/// `KNOWN_MODELS`. Anything not in this list falls back to the legacy
/// `enabled`/`disabled` thinking config, since sending `adaptive` to an
/// unsupported model is rejected with a 400.
const ADAPTIVE_THINKING_MODEL_IDS: &[&str] = &[
    "claude-sonnet-4-6",
    "claude-opus-4-6",
    "claude-opus-4-7",
    "claude-opus-4-8",
    "claude-opus-5",
    "claude-sonnet-5",
    "claude-fable-5",
    "claude-mythos-5",
];

pub const DEFAULT_API_URL: &str = "https://us-east5-aiplatform.googleapis.com";

fn default_true() -> bool {
    true
}
const TOKEN_ENDPOINT: &str = "https://oauth2.googleapis.com/token";

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AdcCredentials {
    pub client_id: String,
    pub client_secret: String,
    pub refresh_token: String,
    #[serde(rename = "type")]
    pub credential_type: Option<String>,
    pub quota_project_id: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct TokenResponse {
    access_token: String,
    expires_in: Option<u64>,
    token_type: Option<String>,
}

#[derive(Clone, Debug)]
pub struct AccessToken {
    pub token: String,
    pub expires_at: Option<Instant>,
}

impl AccessToken {
    pub fn is_expired(&self) -> bool {
        self.expires_at
            .map(|expires_at| Instant::now() >= expires_at)
            .unwrap_or(false)
    }
}

pub fn gcloud_config_dir() -> Option<PathBuf> {
    if let Ok(config_dir) = std::env::var("CLOUDSDK_CONFIG") {
        return Some(PathBuf::from(config_dir));
    }
    dirs::home_dir().map(|home| home.join(".config").join("gcloud"))
}

pub fn read_adc_credentials() -> Result<AdcCredentials> {
    let config_dir = gcloud_config_dir().context("Could not determine gcloud config directory")?;
    let adc_path = config_dir.join("application_default_credentials.json");
    let contents = std::fs::read_to_string(&adc_path)
        .with_context(|| format!("Failed to read ADC file at {}", adc_path.display()))?;
    serde_json::from_str(&contents).context("Failed to parse ADC credentials")
}

pub fn read_default_project() -> Result<String> {
    let config_dir = gcloud_config_dir().context("Could not determine gcloud config directory")?;

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

/// Sends a token-endpoint request and parses the resulting `AccessToken`.
/// Shared by the ADC refresh flow and the service-account JWT exchange,
/// which differ only in how they build the request.
async fn send_token_request(
    client: &dyn HttpClient,
    request: HttpRequest<AsyncBody>,
    error_context: &str,
) -> Result<AccessToken> {
    let mut response = client.send(request).await?;
    let mut text = String::new();
    response.body_mut().read_to_string(&mut text).await?;

    anyhow::ensure!(
        response.status().is_success(),
        "{error_context}: {} {}",
        response.status(),
        text
    );

    let token_response: TokenResponse =
        serde_json::from_str(&text).context("Failed to parse token response")?;

    // Refresh 60 seconds before actual expiry to avoid races
    let expires_at = token_response
        .expires_in
        .map(|secs| Instant::now() + Duration::from_secs(secs.saturating_sub(60)));

    Ok(AccessToken {
        token: token_response.access_token,
        expires_at,
    })
}

pub async fn refresh_access_token(
    client: &dyn HttpClient,
    credentials: &AdcCredentials,
) -> Result<AccessToken> {
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

    send_token_request(client, request, "Failed to refresh access token").await
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ServiceAccountCredentials {
    pub project_id: Option<String>,
    pub private_key: String,
    pub client_email: String,
    pub token_uri: Option<String>,
}

pub fn read_service_account_credentials(path: &str) -> Result<ServiceAccountCredentials> {
    let contents = std::fs::read_to_string(path)
        .with_context(|| format!("Failed to read service account key file at {path}"))?;
    serde_json::from_str(&contents).context("Failed to parse service account key file")
}

#[derive(Serialize)]
struct JwtClaims {
    iss: String,
    scope: String,
    aud: String,
    iat: u64,
    exp: u64,
}

pub async fn exchange_service_account_token(
    client: &dyn HttpClient,
    credentials: &ServiceAccountCredentials,
) -> Result<AccessToken> {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .context("System time before UNIX epoch")?
        .as_secs();

    let token_uri = credentials
        .token_uri
        .as_deref()
        .unwrap_or(TOKEN_ENDPOINT);

    let claims = JwtClaims {
        iss: credentials.client_email.clone(),
        scope: "https://www.googleapis.com/auth/cloud-platform".to_string(),
        aud: token_uri.to_string(),
        iat: now,
        exp: now + 3600,
    };

    let key = EncodingKey::from_rsa_pem(credentials.private_key.as_bytes())
        .context("Failed to parse service account private key")?;

    let jwt = jsonwebtoken::encode(&Header::new(Algorithm::RS256), &claims, &key)
        .context("Failed to create JWT for service account")?;

    let body = format!(
        "grant_type={}&assertion={}",
        "urn:ietf:params:oauth:grant-type:jwt-bearer",
        jwt
    );

    let request = HttpRequest::builder()
        .method(Method::POST)
        .uri(token_uri)
        .header("Content-Type", "application/x-www-form-urlencoded")
        .body(AsyncBody::from(body))?;

    send_token_request(
        client,
        request,
        "Failed to exchange service account JWT for access token",
    )
    .await
}

const ORG_POLICY_URL: &str = "https://cloudresourcemanager.googleapis.com/v1/projects";

#[derive(Deserialize)]
struct OrgPolicyResponse {
    #[serde(rename = "listPolicy")]
    list_policy: Option<OrgPolicyListPolicy>,
}

#[derive(Deserialize)]
struct OrgPolicyListPolicy {
    #[serde(rename = "allowedValues", default)]
    allowed_values: Vec<String>,
    #[serde(rename = "allValues", default)]
    all_values: Option<String>,
}

/// Query the effective org policy for `constraints/vertexai.allowedModels`
/// to get the list of model IDs the project is allowed to use.
/// Returns `None` if the API call fails (e.g. insufficient permissions).
async fn fetch_org_policy_allowed_models(
    client: &dyn HttpClient,
    access_token: &str,
    project_id: &str,
) -> Option<Vec<(String, String)>> {
    let uri = format!("{ORG_POLICY_URL}/{project_id}:getEffectiveOrgPolicy");
    let body = serde_json::json!({
        "constraint": "constraints/vertexai.allowedModels"
    });

    let request = HttpRequest::builder()
        .method(Method::POST)
        .uri(&uri)
        .header("Content-Type", "application/json")
        .header("Authorization", format!("Bearer {access_token}"))
        .body(AsyncBody::from(serde_json::to_string(&body).ok()?))
        .ok()?;

    let mut response = client.send(request).await.ok()?;
    if !response.status().is_success() {
        return None;
    }

    let mut text = String::new();
    response.body_mut().read_to_string(&mut text).await.ok()?;

    let policy: OrgPolicyResponse = serde_json::from_str(&text).ok()?;
    let list_policy = policy.list_policy?;

    // If allValues is "ALLOW", there's no restriction
    if list_policy.all_values.as_deref() == Some("ALLOW") {
        return None;
    }

    // Parse entries like "is:publishers/anthropic/models/claude-sonnet-4-6:predict"
    // into (model_id, publisher) pairs.
    let entries: Vec<(String, String)> = list_policy
        .allowed_values
        .iter()
        .filter_map(|entry| {
            let entry = entry.strip_prefix("is:")?;
            let entry = entry.strip_suffix(":predict").unwrap_or(entry);
            let parts: Vec<&str> = entry.split('/').collect();
            if parts.len() >= 4 && parts[0] == "publishers" && parts[2] == "models" {
                Some((parts[3].to_string(), parts[1].to_string()))
            } else {
                None
            }
        })
        .collect();

    Some(entries)
}

/// Check whether a single model is accessible by making a GET request
/// to the publisher model endpoint. Returns `true` if the model exists
/// and the caller has access.
async fn check_model_accessible(
    client: &dyn HttpClient,
    api_url: &str,
    access_token: &str,
    project_id: &str,
    location_id: &str,
    publisher: &str,
    model_id: &str,
    quota_project_id: Option<&str>,
) -> bool {
    let base_url = vertex_base_url(api_url, project_id, location_id);
    let uri = format!("{base_url}/publishers/{publisher}/models/{model_id}");

    let mut request_builder = HttpRequest::builder()
        .method(Method::GET)
        .uri(&uri)
        .header("Authorization", format!("Bearer {access_token}"));
    if let Some(project) = quota_project_id {
        request_builder = request_builder.header("x-goog-user-project", project);
    }
    let request = match request_builder.body(AsyncBody::default()) {
        Ok(request) => request,
        Err(_) => return false,
    };

    match client.send(request).await {
        Ok(response) => response.status().is_success(),
        Err(_) => false,
    }
}

fn known_model_to_model(metadata: &KnownModelMetadata) -> Model {
    Model::Custom {
        name: metadata.id.to_string(),
        display_name: Some(metadata.display_name.to_string()),
        max_tokens: metadata.max_tokens,
        max_output_tokens: Some(metadata.max_output_tokens),
        publisher: metadata.publisher.as_str().to_string(),
        supports_thinking: metadata.supports_thinking,
        supports_adaptive_thinking: metadata.supports_adaptive_thinking,
    }
}

fn model_from_discovered(model_id: &str, publisher: &str) -> Model {
    if let Some(metadata) = KNOWN_MODELS.iter().find(|m| m.id == model_id) {
        return known_model_to_model(metadata);
    }

    let (max_tokens, max_output_tokens, supports_thinking) = match publisher {
        "google" => (1_048_576, 65_536, true),
        "anthropic" => (200_000, 64_000, true),
        other => {
            log::warn!("Vertex AI: unknown publisher {other:?} for model {model_id:?}");
            (200_000, 8_192, false)
        }
    };
    let supports_adaptive_thinking =
        publisher == "anthropic" && ADAPTIVE_THINKING_MODEL_IDS.contains(&model_id);

    let display_name = format_model_display_name(model_id);

    Model::Custom {
        name: model_id.to_string(),
        display_name: Some(display_name),
        max_tokens,
        max_output_tokens: Some(max_output_tokens),
        publisher: publisher.to_string(),
        supports_thinking,
        supports_adaptive_thinking,
    }
}

fn format_model_display_name(model_id: &str) -> String {
    let words: Vec<String> = model_id
        .split('-')
        .map(|word| {
            let mut characters = word.chars();
            match characters.next() {
                Some(c) if c.is_alphabetic() => {
                    let mut capitalized = c.to_uppercase().to_string();
                    capitalized.push_str(characters.as_str());
                    capitalized
                }
                _ => word.to_string(),
            }
        })
        .collect();
    format!("{} (Vertex)", words.join(" "))
}

/// Fetch the list of models accessible to the authenticated user.
/// First tries the org policy API (single request), falling back to
/// per-model availability probing if the org policy is inaccessible.
pub async fn fetch_available_models(
    client: Arc<dyn HttpClient>,
    api_url: String,
    access_token: String,
    project_id: String,
    location_id: String,
    quota_project_id: Option<String>,
) -> Vec<Model> {
    // Try org policy first — single API call
    if let Some(allowed_entries) =
        fetch_org_policy_allowed_models(client.as_ref(), &access_token, &project_id).await
    {
        let models: Vec<Model> = allowed_entries
            .iter()
            .map(|(model_id, publisher)| model_from_discovered(model_id, publisher))
            .collect();
        let model_names: Vec<&str> = models.iter().map(|m| m.id()).collect();
        log::info!(
            "Vertex AI: discovered {} models via org policy (project: {}): {:?}",
            models.len(),
            project_id,
            model_names
        );
        return models;
    }

    log::info!("Vertex AI: org policy unavailable, probing models individually");

    // Fallback: probe each model individually
    let tasks: Vec<_> = KNOWN_MODELS
        .iter()
        .map(|metadata| {
            let client = client.clone();
            let api_url = api_url.clone();
            let access_token = access_token.clone();
            let project_id = project_id.clone();
            let location_id = location_id.clone();
            let quota_project_id = quota_project_id.clone();
            let id = metadata.id;
            let publisher = metadata.publisher;

            async move {
                let accessible = check_model_accessible(
                    client.as_ref(),
                    &api_url,
                    &access_token,
                    &project_id,
                    &location_id,
                    publisher.as_str(),
                    id,
                    quota_project_id.as_deref(),
                )
                .await;
                accessible.then_some(id)
            }
        })
        .collect();

    let accessible_ids: Vec<&str> = futures::stream::iter(tasks)
        .buffer_unordered(5)
        .filter_map(|result| async move { result })
        .collect()
        .await;

    let models: Vec<Model> = KNOWN_MODELS
        .iter()
        .filter(|metadata| accessible_ids.contains(&metadata.id))
        .map(known_model_to_model)
        .collect();
    let model_names: Vec<&str> = models.iter().map(|m| m.id()).collect();
    log::info!(
        "Vertex AI: discovered {} accessible models via probing: {:?}",
        models.len(),
        model_names
    );
    models
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
        #[serde(default = "default_true")]
        supports_thinking: bool,
        #[serde(default)]
        supports_adaptive_thinking: bool,
    },
}

pub const FAST_MODEL_PREFERENCE: &[&str] = &[
    "gemini-3-flash",
    "gemini-3.5-flash",
    "gemini-3-1-flash-lite",
    "claude-haiku-4-5",
    "claude-3-5-haiku",
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
];

impl Model {
    pub fn default_fast() -> Self {
        Self::ClaudeHaiku4_5
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
        match self {
            Self::Claude35Haiku => false,
            Self::Custom {
                supports_thinking, ..
            } => *supports_thinking,
            _ => true,
        }
    }

    /// Whether the model accepts Anthropic's `thinking: {type: "adaptive"}` mode.
    /// Models before Claude 4.6 only support the legacy `enabled`/`disabled`
    /// thinking config and reject `adaptive` with a 400.
    pub fn supports_adaptive_thinking(&self) -> bool {
        match self {
            Self::ClaudeSonnet4_6 | Self::ClaudeOpus4_6 => true,
            Self::ClaudeSonnet4_5
            | Self::ClaudeSonnet4
            | Self::ClaudeOpus4_5
            | Self::ClaudeHaiku4_5
            | Self::Claude35Haiku
            | Self::Gemini25Pro => false,
            Self::Custom {
                supports_adaptive_thinking,
                ..
            } => *supports_adaptive_thinking,
        }
    }

    /// The model ID to use in the Vertex AI API request URL.
    /// Uses the bare model name without @date suffix for broadest compatibility.
    pub fn vertex_model_id(&self) -> &str {
        self.id()
    }
}

fn vertex_base_url(api_url: &str, project_id: &str, location_id: &str) -> String {
    let host = if api_url == DEFAULT_API_URL {
        if location_id == "global" {
            "https://aiplatform.googleapis.com".to_string()
        } else {
            format!("https://{location_id}-aiplatform.googleapis.com")
        }
    } else {
        api_url.to_string()
    };
    format!("{host}/v1/projects/{project_id}/locations/{location_id}")
}

/// Stream a completion for a Google publisher model (Gemini) on Vertex AI.
pub async fn stream_generate_content(
    client: &dyn HttpClient,
    api_url: &str,
    access_token: &str,
    project_id: &str,
    location_id: &str,
    model_id: &str,
    quota_project_id: Option<&str>,
    request: google_ai::GenerateContentRequest,
) -> Result<BoxStream<'static, Result<google_ai::GenerateContentResponse>>> {
    let base_url = vertex_base_url(api_url, project_id, location_id);
    let uri =
        format!("{base_url}/publishers/google/models/{model_id}:streamGenerateContent?alt=sse");

    let mut request_builder = HttpRequest::builder()
        .method(Method::POST)
        .uri(&uri)
        .header("Content-Type", "application/json")
        .header("Authorization", format!("Bearer {access_token}"));
    if let Some(project) = quota_project_id {
        request_builder = request_builder.header("x-goog-user-project", project);
    }

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
                                Err(error) => {
                                    Some(Err(anyhow!("Error parsing JSON: {error:?}\n{line:?}")))
                                }
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

/// Remove thinking and redacted_thinking content blocks from assistant messages
/// in the serialized request body. Vertex AI rejects thinking signatures from
/// previous conversation turns, so we strip them before sending.
fn strip_thinking_blocks(body: &mut serde_json::Value) {
    let messages = match body.get_mut("messages").and_then(|m| m.as_array_mut()) {
        Some(messages) => messages,
        None => return,
    };

    for message in messages {
        let is_assistant = message
            .get("role")
            .and_then(|r| r.as_str())
            .map(|r| r == "assistant")
            .unwrap_or(false);

        if !is_assistant {
            continue;
        }

        if let Some(content) = message.get_mut("content").and_then(|c| c.as_array_mut()) {
            content.retain(|block| {
                let block_type = block.get("type").and_then(|t| t.as_str()).unwrap_or("");
                block_type != "thinking" && block_type != "redacted_thinking"
            });
        }
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
    quota_project_id: Option<&str>,
    request: anthropic::Request,
) -> Result<
    BoxStream<'static, Result<anthropic::Event, anthropic::AnthropicError>>,
    anthropic::AnthropicError,
> {
    let base_url = vertex_base_url(api_url, project_id, location_id);
    let uri = format!("{base_url}/publishers/anthropic/models/{model_id}:streamRawPredict");

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

    // Strip thinking/redacted_thinking blocks from assistant messages.
    // Vertex AI rejects signatures from previous turns as invalid.
    strip_thinking_blocks(&mut body);

    let serialized =
        serde_json::to_string(&body).map_err(anthropic::AnthropicError::SerializeRequest)?;

    let mut request_builder = HttpRequest::builder()
        .method(Method::POST)
        .uri(&uri)
        .header("Content-Type", "application/json")
        .header("Authorization", format!("Bearer {access_token}"));
    if let Some(project) = quota_project_id {
        request_builder = request_builder.header("x-goog-user-project", project);
    }
    let http_request = request_builder
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
        let _ = response.into_body().read_to_string(&mut body).await;
        Err(anthropic::AnthropicError::HttpResponseError {
            status_code,
            message: format!("Vertex AI streamRawPredict error: {body}"),
        })
    }
}
