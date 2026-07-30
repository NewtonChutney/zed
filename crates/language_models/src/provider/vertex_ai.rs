use anyhow::Result;
use collections::BTreeMap;
use credentials_provider::CredentialsProvider;
use futures::{FutureExt, StreamExt, future::BoxFuture};
use google_ai::GoogleModelMode;
use gpui::{App, AsyncApp, Context, Entity, Task};
use http_client::HttpClient;
use language_model::{
    AuthenticateError, LanguageModelCompletionError, LanguageModelCompletionEvent,
    LanguageModelToolChoice, LanguageModelToolSchemaFormat,
};
use language_model::{
    EnvVar, IconOrSvg, LanguageModel, LanguageModelId, LanguageModelName, LanguageModelProvider,
    LanguageModelProviderId, LanguageModelProviderName, LanguageModelProviderState,
    LanguageModelRequest, ProviderSettingsView, RateLimiter, SubPageProviderSettings, env_var,
};
pub use settings::VertexAiAvailableModel as AvailableModel;
use settings::{Settings, SettingsStore};
use std::sync::{Arc, LazyLock, Mutex as StdMutex};
use strum::IntoEnumIterator;
use ui::{ConfiguredApiCard, List, ListBulletItem, prelude::*};
use util::ResultExt;
use vertex_ai::Publisher;

use crate::provider::anthropic::{AnthropicEventMapper, AnthropicPromptCacheMode, into_anthropic};
use crate::provider::google::{GoogleEventMapper, into_google};

const PROVIDER_ID: LanguageModelProviderId = LanguageModelProviderId::new("vertex_ai");
const PROVIDER_NAME: LanguageModelProviderName = LanguageModelProviderName::new("Vertex AI");

static ZED_VERTEX_ACCESS_TOKEN_VAR: LazyLock<EnvVar> = env_var!("ZED_VERTEX_ACCESS_TOKEN");
static ZED_GOOGLE_APPLICATION_CREDENTIALS_VAR: LazyLock<EnvVar> =
    env_var!("ZED_GOOGLE_APPLICATION_CREDENTIALS");

const GOOGLE_VERTEX_AI_URL: &str = "https://vertex.googleapis.com";

#[derive(Clone, Debug)]
pub enum VertexAiAuth {
    Adc {
        credentials: vertex_ai::AdcCredentials,
    },
    ServiceAccountKeyFile {
        credentials: vertex_ai::ServiceAccountCredentials,
    },
    AccessToken {
        token: String,
    },
}

#[derive(Default, Clone, Debug, PartialEq)]
pub struct VertexAiSettings {
    pub api_url: String,
    pub project_id: String,
    pub location_id: String,
    pub authentication_method: Option<VertexAiAuthMethod>,
    pub service_account_key_file: Option<String>,
    pub available_models: Vec<AvailableModel>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum VertexAiAuthMethod {
    Adc,
    ServiceAccountKeyFile,
    AccessToken,
    Automatic,
}

impl From<settings::VertexAiAuthMethodContent> for VertexAiAuthMethod {
    fn from(value: settings::VertexAiAuthMethodContent) -> Self {
        match value {
            settings::VertexAiAuthMethodContent::Adc => Self::Adc,
            settings::VertexAiAuthMethodContent::ServiceAccountKeyFile => {
                Self::ServiceAccountKeyFile
            }
            settings::VertexAiAuthMethodContent::AccessToken => Self::AccessToken,
            settings::VertexAiAuthMethodContent::Automatic => Self::Automatic,
        }
    }
}

/// Reads ADC credentials and refreshes an access token from them.
async fn authenticate_via_adc(
    http_client: &dyn HttpClient,
    cx: &mut AsyncApp,
) -> Result<(VertexAiAuth, vertex_ai::AccessToken), AuthenticateError> {
    let credentials = cx
        .background_spawn(async { vertex_ai::read_adc_credentials() })
        .await
        .map_err(|_| AuthenticateError::CredentialsNotFound)?;
    let token = vertex_ai::refresh_access_token(http_client, &credentials)
        .await
        .map_err(AuthenticateError::Other)?;
    Ok((VertexAiAuth::Adc { credentials }, token))
}

/// Reads a service account key file and exchanges its JWT for an access token.
async fn authenticate_via_service_account_key_file(
    http_client: &dyn HttpClient,
    path: String,
    cx: &mut AsyncApp,
) -> Result<(VertexAiAuth, vertex_ai::AccessToken), AuthenticateError> {
    let credentials = cx
        .background_spawn(async move { vertex_ai::read_service_account_credentials(&path) })
        .await
        .map_err(|_| AuthenticateError::CredentialsNotFound)?;
    let token = vertex_ai::exchange_service_account_token(http_client, &credentials)
        .await
        .map_err(AuthenticateError::Other)?;
    Ok((VertexAiAuth::ServiceAccountKeyFile { credentials }, token))
}

/// Wraps a pre-obtained bearer token as `VertexAiAuth`. Since the token is
/// already in hand, this can't fail and doesn't need to be async.
fn access_token_auth(token_string: String) -> (VertexAiAuth, vertex_ai::AccessToken) {
    let access_token = vertex_ai::AccessToken {
        token: token_string.clone(),
        expires_at: None,
    };
    (
        VertexAiAuth::AccessToken {
            token: token_string,
        },
        access_token,
    )
}

pub struct VertexAiLanguageModelProvider {
    http_client: Arc<dyn HttpClient>,
    state: Entity<State>,
}

/// Thread-safe shared storage for the access token, so that both entity
/// updates (which need `&mut App`) and `stream_completion` (which only
/// has `&AsyncApp`) can read and write the current token.
type SharedAccessToken = Arc<StdMutex<Option<vertex_ai::AccessToken>>>;

pub struct State {
    shared_access_token: SharedAccessToken,
    auth: Option<VertexAiAuth>,
    project_id: String,
    location_id: String,
    authenticated: bool,
    credentials_from_env: bool,
    credentials_provider: Arc<dyn CredentialsProvider>,
    settings: Option<VertexAiSettings>,
    fetched_models: Vec<vertex_ai::Model>,
    fetch_models_task: Option<Task<()>>,
}

impl State {
    fn is_authenticated(&self) -> bool {
        self.authenticated
            && self
                .shared_access_token
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .is_some()
    }

    fn set_access_token(&self, token: Option<vertex_ai::AccessToken>) {
        *self
            .shared_access_token
            .lock()
            .unwrap_or_else(|e| e.into_inner()) = token;
    }
}

impl VertexAiLanguageModelProvider {
    pub fn new(
        http_client: Arc<dyn HttpClient>,
        credentials_provider: Arc<dyn CredentialsProvider>,
        cx: &mut App,
    ) -> Self {
        let settings = Self::settings(cx).clone();
        let project_id = if settings.project_id.is_empty() {
            vertex_ai::read_default_project().unwrap_or_default()
        } else {
            settings.project_id.clone()
        };
        let location_id = if settings.location_id.is_empty() {
            "global".to_string()
        } else {
            settings.location_id.clone()
        };

        let shared_access_token: SharedAccessToken = Arc::new(StdMutex::new(None));

        let state = cx.new(|cx| {
            cx.observe_global::<SettingsStore>(|this: &mut State, cx| {
                let settings = VertexAiLanguageModelProvider::settings(cx);
                if !settings.project_id.is_empty() {
                    this.project_id = settings.project_id.clone();
                }
                if !settings.location_id.is_empty() {
                    this.location_id = settings.location_id.clone();
                }
                this.settings = Some(settings.clone());
                cx.notify();
            })
            .detach();

            State {
                shared_access_token,
                auth: None,
                project_id,
                location_id,
                authenticated: false,
                credentials_from_env: false,
                credentials_provider,
                settings: Some(settings),
                fetched_models: Vec::new(),
                fetch_models_task: None,
            }
        });

        Self { http_client, state }
    }

    fn create_language_model(&self, model: vertex_ai::Model) -> Arc<dyn LanguageModel> {
        Arc::new(VertexAiLanguageModel {
            id: LanguageModelId::from(format!("vertex_ai/{}", model.id())),
            model,
            state: self.state.clone(),
            http_client: self.http_client.clone(),
            request_limiter: RateLimiter::new(4),
        })
    }

    fn settings(cx: &App) -> &VertexAiSettings {
        &crate::AllLanguageModelSettings::get_global(cx).vertex_ai
    }

    fn api_url(cx: &App) -> String {
        let api_url = &Self::settings(cx).api_url;
        if api_url.is_empty() {
            vertex_ai::DEFAULT_API_URL.to_string()
        } else {
            api_url.clone()
        }
    }
}

impl LanguageModelProviderState for VertexAiLanguageModelProvider {
    type ObservableEntity = State;

    fn observable_entity(&self) -> Option<Entity<Self::ObservableEntity>> {
        Some(self.state.clone())
    }
}

impl LanguageModelProvider for VertexAiLanguageModelProvider {
    fn id(&self) -> LanguageModelProviderId {
        PROVIDER_ID
    }

    fn name(&self) -> LanguageModelProviderName {
        PROVIDER_NAME
    }

    fn icon(&self) -> IconOrSvg {
        IconOrSvg::Icon(IconName::AiGoogle)
    }

    fn default_model(&self, _cx: &App) -> Option<Arc<dyn LanguageModel>> {
        Some(self.create_language_model(vertex_ai::Model::default()))
    }

    fn default_fast_model(&self, cx: &App) -> Option<Arc<dyn LanguageModel>> {
        let state = self.state.read(cx);
        if !state.fetched_models.is_empty() {
            let preferred = vertex_ai::FAST_MODEL_PREFERENCE;
            if let Some(model) = preferred
                .iter()
                .find_map(|id| state.fetched_models.iter().find(|m| m.id() == *id))
            {
                return Some(self.create_language_model(model.clone()));
            }
        }
        Some(self.create_language_model(vertex_ai::Model::default_fast()))
    }

    fn provided_models(&self, cx: &App) -> Vec<Arc<dyn LanguageModel>> {
        let state = self.state.read(cx);
        let mut models = BTreeMap::default();

        if state.fetched_models.is_empty() {
            // Before models are fetched, show the hardcoded defaults
            for model in vertex_ai::Model::iter() {
                if !matches!(model, vertex_ai::Model::Custom { .. }) {
                    models.insert(model.id().to_string(), model);
                }
            }
        } else {
            for model in &state.fetched_models {
                models.insert(model.id().to_string(), model.clone());
            }
        }

        // Settings-configured models are always included
        for model in &Self::settings(cx).available_models {
            models.insert(
                model.name.clone(),
                vertex_ai::Model::Custom {
                    name: model.name.clone(),
                    display_name: model.display_name.clone(),
                    max_tokens: model.max_tokens,
                    max_output_tokens: model.max_output_tokens,
                    publisher: model
                        .publisher
                        .clone()
                        .unwrap_or_else(|| "anthropic".to_string()),
                    supports_thinking: true,
                    // User-configured custom models default to the legacy
                    // thinking config since we don't know if they support
                    // `adaptive` — sending it to an unsupported model 400s.
                    supports_adaptive_thinking: false,
                },
            );
        }

        models
            .into_values()
            .map(|model| self.create_language_model(model))
            .collect()
    }

    fn is_authenticated(&self, cx: &App) -> bool {
        self.state.read(cx).is_authenticated()
    }

    fn authenticate(&self, cx: &mut App) -> Task<Result<(), AuthenticateError>> {
        let http_client = self.http_client.clone();
        let state = self.state.clone();
        let (settings, credentials_provider) = {
            let state_ref = self.state.read(cx);
            (state_ref.settings.clone(), state_ref.credentials_provider.clone())
        };

        cx.spawn(async move |cx| {
            let method = settings
                .as_ref()
                .and_then(|s| s.authentication_method.clone())
                .unwrap_or(VertexAiAuthMethod::Automatic);

            let (auth, access_token, from_env) = match method {
                VertexAiAuthMethod::Adc => {
                    let (auth, token) = authenticate_via_adc(http_client.as_ref(), cx).await?;
                    (auth, token, false)
                }
                VertexAiAuthMethod::ServiceAccountKeyFile => {
                    let path = settings
                        .as_ref()
                        .and_then(|s| s.service_account_key_file.clone())
                        .or_else(|| ZED_GOOGLE_APPLICATION_CREDENTIALS_VAR.value.clone())
                        .ok_or(AuthenticateError::CredentialsNotFound)?;
                    let from_env = settings
                        .as_ref()
                        .and_then(|s| s.service_account_key_file.as_ref())
                        .is_none();
                    let (auth, token) =
                        authenticate_via_service_account_key_file(http_client.as_ref(), path, cx)
                            .await?;
                    (auth, token, from_env)
                }
                VertexAiAuthMethod::AccessToken => {
                    let (token_string, from_env) =
                        if let Some(token) = ZED_VERTEX_ACCESS_TOKEN_VAR.value.clone() {
                            (token, true)
                        } else {
                            let (_username, bytes) = credentials_provider
                                .read_credentials(GOOGLE_VERTEX_AI_URL, cx)
                                .await
                                .map_err(AuthenticateError::Other)?
                                .ok_or(AuthenticateError::CredentialsNotFound)?;
                            let token = String::from_utf8(bytes)
                                .map_err(|error| AuthenticateError::Other(error.into()))?;
                            (token, false)
                        };
                    let (auth, token) = access_token_auth(token_string);
                    (auth, token, from_env)
                }
                VertexAiAuthMethod::Automatic => {
                    if let Some(token) = ZED_VERTEX_ACCESS_TOKEN_VAR.value.clone() {
                        let (auth, token) = access_token_auth(token);
                        (auth, token, true)
                    } else if let Some(path) =
                        ZED_GOOGLE_APPLICATION_CREDENTIALS_VAR.value.clone()
                    {
                        let (auth, token) =
                            authenticate_via_service_account_key_file(
                                http_client.as_ref(),
                                path,
                                cx,
                            )
                            .await?;
                        (auth, token, true)
                    } else {
                        let (auth, token) = authenticate_via_adc(http_client.as_ref(), cx).await?;
                        (auth, token, false)
                    }
                }
            };

            state.update(cx, |state, cx| {
                if state.project_id.is_empty() {
                    if let VertexAiAuth::ServiceAccountKeyFile { ref credentials } = auth {
                        if let Some(ref project_id) = credentials.project_id {
                            state.project_id = project_id.clone();
                        }
                    }
                }

                let quota_project_id = match &auth {
                    VertexAiAuth::Adc { credentials } => credentials.quota_project_id.clone(),
                    _ => None,
                };
                state.auth = Some(auth);
                state.credentials_from_env = from_env;
                let token_string = access_token.token.clone();
                state.set_access_token(Some(access_token));
                state.authenticated = true;

                let http_client = http_client.clone();
                let api_url = VertexAiLanguageModelProvider::api_url(cx);
                let project_id = state.project_id.clone();
                let location_id = state.location_id.clone();

                let task = cx.spawn(async move |this: gpui::WeakEntity<State>, cx| {
                    let models = cx
                        .background_spawn(async move {
                            vertex_ai::fetch_available_models(
                                http_client,
                                api_url,
                                token_string,
                                project_id,
                                location_id,
                                quota_project_id,
                            )
                            .await
                        })
                        .await;

                    this.update(cx, |state, cx| {
                        state.fetched_models = models;
                        state.fetch_models_task = None;
                        cx.notify();
                    })
                    .log_err();
                });
                state.fetch_models_task = Some(task);

                cx.notify();
            });

            Ok(())
        })
    }

    fn settings_view(&self, _cx: &mut App) -> Option<ProviderSettingsView> {
        let state = self.state.clone();
        Some(ProviderSettingsView::SubPage(SubPageProviderSettings::new(
            move |window, cx| {
                cx.new(|cx| ConfigurationView::new(state.clone(), window, cx))
                    .into()
            },
        )))
    }
}

pub struct VertexAiLanguageModel {
    id: LanguageModelId,
    model: vertex_ai::Model,
    state: Entity<State>,
    http_client: Arc<dyn HttpClient>,
    request_limiter: RateLimiter,
}

impl LanguageModel for VertexAiLanguageModel {
    fn id(&self) -> LanguageModelId {
        self.id.clone()
    }

    fn name(&self) -> LanguageModelName {
        LanguageModelName::from(self.model.display_name().to_string())
    }

    fn provider_id(&self) -> LanguageModelProviderId {
        PROVIDER_ID
    }

    fn provider_name(&self) -> LanguageModelProviderName {
        PROVIDER_NAME
    }

    fn supports_tools(&self) -> bool {
        self.model.supports_tools()
    }

    fn supports_images(&self) -> bool {
        self.model.supports_images()
    }

    fn supports_thinking(&self) -> bool {
        self.model.supports_thinking()
    }

    fn supported_effort_levels(&self) -> Vec<language_model::LanguageModelEffortLevel> {
        if !self.model.supports_thinking() {
            return Vec::new();
        }
        match self.model.publisher() {
            Publisher::Google => {
                [
                    google_ai::ThinkingLevel::Minimal,
                    google_ai::ThinkingLevel::Low,
                    google_ai::ThinkingLevel::Medium,
                    google_ai::ThinkingLevel::High,
                ]
                .iter()
                .map(|level| language_model::LanguageModelEffortLevel {
                    name: level.name().into(),
                    value: level.value().into(),
                    is_default: false,
                })
                .collect()
            }
            Publisher::Anthropic => {
                if !self.model.supports_adaptive_thinking() {
                    // Effort levels are only meaningful for `thinking: {type: "adaptive"}`.
                    // Models on the legacy `enabled`/`disabled` thinking config (e.g. Haiku
                    // 4.5) use a fixed budget instead, so offering effort levels for them
                    // would be misleading UI with no effect on the request.
                    return Vec::new();
                }
                [
                    anthropic::Effort::Low,
                    anthropic::Effort::Medium,
                    anthropic::Effort::High,
                    anthropic::Effort::Max,
                ]
                .iter()
                .map(|effort| {
                    let is_default = matches!(effort, anthropic::Effort::High);
                    let (name, value) = match effort {
                        anthropic::Effort::Low => ("Low", "low"),
                        anthropic::Effort::Medium => ("Medium", "medium"),
                        anthropic::Effort::High => ("High", "high"),
                        anthropic::Effort::Max => ("Max", "max"),
                        _ => unreachable!(),
                    };
                    language_model::LanguageModelEffortLevel {
                        name: name.into(),
                        value: value.into(),
                        is_default,
                    }
                })
                .collect()
            }
        }
    }

    fn supports_tool_choice(&self, choice: LanguageModelToolChoice) -> bool {
        match choice {
            LanguageModelToolChoice::Auto
            | LanguageModelToolChoice::Any
            | LanguageModelToolChoice::None => true,
        }
    }

    fn tool_input_format(&self) -> LanguageModelToolSchemaFormat {
        match self.model.publisher() {
            Publisher::Google => LanguageModelToolSchemaFormat::JsonSchemaSubset,
            Publisher::Anthropic => LanguageModelToolSchemaFormat::JsonSchema,
        }
    }

    fn telemetry_id(&self) -> String {
        format!("vertex_ai/{}", self.model.id())
    }

    fn max_token_count(&self) -> u64 {
        self.model.max_token_count()
    }

    fn max_output_tokens(&self) -> Option<u64> {
        self.model.max_output_tokens()
    }

    fn stream_completion(
        &self,
        request: LanguageModelRequest,
        cx: &AsyncApp,
    ) -> BoxFuture<
        'static,
        Result<
            futures::stream::BoxStream<
                'static,
                Result<LanguageModelCompletionEvent, LanguageModelCompletionError>,
            >,
            LanguageModelCompletionError,
        >,
    > {
        let http_client = self.http_client.clone();
        let model = self.model.clone();

        let (shared_access_token, auth, project_id, location_id, api_url) =
            self.state.read_with(cx, |state, cx| {
                let api_url = VertexAiLanguageModelProvider::api_url(cx);
                (
                    state.shared_access_token.clone(),
                    state.auth.clone(),
                    state.project_id.clone(),
                    state.location_id.clone(),
                    api_url,
                )
            });
        let quota_project_id = match &auth {
            Some(VertexAiAuth::Adc { credentials }) => credentials.quota_project_id.clone(),
            _ => None,
        };

        let future = self.request_limiter.stream(async move {
            let current_token = shared_access_token
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .clone();

            let access_token = if current_token
                .as_ref()
                .map(|t| t.is_expired())
                .unwrap_or(true)
            {
                let refreshed = match &auth {
                    Some(VertexAiAuth::Adc { credentials }) => {
                        vertex_ai::refresh_access_token(http_client.as_ref(), credentials)
                            .await
                            .ok()
                    }
                    Some(VertexAiAuth::ServiceAccountKeyFile { credentials }) => {
                        vertex_ai::exchange_service_account_token(
                            http_client.as_ref(),
                            credentials,
                        )
                        .await
                        .ok()
                    }
                    Some(VertexAiAuth::AccessToken { .. }) | None => None,
                };
                if let Some(new_token) = refreshed {
                    log::info!("Vertex AI: refreshed expired access token");
                    *shared_access_token
                        .lock()
                        .unwrap_or_else(|e| e.into_inner()) = Some(new_token.clone());
                    Some(new_token)
                } else {
                    current_token
                }
            } else {
                current_token
            };

            let token_string = access_token
                .as_ref()
                .map(|t| t.token.clone())
                .ok_or_else(|| LanguageModelCompletionError::NoApiKey {
                    provider: PROVIDER_NAME,
                })?;

            let stream: futures::stream::BoxStream<
                'static,
                Result<LanguageModelCompletionEvent, LanguageModelCompletionError>,
            > = match model.publisher() {
                Publisher::Google => {
                    let google_request = into_google(
                        request,
                        model.vertex_model_id().to_string(),
                        GoogleModelMode::Thinking {
                            budget_tokens: None,
                        },
                    )
                    .map_err(LanguageModelCompletionError::from)?;
                    let response = vertex_ai::stream_generate_content(
                        http_client.as_ref(),
                        &api_url,
                        &token_string,
                        &project_id,
                        &location_id,
                        model.vertex_model_id(),
                        quota_project_id.as_deref(),
                        google_request,
                    )
                    .await
                    .map_err(LanguageModelCompletionError::from)?;
                    GoogleEventMapper::new().map_stream(response).boxed()
                }
                Publisher::Anthropic => {
                    let max_output = model.max_output_tokens().unwrap_or(64_000);
                    let mode = if model.supports_adaptive_thinking() {
                        anthropic::AnthropicModelMode::AdaptiveThinking
                    } else if model.supports_thinking() {
                        anthropic::AnthropicModelMode::Thinking {
                            budget_tokens: Some(4_096),
                        }
                    } else {
                        anthropic::AnthropicModelMode::Default
                    };
                    let anthropic_request = into_anthropic(
                        request,
                        model.vertex_model_id().to_string(),
                        1.0,
                        max_output,
                        mode,
                        AnthropicPromptCacheMode::Automatic,
                        &PROVIDER_ID,
                    )
                    .map_err(LanguageModelCompletionError::from)?;
                    let response = vertex_ai::stream_raw_predict(
                        http_client.as_ref(),
                        &api_url,
                        &token_string,
                        &project_id,
                        &location_id,
                        model.vertex_model_id(),
                        quota_project_id.as_deref(),
                        anthropic_request,
                    )
                    .await
                    .map_err(LanguageModelCompletionError::from)?;
                    AnthropicEventMapper::new(PROVIDER_NAME, PROVIDER_ID)
                        .map_stream(response)
                        .boxed()
                }
            };
            Ok(stream)
        });
        async move { Ok(future.await?.boxed()) }.boxed()
    }
}

struct ConfigurationView {
    state: Entity<State>,
    load_credentials_task: Option<Task<()>>,
}

impl ConfigurationView {
    fn new(state: Entity<State>, window: &mut Window, cx: &mut Context<Self>) -> Self {
        cx.observe(&state, |_, _, cx| {
            cx.notify();
        })
        .detach();

        let load_credentials_task = Some(cx.spawn_in(window, {
            let state = state.clone();
            async move |this, cx| {
                let has_adc = cx
                    .background_spawn(async { vertex_ai::read_adc_credentials().is_ok() })
                    .await;

                if has_adc {
                    state.update(cx, |_state, cx| {
                        cx.notify();
                    });
                }
                this.update(cx, |this, cx| {
                    this.load_credentials_task = None;
                    cx.notify();
                })
                .log_err();
            }
        }));

        Self {
            state,
            load_credentials_task,
        }
    }
}

impl Render for ConfigurationView {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let state = self.state.read(cx);

        if self.load_credentials_task.is_some() {
            return div()
                .child(Label::new("Loading credentials..."))
                .into_any_element();
        }

        if state.is_authenticated() {
            let project_id = &state.project_id;
            let location_id = &state.location_id;
            let auth_label = match &state.auth {
                Some(VertexAiAuth::Adc { .. }) => {
                    format!(
                        "Authenticated via gcloud ADC (project: {project_id}, location: {location_id})"
                    )
                }
                Some(VertexAiAuth::ServiceAccountKeyFile { credentials }) => {
                    format!(
                        "Authenticated via service account: {} (project: {project_id}, location: {location_id})",
                        credentials.client_email
                    )
                }
                Some(VertexAiAuth::AccessToken { .. }) if state.credentials_from_env => {
                    format!(
                        "Authenticated via {} env var (project: {project_id}, location: {location_id})",
                        ZED_VERTEX_ACCESS_TOKEN_VAR.name
                    )
                }
                Some(VertexAiAuth::AccessToken { .. }) => {
                    format!(
                        "Authenticated via access token (project: {project_id}, location: {location_id})"
                    )
                }
                None => "Authenticated".into(),
            };
            return ConfiguredApiCard::new("vertex-ai-configured", auth_label).into_any_element();
        }

        v_flex()
            .size_full()
            .child(Label::new(
                "To use Vertex AI, configure authentication using one of these methods:",
            ))
            .child(
                List::new()
                    .child(ListBulletItem::new(
                        "ADC: Run gcloud auth application-default login",
                    ))
                    .child(ListBulletItem::new(
                        "Service Account: Set service_account_key_file in settings or ZED_GOOGLE_APPLICATION_CREDENTIALS env var",
                    ))
                    .child(ListBulletItem::new(
                        "Access Token: Set ZED_VERTEX_ACCESS_TOKEN env var",
                    )),
            )
            .child(
                Label::new(
                    "Configure project_id, location_id, and authentication_method in settings under language_models.vertex_ai",
                )
                .size(LabelSize::Small)
                .color(Color::Muted),
            )
            .into_any_element()
    }
}
