use anyhow::Result;
use collections::BTreeMap;
use futures::{FutureExt, StreamExt, future::BoxFuture};
use google_ai::GoogleModelMode;
use gpui::{AnyView, App, AsyncApp, Context, Entity, Task, Window};
use http_client::HttpClient;
use language_model::{
    AuthenticateError, ConfigurationViewTargetAgent, LanguageModelCompletionError,
    LanguageModelCompletionEvent, LanguageModelToolChoice, LanguageModelToolSchemaFormat,
};
use language_model::{
    IconOrSvg, LanguageModel, LanguageModelId, LanguageModelName, LanguageModelProvider,
    LanguageModelProviderId, LanguageModelProviderName, LanguageModelProviderState,
    LanguageModelRequest, RateLimiter,
};
pub use settings::VertexAiAvailableModel as AvailableModel;
use settings::{Settings, SettingsStore};
use std::pin::Pin;
use std::sync::Arc;
use strum::IntoEnumIterator;
use ui::{ConfiguredApiCard, List, ListBulletItem, prelude::*};
use util::ResultExt;
use vertex_ai::Publisher;

use crate::provider::anthropic::{AnthropicEventMapper, into_anthropic};
use crate::provider::google::{GoogleEventMapper, into_google};

const PROVIDER_ID: LanguageModelProviderId = LanguageModelProviderId::new("vertex_ai");
const PROVIDER_NAME: LanguageModelProviderName = LanguageModelProviderName::new("Vertex AI");

#[derive(Default, Clone, Debug, PartialEq)]
pub struct VertexAiSettings {
    pub api_url: String,
    pub project_id: String,
    pub location_id: String,
    pub available_models: Vec<AvailableModel>,
}

pub struct VertexAiLanguageModelProvider {
    http_client: Arc<dyn HttpClient>,
    state: Entity<State>,
}

pub struct State {
    access_token: Option<vertex_ai::AccessToken>,
    credentials: Option<vertex_ai::AdcCredentials>,
    project_id: String,
    location_id: String,
    authenticated: bool,
    fetched_models: Vec<vertex_ai::Model>,
    fetch_models_task: Option<Task<()>>,
}

impl State {
    fn is_authenticated(&self) -> bool {
        self.authenticated && self.access_token.is_some()
    }

}

impl VertexAiLanguageModelProvider {
    pub fn new(http_client: Arc<dyn HttpClient>, cx: &mut App) -> Self {
        let settings = Self::settings(cx);
        let project_id = if settings.project_id.is_empty() {
            vertex_ai::read_default_project().unwrap_or_default()
        } else {
            settings.project_id.clone()
        };
        let location_id = if settings.location_id.is_empty() {
            "us-east5".to_string()
        } else {
            settings.location_id.clone()
        };

        let state = cx.new(|cx| {
            cx.observe_global::<SettingsStore>(|this: &mut State, cx| {
                let settings = VertexAiLanguageModelProvider::settings(cx);
                if !settings.project_id.is_empty() {
                    this.project_id = settings.project_id.clone();
                }
                if !settings.location_id.is_empty() {
                    this.location_id = settings.location_id.clone();
                }
                cx.notify();
            })
            .detach();

            State {
                access_token: None,
                credentials: None,
                project_id,
                location_id,
                authenticated: false,
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

    fn default_fast_model(&self, _cx: &App) -> Option<Arc<dyn LanguageModel>> {
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

        cx.spawn(async move |cx| {
            let credentials = cx
                .background_spawn(async { vertex_ai::read_adc_credentials() })
                .await
                .map_err(|_| AuthenticateError::CredentialsNotFound)?;

            let access_token =
                vertex_ai::refresh_access_token(http_client.as_ref(), &credentials)
                    .await
                    .map_err(|error| AuthenticateError::Other(error.into()))?;

            state.update(cx, |state, cx| {
                state.credentials = Some(credentials);
                let token_string = access_token.token.clone();
                state.access_token = Some(access_token);
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

    fn configuration_view(
        &self,
        _target_agent: ConfigurationViewTargetAgent,
        window: &mut Window,
        cx: &mut App,
    ) -> AnyView {
        cx.new(|cx| ConfigurationView::new(self.state.clone(), window, cx))
            .into()
    }

    fn reset_credentials(&self, cx: &mut App) -> Task<Result<()>> {
        self.state.update(cx, |state, cx| {
            state.access_token = None;
            state.credentials = None;
            state.authenticated = false;
            state.fetched_models.clear();
            state.fetch_models_task = None;
            cx.notify();
            Task::ready(Ok(()))
        })
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

    fn count_tokens(
        &self,
        request: LanguageModelRequest,
        cx: &App,
    ) -> BoxFuture<'static, Result<u64>> {
        crate::provider::google::count_google_tokens(request, cx)
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

        let state_data = self.state.read_with(cx, |state, cx| {
            let api_url = VertexAiLanguageModelProvider::api_url(cx);
            (
                state.access_token.clone(),
                state.credentials.clone(),
                state.project_id.clone(),
                state.location_id.clone(),
                api_url,
            )
        });

        let future = self.request_limiter.stream(async move {
            let (mut access_token, credentials, project_id, location_id, api_url) = state_data;

            // Auto-refresh token if expired
            if access_token.as_ref().map(|t| t.is_expired()).unwrap_or(true) {
                if let Some(credentials) = &credentials {
                    match vertex_ai::refresh_access_token(http_client.as_ref(), credentials).await {
                        Ok(new_token) => {
                            log::info!("Vertex AI: refreshed expired access token");
                            access_token = Some(new_token);
                        }
                        Err(error) => {
                            log::error!("Vertex AI: failed to refresh token: {error}");
                        }
                    }
                }
            }

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
                        GoogleModelMode::Thinking { budget_tokens: None },
                    );
                    let response = vertex_ai::stream_generate_content(
                        http_client.as_ref(),
                        &api_url,
                        &token_string,
                        &project_id,
                        &location_id,
                        model.vertex_model_id(),
                        google_request,
                    )
                    .await
                    .map_err(LanguageModelCompletionError::from)?;
                    GoogleEventMapper::new().map_stream(response).boxed()
                }
                Publisher::Anthropic => {
                    let max_output = model.max_output_tokens().unwrap_or(64_000);
                    let mode = if model.supports_thinking() {
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
                    );
                    let response = vertex_ai::stream_raw_predict(
                        http_client.as_ref(),
                        &api_url,
                        &token_string,
                        &project_id,
                        &location_id,
                        model.vertex_model_id(),
                        anthropic_request,
                    )
                    .await
                    .map_err(LanguageModelCompletionError::from)?;
                    AnthropicEventMapper::new()
                        .map_stream(Pin::from(response))
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
                let result = cx
                    .background_spawn(async { vertex_ai::read_adc_credentials() })
                    .await;

                if let Ok(credentials) = result {
                    state.update(cx, |state, cx| {
                        state.credentials = Some(credentials);
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
                .child(Label::new("Loading gcloud credentials..."))
                .into_any_element();
        }

        if state.is_authenticated() {
            let project_id = state.project_id.clone();
            let location_id = state.location_id.clone();
            return ConfiguredApiCard::new(format!(
                "Authenticated via gcloud ADC (project: {project_id}, location: {location_id})"
            ))
            .into_any_element();
        }

        v_flex()
            .size_full()
            .child(Label::new(
                "To use Vertex AI, you need Google Cloud Application Default Credentials configured.",
            ))
            .child(
                List::new()
                    .child(ListBulletItem::new(
                        "Run: gcloud auth application-default login",
                    ))
                    .child(ListBulletItem::new(
                        "Configure project_id and location_id in settings under language_models.vertex_ai",
                    )),
            )
            .child(
                Label::new(
                    "Credentials are read from ~/.config/gcloud/application_default_credentials.json",
                )
                .size(LabelSize::Small)
                .color(Color::Muted),
            )
            .into_any_element()
    }
}
