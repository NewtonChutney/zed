---
title: Use a Gateway - Zed
description: Configure OpenRouter, Vercel AI Gateway, Amazon Bedrock, Google Vertex AI, and other gateway or cloud model platforms in Zed.
---

# Use a Gateway

Use a gateway when you route model requests through a platform such as OpenRouter, Vercel AI Gateway, Amazon Bedrock, Google Vertex AI, or another OpenAI-compatible service.

| Gateway                   | Zed AI features | External Agents | Terminal Threads | Notes                                        |
| ------------------------- | --------------- | --------------- | ---------------- | -------------------------------------------- |
| OpenRouter                | Yes             | Separate config | Separate config  | Uses OpenRouter API access                   |
| Vercel AI Gateway         | Yes             | Separate config | Separate config  | Uses Vercel AI Gateway API access            |
| Amazon Bedrock            | Yes             | Separate config | Separate config  | Uses AWS credentials or Bedrock bearer token |
| Google Vertex AI          | Yes             | Separate config | Separate config  | Uses Google Cloud ADC credentials            |
| OpenAI-compatible gateway | Yes             | Separate config | Separate config  | Configure base URL, model, and key           |

## OpenRouter {#openrouter}

Use OpenRouter when you want to route Zed AI features through OpenRouter.

1. Visit [OpenRouter](https://openrouter.ai) and create an account.
2. Generate an API key from your [OpenRouter keys page](https://openrouter.ai/keys).
3. Open **Settings → AI → LLM Providers** with {#action agent::OpenSettings} and find the OpenRouter row.
4. Enter your OpenRouter API key.

Zed also reads `OPENROUTER_API_KEY` from the local Zed process environment.

When using OpenRouter as your assistant provider, explicitly select a model in your settings:

```json [settings]
{
  "agent": {
    "default_model": {
      "provider": "openrouter",
      "model": "openrouter/auto"
    }
  }
}
```

The `openrouter/auto` model routes requests to an available model selected by OpenRouter. You can also specify any model available through OpenRouter's API.

### OpenRouter Custom Models {#openrouter-custom-models}

You can add custom models to the OpenRouter provider in settings:

```json [settings]
{
  "language_models": {
    "open_router": {
      "api_url": "https://openrouter.ai/api/v1",
      "available_models": [
        {
          "name": "google/gemini-2.0-flash-thinking-exp",
          "display_name": "Gemini 2.0 Flash (Thinking)",
          "max_tokens": 200000,
          "max_output_tokens": 8192,
          "supports_tools": true,
          "supports_images": true,
          "mode": {
            "type": "thinking",
            "budget_tokens": 8000
          }
        }
      ]
    }
  }
}
```

Custom model entries support fields such as `name`, `display_name`, `max_tokens`, `max_output_tokens`, `max_completion_tokens`, `supports_tools`, `supports_images`, and `mode`.

### OpenRouter Provider Routing {#openrouter-provider-routing}

You can control how OpenRouter routes a custom model request among upstream providers with the `provider` object on each model entry.

Supported fields include `order`, `allow_fallbacks`, `require_parameters`, `data_collection`, `only`, `ignore`, `quantizations`, and `sort`.

```json [settings]
{
  "language_models": {
    "open_router": {
      "available_models": [
        {
          "name": "openrouter/auto",
          "display_name": "Auto Router",
          "max_tokens": 2000000,
          "supports_tools": true,
          "provider": {
            "order": ["anthropic", "openai"],
            "allow_fallbacks": true,
            "require_parameters": true,
            "data_collection": "allow"
          }
        }
      ]
    }
  }
}
```

## Vercel AI Gateway {#vercel-ai-gateway}

Use Vercel AI Gateway when you want to route Zed AI features through Vercel.

1. Create an API key from your Vercel AI Gateway keys page.
2. Open **Settings → AI → LLM Providers** with {#action agent::OpenSettings} and find the Vercel AI Gateway row.
3. Enter your Vercel AI Gateway API key.

Zed also reads `VERCEL_AI_GATEWAY_API_KEY` from the local Zed process environment.

You can set a custom endpoint for Vercel AI Gateway in settings:

```json [settings]
{
  "language_models": {
    "vercel_ai_gateway": {
      "api_url": "https://ai-gateway.vercel.sh/v1"
    }
  }
}
```

## Amazon Bedrock {#amazon-bedrock}

Use Amazon Bedrock when you want model access through AWS.

Bedrock supports models that support streaming tool use. See [Amazon Bedrock's Tool Use documentation](https://docs.aws.amazon.com/bedrock/latest/userguide/conversation-inference-supported-models-features.html).

Your AWS credentials need these permissions:

- `bedrock:InvokeModelWithResponseStream`
- `bedrock:InvokeModel`

Bedrock supports Zed-prefixed AWS environment variables so Zed does not override or consume your normal AWS credentials:

- `ZED_ACCESS_KEY_ID`
- `ZED_SECRET_ACCESS_KEY`
- `ZED_SESSION_TOKEN`
- `ZED_AWS_PROFILE`
- `ZED_AWS_REGION`
- `ZED_AWS_ENDPOINT`
- `ZED_BEDROCK_BEARER_TOKEN`

### Bedrock Authentication {#bedrock-authentication}

You can authenticate with a named profile, static credentials, or a Bedrock API key.

For a named profile, configure Bedrock in settings:

```json [settings]
{
  "language_models": {
    "bedrock": {
      "authentication_method": "named_profile",
      "region": "your-aws-region",
      "profile": "your-profile-name"
    }
  }
}
```

For static credentials, open Agent Settings with {#action agent::OpenSettings}, go to the Amazon Bedrock section, and enter the access key ID, secret access key, and region.

For a Bedrock API key, choose API key authentication:

```json [settings]
{
  "language_models": {
    "bedrock": {
      "authentication_method": "api_key",
      "region": "your-aws-region"
    }
  }
}
```

The API key itself is stored in the system keychain, not in `settings.json`.

### Bedrock Cross-Region Inference {#bedrock-cross-region-inference}

Zed uses [Cross-Region inference](https://docs.aws.amazon.com/bedrock/latest/userguide/cross-region-inference.html) for Bedrock on a best-effort basis.

By default, Zed uses regional inference profiles. To opt into global profiles, add `allow_global`:

```json [settings]
{
  "language_models": {
    "bedrock": {
      "authentication_method": "named_profile",
      "region": "your-aws-region",
      "profile": "your-profile-name",
      "allow_global": true
    }
  }
}
```

Only some models support global inference profiles. See the AWS Bedrock supported models documentation for the current list.

### Bedrock Guardrails {#bedrock-guardrails}

Some AWS environments require a guardrail on every Bedrock API call. Add `guardrail_identifier` to apply a guardrail to all Bedrock requests:

```json [settings]
{
  "language_models": {
    "bedrock": {
      "guardrail_identifier": "arn:aws:bedrock:us-east-1:123456789012:guardrail/abc123",
      "guardrail_version": "DRAFT"
    }
  }
}
```

### Bedrock Mantle Models {#bedrock-mantle-models}

Some models, such as the GPT-5.6 family (Sol, Terra, and Luna), GPT-5.5, GPT-5.4, and Grok 4.3, aren't available through Bedrock's Converse API and are only reachable through `bedrock-mantle`, AWS's OpenAI-compatible inference endpoint. Zed routes these models through `bedrock-mantle` automatically; they appear alongside the rest of the Bedrock models in the model picker once you're authenticated, with no extra configuration required.

Mantle models require IAM permissions for the `bedrock-mantle` endpoint (for example via the `AmazonBedrockMantleInferenceAccess` managed policy) in addition to whatever permissions your existing Bedrock credentials already have, and `bedrock-mantle` is only available in [some AWS Regions](https://docs.aws.amazon.com/bedrock/latest/userguide/bedrock-mantle.html#regions). Zed surfaces an error naming the current Region and the supported ones if you try to use a Mantle model outside of them.

#### Custom Bedrock Mantle Models {#bedrock-mantle-custom-models}

You can add custom models served through `bedrock-mantle` with `mantle_available_models`:

```json [settings]
{
  "language_models": {
    "bedrock": {
      "mantle_available_models": [
        {
          "name": "openai.gpt-oss-120b",
          "display_name": "GPT-OSS 120B",
          "max_tokens": 128000,
          "protocol": "chat_completions",
          "supports_tools": true,
          "supports_images": false,
          "supports_thinking": true
        }
      ]
    }
  }
}
```

`protocol` selects which OpenAI-compatible API the model is called through, and must be either `chat_completions` or `responses`. Set `supports_thinking` to `true` for custom Mantle models that accept OpenAI reasoning effort parameters; Zed will then expose `low`, `medium`, `high`, and `xhigh` in the thinking effort picker, while disabling thinking sends `none`.

## Google Vertex AI {#vertex-ai}

Use [Google Vertex AI](https://cloud.google.com/vertex-ai) when you want access to Gemini and Claude models through Google Cloud's managed service.

### Vertex AI Authentication {#vertex-ai-authentication}

Vertex AI uses Google Cloud's Application Default Credentials (ADC) for authentication:

1. [Install the Google Cloud CLI](https://cloud.google.com/sdk/docs/install)
2. Run `gcloud auth application-default login` to authenticate
3. Zed will automatically use your credentials

You can also authenticate by setting the `GOOGLE_APPLICATION_CREDENTIALS` environment variable to point to a service account JSON key file.

### Vertex AI Configuration {#vertex-ai-configuration}

Add the following to your Zed settings file to configure Vertex AI ([how to edit](../configuring-zed.md#settings-files)):

```json [settings]
{
  "language_models": {
    "vertex_ai": {
      "project_id": "your-google-cloud-project-id",
      "location_id": "us-central1"
    }
  }
}
```

- `project_id` — Your Google Cloud project ID (required). Find it in the [Google Cloud Console](https://console.cloud.google.com/project).
- `location_id` — The Google Cloud region (defaults to `us-east5`). Common regions: `us-central1`, `us-east1`, `us-east5`, `europe-west1`, `asia-northeast1`.
- `api_url` — Custom API base URL. Leave unset to use Google's managed endpoint.

### Vertex AI Custom Models {#vertex-ai-custom-models}

Vertex AI auto-discovers available models from your project's org policy. To add models manually or customize parameters:

```json [settings]
{
  "language_models": {
    "vertex_ai": {
      "project_id": "your-google-cloud-project-id",
      "location_id": "us-central1",
      "available_models": [
        {
          "name": "gemini-2.0-pro-exp-02-05",
          "display_name": "Gemini 2.0 Pro",
          "max_tokens": 1000000,
          "max_output_tokens": 8192,
          "publisher": "google"
        },
        {
          "name": "claude-3-5-sonnet@20241022",
          "display_name": "Claude 3.5 Sonnet",
          "max_tokens": 200000,
          "max_output_tokens": 4096,
          "publisher": "anthropic"
        }
      ]
    }
  }
}
```

- `name` — The model ID as it appears in Google Cloud (required)
- `display_name` — A human-readable name (optional)
- `max_tokens` — Maximum input tokens (required)
- `max_output_tokens` — Maximum output tokens (optional)
- `publisher` — `"google"` for Gemini models or `"anthropic"` for Claude models (optional, defaults to `"anthropic"`)

## OpenAI-Compatible Gateways {#openai-compatible}

If your gateway exposes an OpenAI-compatible API, configure it with [Use API Access](./use-api-access.md#openai-compatible).
