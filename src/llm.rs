use anyhow::Result;
use bevy::utils::BoxedFuture;
use derivative::Derivative;
use serde::{Deserialize, Serialize};

use crate::game::{
    Actor, Card, ChatRecord, DummyActor, PlayerData, PublicState, Role, Stake, StakeState,
    TradeState,
};

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SamplerKind {
    #[default]
    Nucleus,
    Typical,
}

#[derive(Debug, Derivative, Clone, Serialize, Deserialize)]
#[derivative(Default)]
pub struct Sampler {
    #[serde(rename = "type")]
    pub kind: SamplerKind,
    #[derivative(Default(value = "0.5"))]
    pub top_p: f32,
    #[derivative(Default(value = "128"))]
    pub top_k: u32,
    #[derivative(Default(value = "0.5"))]
    pub tau: f32,
    #[derivative(Default(value = "1.0"))]
    pub temperature: f32,
    #[derivative(Default(value = "0.3"))]
    pub presence_penalty: f32,
    #[derivative(Default(value = "0.3"))]
    pub frequency_penalty: f32,
    #[derivative(Default(value = "0.996"))]
    pub penalty_decay: f32,
}

#[derive(Debug, Derivative, Clone, Serialize, Deserialize)]
#[derivative(Default)]
pub struct CompletionRequest {
    pub prompt: String,
    pub state: uuid::Uuid,
    pub stop: Vec<String>,
    pub stream: bool,
    #[serde(skip_serializing_if = "String::is_empty")]
    pub bnf_schema: String,
    #[derivative(Default(value = "1024"))]
    pub max_tokens: u32,
    #[serde(rename = "sampler_override")]
    pub sampler: Sampler,
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct CompletionResponse {
    pub choices: Vec<Choice>,
    pub model: String,
}

impl CompletionResponse {
    pub fn model_text(&self) -> String {
        self.choices
            .first()
            .map(|choice| choice.text.as_str())
            .unwrap_or("")
            .trim()
            .to_owned()
    }
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct Choice {
    pub index: u32,
    pub text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmRecord {
    pub request: CompletionRequest,
    pub response: CompletionResponse,
}

#[derive(Debug, Default, Clone)]
pub struct LlmActor {
    pub state: uuid::Uuid,
    pub llm: Vec<LlmRecord>,
    pub chat: Vec<ChatRecord>,
    pub dummy: DummyActor,
}

impl LlmActor {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn assemble_prompt(&self) -> String {
        let mut text = String::new();
        let mut last = Role::default();
        for record in &self.chat {
            let content = record.content.replace("\r\n", "\n");
            let content = content.trim();

            text = match &record.role {
                x if x == &last => format!("{text}\n{content}"),
                Role::System(_) => format!("{text}\n\n{} (AI): {content}", record.role),
                x => format!("{text}\n\n{x}: {content}"),
            };
            last = record.role.clone();
        }

        text
    }

    pub fn assemble_prompt_role(&self, role: &Role) -> String {
        format!("{}\n\n{role}:", self.assemble_prompt())
    }

    pub async fn call_llm(&self, request: &CompletionRequest) -> Result<CompletionResponse> {
        let response = ehttp::fetch_async(ehttp::Request::json(
            "http://localhost:65530/api/oai/completions",
            request,
        )?)
        .await
        .map_err(|err| anyhow::anyhow!(err))?;
        Ok(response.json()?)
    }

    pub async fn notify<'a>(&'a mut self, data: &'a PlayerData, state: &'a PublicState) {
        if self.chat.is_empty() {
            self.chat.extend([
                ChatRecord::new(
                    Role::actor(data.entity, "Boss"),
                    include_str!("prompts/op_boss.txt"),
                ),
                ChatRecord::new(
                    Role::system(data.entity),
                    format!(include_str!("prompts/op_system.txt"), data.name.to_string()),
                ),
                ChatRecord::new(
                    Role::actor(data.entity, data.name.clone()),
                    include_str!("prompts/op_me.txt"),
                ),
            ]);
        }

        self.chat.push(ChatRecord::new(
            Role::system(data.entity),
            format!(
                include_str!("prompts/notify.txt"),
                state.total_cards(),
                serde_json::to_string_pretty(&state).expect("failed to serialize public state"),
                serde_json::to_string_pretty(&data.inventory)
                    .expect("failed to serialize inventory")
            ),
        ));

        self.chat.push(loop {
            let role = Role::actor(data.entity, data.name.clone());
            let prompt = self.assemble_prompt_role(&role);
            let request = CompletionRequest {
                prompt,
                state: self.state,
                stop: vec!["\n\n".into()],
                sampler: Sampler {
                    kind: SamplerKind::Typical,
                    ..Default::default()
                },
                ..Default::default()
            };
            let response = match self.call_llm(&request).await {
                Ok(response) => response,
                Err(err) => {
                    bevy::log::error!("{err}");
                    continue;
                }
            };

            let record = ChatRecord::new(role, response.model_text());
            bevy::log::info!("{record}");

            self.llm.push(LlmRecord { request, response });
            break record;
        });
    }
}

impl Actor for LlmActor {
    fn notify<'a>(
        &'a mut self,
        data: &'a PlayerData,
        state: &'a PublicState,
    ) -> BoxedFuture<'a, ()> {
        Box::pin(self.notify(data, state))
    }

    fn feedback<'a>(&'a mut self, data: &'a PlayerData, text: String) -> BoxedFuture<'a, ()> {
        self.dummy.feedback(data, text)
    }

    fn chat<'a>(
        &'a mut self,
        data: &'a PlayerData,
        history: &'a [ChatRecord],
        kind: crate::game::ChatKind,
    ) -> BoxedFuture<'a, Vec<ChatRecord>> {
        self.dummy.chat(data, history, kind)
    }

    fn trade<'a>(
        &'a mut self,
        data: &'a PlayerData,
        history: &'a [ChatRecord],
    ) -> BoxedFuture<'a, crate::game::Trade> {
        self.dummy.trade(data, history)
    }

    fn accept_trade<'a>(
        &'a mut self,
        data: &'a PlayerData,
        history: &'a [ChatRecord],
        state: TradeState<'a>,
    ) -> BoxedFuture<'a, bool> {
        self.dummy.accept_trade(data, history, state)
    }

    fn bet<'a>(
        &'a mut self,
        data: &'a PlayerData,
        history: &'a [ChatRecord],
    ) -> BoxedFuture<'a, Stake> {
        self.dummy.bet(data, history)
    }

    fn accept_duel<'a>(
        &'a mut self,
        data: &'a PlayerData,
        history: &'a [ChatRecord],
        state: StakeState<'a>,
    ) -> BoxedFuture<'a, Option<Card>> {
        self.dummy.accept_duel(data, history, state)
    }
}
