use std::sync::Arc;

use anyhow::Result;
use async_std::sync::Mutex;
use bevy::utils::BoxedFuture;
use derivative::Derivative;
use serde::{Deserialize, Serialize};

use crate::game::{
    Actor, Card, ChatKind, ChatRecord, DummyActor, OpponentData, PlayerData, PublicState, Role,
    Stake, StakeState, TradeState, NUM_CHAT_ROUNDS, SYSTEM_NAME,
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
    pub llm: Arc<Mutex<Vec<LlmRecord>>>,
    pub chat: Vec<ChatRecord>,
    pub dummy: DummyActor,
}

impl LlmActor {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn prompt(&self) -> String {
        let mut text = "Below is a log from the game **Ark of Destinies**.".into();
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

    pub fn prompt_role(&self, role: &Role) -> String {
        format!("{}\n\n{role}:", self.prompt())
    }

    pub fn prompt_prefix(&self, prefix: impl AsRef<str>) -> String {
        let prefix = prefix.as_ref();
        format!("{}{prefix}", self.prompt())
    }

    pub async fn call_llm(&self, request: &CompletionRequest) -> Result<CompletionResponse> {
        async_std::task::yield_now().await;
        let response = ehttp::fetch_async(ehttp::Request::json(
            "http://localhost:65530/api/oai/completions",
            request,
        )?)
        .await
        .map_err(|err| anyhow::anyhow!(err))?;
        Ok(response.json()?)
    }

    pub async fn chat_llm<T: AsRef<str>>(
        &self,
        role: Role,
        prompt: impl AsRef<str>,
        prefix: impl AsRef<str>,
        names: &[T],
        sampler: Sampler,
    ) -> ChatRecord {
        loop {
            let prompt = prompt.as_ref();
            let prefix = prefix.as_ref();
            let sampler = sampler.clone();

            let mut stop = vec![
                "\n\n".into(),
                format!("\n{}", SYSTEM_NAME),
                "\nUser:".into(),
                "\nQ:".into(),
                "\nAssistant:".into(),
                "\nAI:".into(),
                "\n[".into(),
            ];
            stop.extend(
                names
                    .iter()
                    .map(|name| format!("\n{}", name.as_ref().trim())),
            );

            let request = CompletionRequest {
                prompt: format!("{prompt}{prefix}"),
                state: self.state,
                stop,
                sampler,
                ..Default::default()
            };
            let response = match self.call_llm(&request).await {
                Ok(response) => response,
                Err(err) => {
                    bevy::log::error!("{err}");
                    continue;
                }
            };

            let content = format!("{prefix}{}", response.model_text());
            let record = ChatRecord::new(role, content);
            bevy::log::info!("{record}");

            self.llm.lock().await.push(LlmRecord { request, response });
            break record;
        }
    }

    pub async fn notify<'a>(&'a mut self, player: &'a PlayerData, state: &'a PublicState) {
        if self.chat.is_empty() {
            self.chat.extend([
                ChatRecord::new(
                    Role::actor(player.entity, "Boss"),
                    include_str!("prompts/op_boss.md"),
                ),
                ChatRecord::new(
                    Role::system(player.entity),
                    format!(
                        include_str!("prompts/op_system.md"),
                        player.name.to_string()
                    ),
                ),
                ChatRecord::new(
                    Role::actor(player.entity, player.name.clone()),
                    include_str!("prompts/op_me.md"),
                ),
            ]);
        }

        // system reports current state
        self.chat.push(ChatRecord::new(
            Role::system(player.entity),
            format!(
                include_str!("prompts/notify_1.md"),
                state.total_cards(),
                state.rock,
                state.paper,
                state.scissors,
                player.inventory.star,
                player.inventory.coin,
                player.inventory.rock,
                player.inventory.paper,
                player.inventory.scissors,
            ),
        ));

        // player reflects
        self.chat.push({
            let role = Role::actor(player.entity, &player.name);
            let prompt = self.prompt_role(&role);
            let names = &[&player.name];
            let sampler = Sampler {
                kind: SamplerKind::Typical,
                ..Default::default()
            };
            self.chat_llm(role, prompt, "", names, sampler).await
        });
    }

    pub async fn chat_trade<'a>(
        &'a mut self,
        player: &'a PlayerData,
        opponent: &'a OpponentData,
        history: &'a [ChatRecord],
        round: usize,
    ) -> Vec<ChatRecord> {
        // update records
        for record in history {
            if !self.chat.iter().any(|x| x == record) {
                self.chat.push(record.clone());
            }
        }

        let mut public_records = vec![];

        // player starts negotiation
        if round == 0 || round == 1 {
            self.chat.push(ChatRecord::new(
                Role::system(player.entity),
                format!(
                    include_str!("prompts/trade_0.md"),
                    opponent.name, opponent.star, opponent.card
                ),
            ));
        }

        // system notifies last round
        if round == (NUM_CHAT_ROUNDS - 1) * 2 || round == (NUM_CHAT_ROUNDS - 1) * 2 + 1 {
            self.chat.push(ChatRecord::new(
                Role::system(player.entity),
                include_str!("prompts/trade_1.md"),
            ));
        }

        // player thinks
        self.chat.push({
            let role = Role::inner(player.entity, &player.name);
            let prompt = self.prompt_role(&role);
            let names = &[&player.name, &opponent.name];
            let sampler = Default::default();
            self.chat_llm(role, prompt, "", names, sampler).await
        });

        let record = if round == 0 || round == 1 {
            let role = Role::actor(player.entity, &player.name);
            let prompt = self.prompt_role(&role);
            let prefix = format!(" (To {})", opponent.name);
            let names = &[&player.name, &opponent.name];
            let sampler = Sampler {
                kind: SamplerKind::Typical,
                ..Default::default()
            };
            self.chat_llm(role, prompt, prefix, names, sampler).await
        } else {
            let role = Role::actor(player.entity, &player.name);
            let prompt = self.prompt_role(&role);
            let names = &[&player.name, &opponent.name];
            let sampler = Sampler {
                kind: SamplerKind::Typical,
                ..Default::default()
            };
            self.chat_llm(role, prompt, "", names, sampler).await
        };
        public_records.push(record.clone());
        self.chat.push(record);

        public_records
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
        player: &'a PlayerData,
        opponent: &'a OpponentData,
        history: &'a [ChatRecord],
        kind: ChatKind,
    ) -> BoxedFuture<'a, Vec<ChatRecord>> {
        match kind {
            ChatKind::Trade(round) => Box::pin(self.chat_trade(player, opponent, history, round)),
            ChatKind::Duel(_) => self.dummy.chat(player, opponent, history, kind),
        }
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
