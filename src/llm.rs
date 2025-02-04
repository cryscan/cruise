use std::sync::Arc;

use anyhow::Result;
use async_std::sync::Mutex;
use bevy::utils::BoxedFuture;
use derivative::Derivative;
use serde::{Deserialize, Serialize};

use crate::game::{
    Actor, Card, ChatKind, ChatRecord, DummyActor, OpponentData, PlayerData, PublicState, Role,
    Stake, StakeState, Trade, TradeState, ASSISTANT_NAME, NUM_CHAT_ROUNDS, SYSTEM_NAME,
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

    pub fn prompt_story(records: &[ChatRecord]) -> String {
        let mut text = String::new();
        let mut last = Role::default();
        for record in records {
            let content = record.content.replace("\r\n", "\n");
            let content = content.trim();

            text = match &record.role {
                x if x == &last => format!("{text}\n{content}"),
                Role::Assistant(_) => format!("{text}\n\n{} (AI): {content}", record.role),
                x => format!("{text}\n\n{x}: {content}"),
            };
            last = record.role.clone();
        }

        text
    }

    pub fn prompt_role(records: &[ChatRecord], role: &Role) -> String {
        format!("{}\n\n{role}:", Self::prompt_story(records))
    }

    pub fn prompt_prefix(records: &[ChatRecord], prefix: impl AsRef<str>) -> String {
        let prefix = prefix.as_ref();
        format!("{}{prefix}", Self::prompt_story(records))
    }

    pub fn prompt_compact(records: &[ChatRecord]) -> String {
        Self::prompt_story(records)
            .replace("\n\n", "\n")
            .trim()
            .to_string()
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

    pub async fn chat_llm(
        &self,
        role: Role,
        prompt: impl AsRef<str>,
        prefix: impl AsRef<str>,
        bnf_schema: impl AsRef<str>,
        player: Option<&PlayerData>,
        opponent: Option<&OpponentData>,
        sampler: Sampler,
    ) -> ChatRecord {
        loop {
            let prompt = prompt.as_ref();
            let prefix = prefix.as_ref();
            let bnf_schema = bnf_schema.as_ref().into();
            let sampler = sampler.clone();

            let mut stop = vec![
                "\n\n".into(),
                format!("\n{}", ASSISTANT_NAME),
                format!("{}:", ASSISTANT_NAME),
                format!("\n{}", SYSTEM_NAME),
                format!("{}:", SYSTEM_NAME),
                "\nUser:".into(),
                "\nQ:".into(),
                "\nAssistant:".into(),
                "\nAI:".into(),
                "\n{".into(),
                "\n[".into(),
                "\n<".into(),
            ];
            if let Some(player) = player {
                stop.extend([
                    format!("\n{}", player.name),
                    format!("\n*{}", player.name),
                    format!("\n**{}", player.name),
                    format!("{}:", player.name),
                ]);
            }
            if let Some(opponent) = opponent {
                stop.extend([
                    format!("\n{}", opponent.name),
                    format!("\n*{}", opponent.name),
                    format!("\n**{}", opponent.name),
                    format!("\n({}", opponent.name),
                    format!("{}:", opponent.name),
                ]);
            }

            let request = CompletionRequest {
                prompt: format!("{prompt}{prefix}"),
                state: self.state,
                stop,
                sampler,
                bnf_schema,
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
                    Role::Assistant(player.entity),
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
            Role::Assistant(player.entity),
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
            let prompt = Self::prompt_role(&self.chat, &role);
            let sampler = Sampler {
                kind: SamplerKind::Typical,
                ..Default::default()
            };
            self.chat_llm(role, prompt, "", "", Some(&player), None, sampler)
                .await
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
                Role::Assistant(player.entity),
                format!(
                    include_str!("prompts/trade_0.md"),
                    opponent.name, opponent.star, opponent.card
                ),
            ));
        }

        // system notifies last round
        if round == (NUM_CHAT_ROUNDS - 1) * 2 || round == (NUM_CHAT_ROUNDS - 1) * 2 + 1 {
            self.chat.push(ChatRecord::new(
                Role::Assistant(player.entity),
                include_str!("prompts/trade_1.md"),
            ));
        }

        // player public words
        let record = {
            let role = Role::actor(player.entity, &player.name);
            let prompt = Self::prompt_role(&self.chat, &role);
            let sampler = Sampler {
                kind: SamplerKind::Typical,
                ..Default::default()
            };
            self.chat_llm(
                role,
                prompt,
                "",
                "",
                Some(&player),
                Some(&opponent),
                sampler,
            )
            .await
        };
        public_records.push(record.clone());
        self.chat.push(record);

        public_records
    }

    pub async fn trade<'a>(
        &'a mut self,
        player: &'a PlayerData,
        opponent: &'a OpponentData,
        history: &'a [ChatRecord],
    ) -> Trade {
        loop {
            let prompt = format!(
                include_str!("prompts/trade_3.md"),
                player.name,
                opponent.name,
                Self::prompt_compact(history)
            );

            let sampler = Sampler {
                top_p: 0.8,
                presence_penalty: 0.0,
                frequency_penalty: 0.0,
                ..Default::default()
            };

            let request = CompletionRequest {
                prompt,
                state: self.state,
                stop: vec!["\n\n".into()],
                sampler,
                bnf_schema: include_str!("prompts/bnf_trade.txt").into(),
                ..Default::default()
            };
            let response = match self.call_llm(&request).await {
                Ok(response) => response,
                Err(err) => {
                    bevy::log::error!("{err}");
                    continue;
                }
            };

            let json = response.model_text();
            bevy::log::info!("{}: {json}", player.name);

            self.llm.lock().await.push(LlmRecord { request, response });

            match serde_json::from_str::<Trade>(&json) {
                Ok(trade) => break trade.normalize(&player.inventory),
                Err(err) => {
                    bevy::log::error!("{err}");
                    continue;
                }
            };
        }
    }

    pub async fn accept_trade<'a>(
        &'a mut self,
        player: &'a PlayerData,
        opponent: &'a OpponentData,
        _history: &'a [ChatRecord],
        state: TradeState<'a>,
    ) -> bool {
        // display contract form
        self.chat.extend([
            ChatRecord::new(
                Role::Assistant(player.entity),
                include_str!("prompts/trade_4.md"),
            ),
            ChatRecord::new(
                Role::System(player.entity),
                format!(
                    include_str!("prompts/trade_5.md"),
                    player.name,
                    opponent.name,
                    state.this.star,
                    state.this.coin,
                    state.this.rock,
                    state.this.paper,
                    state.this.scissors,
                    state.that.star,
                    state.that.coin,
                    state.that.rock,
                    state.that.paper,
                    state.that.scissors,
                ),
            ),
            ChatRecord::new(
                Role::Assistant(player.entity),
                include_str!("prompts/trade_6.md"),
            ),
        ]);

        self.chat.push({
            let role = Role::actor(player.entity, &player.name);
            let prompt = Self::prompt_role(&self.chat, &role);
            let sampler = Sampler {
                kind: SamplerKind::Typical,
                ..Default::default()
            };
            self.chat_llm(
                role,
                prompt,
                "",
                "",
                Some(&player),
                Some(&opponent),
                sampler,
            )
            .await
        });

        let record = {
            let role = Role::actor(player.entity, &player.name);
            let prompt = Self::prompt_role(&self.chat, &role);
            let sampler = Sampler {
                kind: SamplerKind::Typical,
                ..Default::default()
            };
            let prefixes = [
                "So, my answer is \"",
                "So I think I will give it a \"",
                "Hmm... I have reviewed it and I guess I shall give it a \"",
                "My verdict is \"",
                "After consideration, my response is \"",
                "The answer I provide is \"",
                "Upon review, I decide on \"",
                "My decision stands as \"",
                "I give my response with a \"",
            ];
            self.chat_llm(
                role,
                prompt,
                fastrand::choice(&prefixes).unwrap(),
                "start ::= \"Yes\\\".\" | \"No\\\".\";",
                Some(&player),
                Some(&opponent),
                sampler,
            )
            .await
        };
        let ans = record.content.contains("Yes");
        self.chat.push(record);
        ans
    }

    pub async fn feedback_trade<'a>(&'a mut self, player: &'a PlayerData, state: [bool; 2]) {
        // system reports trade result
        let record = match state {
            [true, true] => ChatRecord::new(
                Role::Assistant(player.entity),
                format!(
                    include_str!("prompts/trade_7_0.md"),
                    player.inventory.star,
                    player.inventory.coin,
                    player.inventory.rock,
                    player.inventory.paper,
                    player.inventory.scissors
                ),
            ),
            _ => ChatRecord::new(
                Role::Assistant(player.entity),
                include_str!("prompts/trade_7_1.md"),
            ),
        };
        self.chat.push(record);

        // player reflects
        self.chat.push({
            let role = Role::actor(player.entity, &player.name);
            let prompt = Self::prompt_role(&self.chat, &role);
            let sampler = Sampler {
                kind: SamplerKind::Typical,
                ..Default::default()
            };
            self.chat_llm(role, prompt, "", "", Some(&player), None, sampler)
                .await
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

    fn feedback_error<'a>(&'a mut self, data: &'a PlayerData, text: String) -> BoxedFuture<'a, ()> {
        self.dummy.feedback_error(data, text)
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
        player: &'a PlayerData,
        opponent: &'a OpponentData,
        history: &'a [ChatRecord],
    ) -> BoxedFuture<'a, Trade> {
        Box::pin(self.trade(player, opponent, history))
    }

    fn accept_trade<'a>(
        &'a mut self,
        player: &'a PlayerData,
        opponent: &'a OpponentData,
        history: &'a [ChatRecord],
        state: TradeState<'a>,
    ) -> BoxedFuture<'a, bool> {
        Box::pin(self.accept_trade(player, opponent, history, state))
    }

    fn feedback_trade<'a>(
        &'a mut self,
        player: &'a PlayerData,
        state: [bool; 2],
    ) -> BoxedFuture<'a, ()> {
        Box::pin(self.feedback_trade(player, state))
    }

    fn bet<'a>(
        &'a mut self,
        player: &'a PlayerData,
        opponent: &'a OpponentData,
        history: &'a [ChatRecord],
    ) -> BoxedFuture<'a, Stake> {
        self.dummy.bet(player, opponent, history)
    }

    fn accept_duel<'a>(
        &'a mut self,
        players: &'a PlayerData,
        opponent: &'a OpponentData,
        history: &'a [ChatRecord],
        state: StakeState<'a>,
    ) -> BoxedFuture<'a, Option<Card>> {
        self.dummy.accept_duel(players, opponent, history, state)
    }
}
