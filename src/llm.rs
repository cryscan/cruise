use std::{collections::HashMap, path::PathBuf, sync::Arc};

use anyhow::Result;
use async_std::sync::Mutex;
use bevy::{core::Name, utils::BoxedFuture};
use derivative::Derivative;
use futures::join;
use itertools::Itertools;
use serde::{de::DeserializeOwned, Deserialize, Serialize};

use crate::game::{
    Actor, Card, ChatKind, ChatRecord, DuelResult, DummyActor, Inventory, OpponentData, PlayerData,
    PublicState, Role, Stake, StakeState, Trade, TradeState, ASSISTANT_NAME, NUM_CHAT_ROUNDS,
    SYSTEM_NAME,
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
    pub bias: HashMap<u16, f32>,
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
pub struct ChooseRequest {
    #[serde(rename = "input")]
    pub prompt: String,
    pub state: uuid::Uuid,
    pub choices: Vec<String>,
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct ChooseResponse {
    pub data: Vec<ChooseItem>,
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct ChooseItem {
    pub choice: String,
    pub index: usize,
    pub rank: usize,
    pub perplexity: f32,
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct Choice {
    pub index: u32,
    pub text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LlmRecord {
    Completion {
        role: Role,
        player: Option<PlayerData>,
        opponent: Option<OpponentData>,
        request: Box<CompletionRequest>,
        response: Box<CompletionResponse>,
    },
    Choose {
        role: Role,
        request: Box<ChooseRequest>,
        response: Box<ChooseResponse>,
    },
}

#[derive(Debug, Default, Clone)]
pub struct LlmActor {
    pub url: String,
    pub output: PathBuf,

    pub chat: Vec<ChatRecord>,
    pub history: Arc<Mutex<Vec<LlmRecord>>>,

    pub state: uuid::Uuid,
    pub dummy: DummyActor,
}

impl LlmActor {
    pub fn new(url: impl ToString, output: PathBuf) -> Self {
        let url = url.to_string();
        Self {
            url,
            output,
            ..Default::default()
        }
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

    pub async fn call_llm<T: DeserializeOwned>(
        &self,
        url: impl ToString,
        request: &impl Serialize,
    ) -> Result<T> {
        async_std::task::yield_now().await;
        let response = ehttp::fetch_async(ehttp::Request::json(url, request)?)
            .await
            .map_err(|err| anyhow::anyhow!(err))?;
        Ok(response.json()?)
    }

    #[allow(clippy::too_many_arguments)]
    pub async fn chat_llm(
        &self,
        head: impl AsRef<str>,
        role: &Role,
        prompt: impl AsRef<str>,
        prefix: impl AsRef<str>,
        bnf_schema: impl AsRef<str>,
        stop: &[impl AsRef<str>],
        bias: &[(u16, f32)],
        player: Option<&PlayerData>,
        opponent: Option<&OpponentData>,
        sampler: Sampler,
    ) -> ChatRecord {
        loop {
            let head = head.as_ref();
            let prompt = prompt.as_ref();
            let prefix = prefix.as_ref();
            let bnf_schema = bnf_schema.as_ref().into();
            let sampler = sampler.clone();

            let mut stop = stop.iter().map(|x| x.as_ref().to_string()).collect_vec();
            stop.extend([
                format!("{}:", ASSISTANT_NAME),
                format!("{}:", SYSTEM_NAME),
                "\nUser:".into(),
                "\nQ:".into(),
                "\nAssistant:".into(),
                "\nAI:".into(),
            ]);
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

            let bias = bias.iter().cloned().collect();

            let request = CompletionRequest {
                prompt: format!("{prompt}{prefix}"),
                state: self.state,
                stop,
                bias,
                sampler,
                bnf_schema,
                ..Default::default()
            };
            let response: CompletionResponse = match self
                .call_llm(format!("{}/api/oai/completions", self.url), &request)
                .await
            {
                Ok(response) => response,
                Err(err) => {
                    bevy::log::error!("{err}");
                    continue;
                }
            };

            let content = format!("{prefix}{}", response.model_text());
            if content.is_empty() {
                bevy::log::warn!("[{role}]{head} empty response");
                continue;
            }

            let record = ChatRecord::new(role.clone(), content);
            // bevy::log::info!("{head}[prompt] {prompt}{prefix}");
            bevy::log::info!("{head} {record}");

            {
                let role = role.clone();
                let player = player.cloned();
                let opponent = opponent.cloned();
                let request = Box::new(request);
                let response = Box::new(response);
                self.history.lock().await.push(LlmRecord::Completion {
                    role,
                    player,
                    opponent,
                    request,
                    response,
                });
            }

            break record;
        }
    }

    pub async fn choose_llm(
        &self,
        head: impl AsRef<str>,
        role: &Role,
        prompt: impl AsRef<str>,
        choices: &[impl AsRef<str>],
    ) -> Vec<String> {
        loop {
            let head = head.as_ref();
            let prompt = prompt.as_ref().to_string();
            let choices = choices
                .iter()
                .map(|choice| choice.as_ref().to_string())
                .collect_vec();

            let request = ChooseRequest {
                prompt,
                state: self.state,
                choices,
            };
            let response: ChooseResponse = match self
                .call_llm(format!("{}/api/oai/chooses", self.url), &request)
                .await
            {
                Ok(response) => response,
                Err(err) => {
                    bevy::log::error!("{err}");
                    continue;
                }
            };

            let choices = response
                .data
                .iter()
                .map(|item| item.choice.clone())
                .collect_vec();
            bevy::log::info!("{head} {role}: {:?}", choices);

            {
                let role = role.clone();
                let request = Box::new(request);
                let response = Box::new(response);
                self.history.lock().await.push(LlmRecord::Choose {
                    role,
                    request,
                    response,
                });
            }

            break choices;
        }
    }

    pub async fn notify<'a>(&'a mut self, player: &'a PlayerData, state: &'a PublicState) {
        self.chat.clear();

        self.chat.extend([
            ChatRecord::new(
                Role::Assistant(player.entity),
                format!(
                    include_str!("prompts/notify_0_ai.md"),
                    player = player.name,
                    ai = ASSISTANT_NAME
                ),
            ),
            ChatRecord::new(
                Role::actor(player.entity, &player.name),
                "Ahh... I feel so dizzy... What happened to me? What's the situation right now?",
            ),
            ChatRecord::new(
                Role::Assistant(player.entity),
                format!(
                    include_str!("prompts/notify_1_ai.md"),
                    num_players = state.player,
                ),
            ),
            ChatRecord::new(
                Role::actor(player.entity, &player.name),
                include_str!("prompts/notify_2_user.md"),
            ),
            ChatRecord::new(
                Role::Assistant(player.entity),
                format!(
                    include_str!("prompts/notify_3_ai.md"),
                    star = player.inventory.star,
                    coin = player.inventory.coin,
                    num_cards = player.inventory.num_cards(),
                    rock = player.inventory.rock,
                    paper = player.inventory.paper,
                    scissors = player.inventory.scissors,
                    num_players = state.player,
                    total_rock = state.rock,
                    total_paper = state.paper,
                    total_scissors = state.scissors,
                ),
            ),
            ChatRecord::new(
                Role::actor(player.entity, &player.name),
                "Looks like it's just luck.",
            ),
            ChatRecord::new(
                Role::Assistant(player.entity),
                include_str!("prompts/notify_4_ai.md"),
            ),
            ChatRecord::new(
                Role::actor(player.entity, &player.name),
                "How much time do we left before game over?",
            ),
            match player.timer.0 {
                0..=2 => ChatRecord::new(
                    Role::Assistant(player.entity),
                    format!(
                        include_str!("prompts/notify_5_1_ai.md"),
                        player.timer.0 * 10,
                        player.timer.0,
                    ),
                ),
                _ => ChatRecord::new(
                    Role::Assistant(player.entity),
                    format!(
                        include_str!("prompts/notify_5_0_ai.md"),
                        player.timer.0 * 10,
                        player.timer.0,
                    ),
                ),
            },
            ChatRecord::new(
                Role::actor(player.entity, &player.name),
                format!(include_str!("prompts/notify_6_user.md"), ASSISTANT_NAME),
            ),
        ]);

        // AI advices
        self.chat.push({
            let role = Role::Assistant(player.entity);
            let prompt = Self::prompt_role(&self.chat, &role);
            self.chat_llm(
                format!("[notify][{}]", player.name),
                &role,
                prompt,
                // " Based on the situation and your current status,",
                "",
                "",
                &["\n\n"],
                &[(59, -1.0e30)],
                Some(player),
                None,
                Sampler {
                    kind: SamplerKind::Typical,
                    temperature: 1.5,
                    ..Default::default()
                },
            )
            .await
        });

        self.chat.push(ChatRecord::new(
            Role::actor(player.entity, &player.name),
            format!("I see, thank you, {ASSISTANT_NAME}."),
        ));
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
            self.chat.extend([
                ChatRecord::new(
                    Role::System(player.entity),
                    format!("*{} joins chat*", opponent.name),
                ),
                ChatRecord::new(
                    Role::Assistant(player.entity),
                    format!(
                        include_str!("prompts/trade_0.md"),
                        opponent.name, opponent.star, opponent.card
                    ),
                ),
            ]);
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
                temperature: 1.5,
                ..Default::default()
            };
            self.chat_llm(
                format!("[chat][{round}]"),
                &role,
                prompt,
                "",
                "",
                &["\n\n", "\n"],
                &[],
                Some(player),
                Some(opponent),
                sampler,
            )
            .await
        };
        public_records.push(record.clone());
        self.chat.push(record);

        if round == (NUM_CHAT_ROUNDS - 1) * 2 || round == (NUM_CHAT_ROUNDS - 1) * 2 + 1 {
            self.chat.push(ChatRecord::new(
                Role::System(player.entity),
                format!("*{} leaves chat*", opponent.name),
            ));
        }

        public_records
    }

    pub async fn trade_item<'a>(
        &'a self,
        player: &'a PlayerData,
        opponent: &'a OpponentData,
        _history: &'a [ChatRecord],
        item: impl AsRef<str> + 'a,
        choices: impl Iterator<Item = usize> + 'a,
    ) -> usize {
        let item = item.as_ref();

        let mut chat = self.chat.clone();
        chat.push(ChatRecord::new(
            Role::Assistant(player.entity),
            format!(include_str!("prompts/trade_3_1.md"), opponent.name, item),
        ));

        let role = Role::actor(player.entity, &player.name);
        let prompt = Self::prompt_role(&chat, &role);

        {
            let choices = [
                " I think I wouldn't like to",
                " I would like to",
                " Hmm... I think I wouldn't like to",
                " Hmm... I wouldn't like to",
            ];
            let choices = self
                .choose_llm(
                    format!("[trade][{item}][{}][0]", player.name),
                    &role,
                    &prompt,
                    &choices,
                )
                .await;
            match choices[0].as_ref() {
                " I think I wouldn't like to" | " Hmm... I think I wouldn't like to" => return 0,
                " I would like to" | " Hmm... I would like to" => {}
                _ => unreachable!(),
            }
        }

        let prompt = format!("{prompt} Hmm... I would like to offer {}", opponent.name);
        let choices = choices.map(|x| format!(" {x}")).collect_vec();
        let choices = self
            .choose_llm(
                format!("[trade][{item}][{}][1]", player.name),
                &role,
                prompt,
                &choices,
            )
            .await;

        choices
            .into_iter()
            .map(|x| x.trim().parse::<usize>().expect("cannot parse the result"))
            .next()
            .unwrap_or(0)
    }

    pub async fn trade<'a>(
        &'a mut self,
        player: &'a PlayerData,
        opponent: &'a OpponentData,
        history: &'a [ChatRecord],
    ) -> Trade {
        self.chat.push(ChatRecord::new(
            Role::Assistant(player.entity),
            format!(include_str!("prompts/trade_3_0.md"), opponent.name),
        ));

        self.chat.push({
            let role = Role::actor(player.entity, &player.name);
            let prompt = Self::prompt_role(&self.chat, &role);
            let sampler = Sampler {
                kind: SamplerKind::Typical,
                temperature: 1.5,
                ..Default::default()
            };
            self.chat_llm(
                "[trade][summarize]",
                &role,
                prompt,
                "",
                "",
                &["\n\n", "\n"],
                &[],
                Some(player),
                Some(opponent),
                sampler,
            )
            .await
        });

        let Inventory {
            star,
            coin,
            rock,
            paper,
            scissors,
        } = player.inventory.clone();

        let (star, coin, rock, paper, scissors) = join!(
            self.trade_item(player, opponent, history, "stars", 0..star),
            self.trade_item(player, opponent, history, "coins", 0..coin),
            self.trade_item(player, opponent, history, "rock cards", 0..rock),
            self.trade_item(player, opponent, history, "paper cards", 0..paper),
            self.trade_item(player, opponent, history, "scissors cards", 0..scissors)
        );

        let trade = Trade {
            star,
            coin,
            rock,
            paper,
            scissors,
        };
        bevy::log::info!("[trade][{}] {:?}", player.name, trade);
        trade
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

        // player reacts to the contract
        self.chat.push({
            let role = Role::actor(player.entity, &player.name);
            let prompt = Self::prompt_role(&self.chat, &role);
            let sampler = Sampler {
                kind: SamplerKind::Typical,
                temperature: 1.5,
                ..Default::default()
            };
            self.chat_llm(
                "[trade][accept]",
                &role,
                prompt,
                "",
                "",
                &["\n\n", "\n"],
                &[],
                Some(player),
                Some(opponent),
                sampler,
            )
            .await
        });

        self.chat.push(ChatRecord::new(
            Role::System(player.entity),
            include_str!("prompts/trade_7.md"),
        ));

        let record = {
            let role = Role::actor(player.entity, &player.name);
            let prompt = Self::prompt_role(&self.chat, &role);
            let sampler = Sampler {
                kind: SamplerKind::Typical,
                temperature: 1.5,
                ..Default::default()
            };
            let prefixes = [
                " So, my answer is \"",
                " So I think I will give it a \"",
                " Hmm... I have reviewed it and I guess I shall give it a \"",
                " My verdict is \"",
                " After consideration, my response is \"",
                " The answer I provide is \"",
                " Upon review, I decide on \"",
                " My decision stands as \"",
                " I give my response with a \"",
            ];
            self.chat_llm(
                "[trade][confirm]",
                &role,
                prompt,
                fastrand::choice(&prefixes).unwrap(),
                "start ::= \"Yes\\\".\" | \"No\\\".\";",
                &["\n\n", "\n"],
                &[],
                Some(player),
                Some(opponent),
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
                    include_str!("prompts/trade_8_0.md"),
                    star = player.inventory.star,
                    coin = player.inventory.coin,
                    num_cards = player.inventory.num_cards(),
                    rock = player.inventory.rock,
                    paper = player.inventory.paper,
                    scissors = player.inventory.scissors,
                ),
            ),
            _ => ChatRecord::new(
                Role::Assistant(player.entity),
                include_str!("prompts/trade_8_1.md"),
            ),
        };
        self.chat.push(record);

        // player reflects
        self.chat.push({
            let role = Role::actor(player.entity, &player.name);
            let prompt = Self::prompt_role(&self.chat, &role);
            let sampler = Sampler {
                kind: SamplerKind::Typical,
                temperature: 1.5,
                ..Default::default()
            };
            self.chat_llm(
                "[trade][feedback]",
                &role,
                prompt,
                "",
                "",
                &["\n\n", "\n"],
                &[],
                Some(player),
                None,
                sampler,
            )
            .await
        });
    }

    pub async fn bet<'a>(
        &'a mut self,
        player: &'a PlayerData,
        opponent: &'a OpponentData,
        history: &'a [ChatRecord],
    ) -> Stake {
        // system reports opponent status
        self.chat.extend([
            ChatRecord::new(
                Role::Assistant(player.entity),
                format!(
                    include_str!("prompts/duel_0_ai.md"),
                    opponent.name, opponent.star, opponent.card
                ),
            ),
            ChatRecord::new(
                Role::actor(player.entity, &player.name),
                format!(include_str!("prompts/duel_1_user.md"), ASSISTANT_NAME),
            ),
        ]);

        // AI advices
        self.chat.push({
            let role = Role::Assistant(player.entity);
            let prompt = Self::prompt_role(&self.chat, &role);
            self.chat_llm(
                format!("[bet][0][{}]", player.name),
                &role,
                prompt,
                " Let's analyze the situation.",
                "",
                &["\n\n"],
                &[],
                Some(player),
                Some(opponent),
                Default::default(),
            )
            .await
        });

        // player reflects
        self.chat.push({
            let role = Role::actor(player.entity, &player.name);
            let prompt = Self::prompt_role(&self.chat, &role);
            let sampler = Sampler {
                kind: SamplerKind::Typical,
                temperature: 1.5,
                ..Default::default()
            };
            self.chat_llm(
                "[bet][1]",
                &role,
                prompt,
                "",
                "",
                &["\n\n", "\n"],
                &[],
                Some(player),
                None,
                sampler,
            )
            .await
        });

        self.dummy.bet(player, opponent, history).await
    }

    pub async fn accept_duel<'a>(
        &'a mut self,
        player: &'a PlayerData,
        opponent: &'a OpponentData,
        _history: &'a [ChatRecord],
        _state: StakeState<'a>,
    ) -> Option<Card> {
        let mut history = vec![];

        let record = ChatRecord::new(
            Role::Assistant(player.entity),
            format!(
                include_str!("prompts/duel_2.md"),
                num_cards = player.inventory.num_cards(),
                rock = player.inventory.rock,
                paper = player.inventory.paper,
                scissors = player.inventory.scissors
            ),
        );
        // history.push(record.clone());
        self.chat.push(record);

        // player prepares action
        let record = {
            let role = Role::actor(player.entity, &player.name);
            let prompt = Self::prompt_role(&self.chat, &role);
            let sampler = Sampler {
                kind: SamplerKind::Typical,
                temperature: 1.5,
                ..Default::default()
            };
            self.chat_llm(
                "[duel][prepare]",
                &role,
                prompt,
                "",
                "",
                &["\n\n", "\n"],
                &[],
                Some(player),
                None,
                sampler,
            )
            .await
        };
        history.push(record.clone());
        self.chat.push(record);

        self.chat.push(ChatRecord::new(
            Role::Assistant(player.entity),
            include_str!("prompts/duel_4.md"),
        ));

        let deck = [
            vec![Card::Rock; player.inventory.rock.min(1)],
            vec![Card::Paper; player.inventory.paper.min(1)],
            vec![Card::Scissors; player.inventory.scissors.min(1)],
        ]
        .concat();
        let choices = deck.into_iter().map(|card| card.to_string()).collect_vec();

        if !choices.is_empty() {
            let role = Role::actor(player.entity, &player.name);
            let prefixes = [
                " Ok, the card I wish to draw is \"",
                " All right, the card I'm drawing is \"",
                " Fine, the card I draw turns out to be \"",
            ];
            let prefix = fastrand::choice(&prefixes).unwrap();
            let prompt = Self::prompt_role(&self.chat, &role);
            let choices = self
                .choose_llm(
                    "[duel][confirm]",
                    &role,
                    format!("{prompt}{prefix}"),
                    &choices,
                )
                .await;
            let card = match choices[0].as_ref() {
                "Rock" => Card::Rock,
                "Paper" => Card::Paper,
                "Scissors" => Card::Scissors,
                _ => unreachable!(),
            };
            self.chat.push({
                let content = format!("{prefix}{card}\".");
                ChatRecord::new(role, content)
            });
            Some(card)
        } else {
            self.chat.push({
                let role = Role::actor(player.entity, &player.name);
                let content = format!(" I don't want to duel with {}.", opponent.name);
                ChatRecord::new(role, content)
            });
            None
        }
    }

    pub async fn feedback_duel<'a>(&'a mut self, player: &'a PlayerData, result: DuelResult) {
        let prompt = match result {
            DuelResult::Tie(card) => format!("It's a tie, you both draw \"{card}\" card."),
            DuelResult::Win(this, that) => format!("\"{this}\" vs. \"{that}\". You win!"),
            DuelResult::Lose(this, that) => format!("\"{this}\" vs. \"{that}\". You lose."),
        };
        let prompt = format!("Let's reveal duel result... {prompt}");
        self.chat
            .push(ChatRecord::new(Role::System(player.entity), prompt));

        // player reflects
        self.chat.push({
            let role = Role::actor(player.entity, &player.name);
            let prompt = Self::prompt_role(&self.chat, &role);
            let sampler = Sampler {
                kind: SamplerKind::Typical,
                temperature: 1.5,
                ..Default::default()
            };
            self.chat_llm(
                "[duel][feedback]",
                &role,
                prompt,
                "",
                "",
                &["\n\n", "\n"],
                &[],
                Some(player),
                None,
                sampler,
            )
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
        Box::pin(self.bet(player, opponent, history))
    }

    fn accept_duel<'a>(
        &'a mut self,
        player: &'a PlayerData,
        opponent: &'a OpponentData,
        history: &'a [ChatRecord],
        state: StakeState<'a>,
    ) -> BoxedFuture<'a, Option<Card>> {
        Box::pin(self.accept_duel(player, opponent, history, state))
    }

    fn feedback_duel<'a>(
        &'a mut self,
        player: &'a PlayerData,
        result: DuelResult,
    ) -> BoxedFuture<'a, ()> {
        Box::pin(self.feedback_duel(player, result))
    }

    fn dump<'a>(&'a self, player: &'a PlayerData) -> BoxedFuture<'a, Result<Vec<u8>>> {
        #[derive(Serialize)]
        struct DumpData<'a> {
            name: &'a Name,
            inventory: &'a Inventory,
            history: &'a [LlmRecord],
        }

        Box::pin(async move {
            let history = self.history.lock().await;
            let data = DumpData {
                name: &player.name,
                inventory: &player.inventory,
                history: &history[..],
            };
            Ok(serde_json::to_vec(&data)?)
        })
    }
}
