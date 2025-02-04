use std::{
    fmt::Display,
    ops::{Add, Not},
    sync::Arc,
};

use anyhow::{bail, Result};
use async_std::{sync::Mutex, task::block_on};
use bevy::{
    prelude::*,
    tasks::{futures_lite::future, IoTaskPool, Task},
    utils::{BoxedFuture, ConditionalSend},
};
use derivative::Derivative;
use futures::join;
use itertools::Itertools;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::llm::LlmActor;

pub const NUM_PLAYERS: usize = 16;
pub const MIN_MATCH_PLAYERS: usize = 2;
pub const NUM_CHAT_ROUNDS: usize = 4;
pub const MAX_TRAIL_ROUNDS: usize = 3;

pub const SYSTEM_NAME: &str = "System";
pub const ASSISTANT_NAME: &str = "Stellaris";
const NAMES: &str = include_str!("names.txt");

#[derive(Debug, Default)]
pub struct GamePlugin;

impl Plugin for GamePlugin {
    fn build(&self, app: &mut App) {
        app.register_type::<Inventory>()
            .register_type::<Table>()
            .register_type::<PublicState>()
            .init_resource::<PublicState>()
            .add_systems(Startup, setup_scene)
            .add_systems(
                Update,
                (
                    update_public_state,
                    match_players,
                    remove_players,
                    start_duel,
                    poll_duel,
                ),
            );
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Card {
    Rock,
    Paper,
    Scissors,
}

impl Card {
    pub fn compare(self: Card, rhs: Card) -> Option<usize> {
        match (self, rhs) {
            (Card::Rock, Card::Rock) => None,
            (Card::Rock, Card::Paper) => Some(1),
            (Card::Rock, Card::Scissors) => Some(0),
            (Card::Paper, Card::Rock) => Some(0),
            (Card::Paper, Card::Paper) => None,
            (Card::Paper, Card::Scissors) => Some(1),
            (Card::Scissors, Card::Rock) => Some(1),
            (Card::Scissors, Card::Paper) => Some(0),
            (Card::Scissors, Card::Scissors) => None,
        }
    }
}

impl Display for Card {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Card::Rock => write!(f, "Rock"),
            Card::Paper => write!(f, "Paper"),
            Card::Scissors => write!(f, "Scissors"),
        }
    }
}

#[derive(Debug, Derivative, Clone, Component, Reflect, Serialize, Deserialize)]
#[derivative(Default)]
#[reflect(Component)]
pub struct Inventory {
    #[derivative(Default(value = "3"))]
    pub star: u32,
    #[derivative(Default(value = "1000"))]
    pub coin: u32,
    #[derivative(Default(value = "3"))]
    pub rock: u32,
    #[derivative(Default(value = "3"))]
    pub paper: u32,
    #[derivative(Default(value = "3"))]
    pub scissors: u32,
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct Trade {
    pub star: u32,
    pub coin: u32,
    pub rock: u32,
    pub paper: u32,
    pub scissors: u32,
}

impl Trade {
    pub fn normalize(self, inventory: &Inventory) -> Self {
        assert!(inventory.star > 0);
        Self {
            star: self.star.min(inventory.star - 1),
            coin: self.coin.min(inventory.coin),
            rock: self.rock.min(inventory.rock),
            paper: self.paper.min(inventory.paper),
            scissors: self.scissors.min(inventory.scissors),
        }
    }
}

#[derive(Debug, Clone, Copy, Error)]
pub enum TradeError {
    #[error("cannot take out {1} star(s) since you only have {0}")]
    Star(u32, u32),
    #[error("cannot take out {1} coin(s) since you only have {0}")]
    Coin(u32, u32),
    #[error("cannot take out {1} rock card(s) since you only have {0}")]
    Rock(u32, u32),
    #[error("cannot take out {1} paper card(s) since you only have {0}")]
    Paper(u32, u32),
    #[error("cannot take out {1} scissors card(s) since you only have {0}")]
    Scissors(u32, u32),
}

#[derive(Debug, Derivative, Clone, Serialize, Deserialize)]
#[derivative(Default)]
pub struct Stake {
    #[derivative(Default(value = "1"))]
    pub star: u32,
    pub coin: u32,
}

impl Stake {
    pub fn normalize(self) -> Self {
        let Self { star, coin } = self;
        Self {
            star: star.max(1),
            coin,
        }
    }
}

impl Add<Stake> for Stake {
    type Output = Self;

    fn add(self, rhs: Stake) -> Self::Output {
        Self::Output {
            star: self.star + rhs.star,
            coin: self.coin + rhs.coin,
        }
    }
}

#[derive(Debug, Clone, Copy, Error)]
pub enum StakeError {
    #[error("cannot take out {1} star(s) since you only have {0} star(s)")]
    Star(u32, u32),
    #[error("cannot take out {1} coin(s) since you only have {0} coin(s)")]
    Coin(u32, u32),
}

#[derive(Debug, Clone, Copy, Error)]
pub enum DuelError {
    #[error("cannot draw rock since you do not have such card")]
    Rock,
    #[error("cannot draw paper since you do not have such card")]
    Paper,
    #[error("cannot draw scissors since you do not have such card")]
    Scissors,
}

impl Inventory {
    pub fn num_cards(&self) -> u32 {
        self.rock + self.paper + self.scissors
    }

    pub fn is_alive(&self) -> bool {
        self.star > 0
    }

    pub fn is_safe(&self) -> bool {
        self.star >= 3 && self.num_cards() == 0
    }

    pub fn can_duel(&self) -> bool {
        self.num_cards() > 0
    }

    pub fn split_trade(&self, trade: &Trade) -> Result<Self, TradeError> {
        match (self, trade) {
            (x, y) if x.star < y.star => Err(TradeError::Star(x.star, y.star)),
            (x, y) if x.coin < y.coin => Err(TradeError::Coin(x.coin, y.coin)),
            (x, y) if x.rock < y.rock => Err(TradeError::Rock(x.rock, y.rock)),
            (x, y) if x.paper < y.paper => Err(TradeError::Paper(x.paper, y.paper)),
            (x, y) if x.scissors < y.scissors => Err(TradeError::Scissors(x.scissors, y.scissors)),
            (x, y) => Ok(Self {
                star: x.star - y.star,
                coin: x.coin - y.coin,
                rock: x.rock - y.rock,
                paper: x.paper - y.paper,
                scissors: x.scissors - y.scissors,
            }),
        }
    }

    pub fn apply_trade(&mut self, trade: &Trade) {
        self.star += trade.star;
        self.coin += trade.coin;
        self.rock += trade.rock;
        self.paper += trade.paper;
        self.scissors += trade.scissors;
    }

    pub fn split_stake(&self, stake: &Stake) -> Result<Self, StakeError> {
        match (self, stake) {
            (x, y) if x.star < y.star => Err(StakeError::Star(x.star, y.star)),
            (x, y) if x.coin < y.coin => Err(StakeError::Coin(x.coin, y.coin)),
            (x, y) => Ok(Self {
                star: x.star - y.star,
                coin: x.coin - y.coin,
                ..x.clone()
            }),
        }
    }

    pub fn apply_stake(&mut self, stake: &Stake) {
        self.star += stake.star;
        self.coin += stake.coin;
    }

    pub fn split_duel(&self, card: Card) -> Result<Self, DuelError> {
        match (self, card) {
            (x, Card::Rock) if x.rock == 0 => Err(DuelError::Rock),
            (x, Card::Paper) if x.paper == 0 => Err(DuelError::Paper),
            (x, Card::Scissors) if x.scissors == 0 => Err(DuelError::Scissors),
            (x, Card::Rock) => Ok(Self {
                rock: x.rock - 1,
                ..x.clone()
            }),
            (x, Card::Paper) => Ok(Self {
                paper: x.paper - 1,
                ..x.clone()
            }),
            (x, Card::Scissors) => Ok(Self {
                scissors: x.scissors - 1,
                ..x.clone()
            }),
        }
    }
}

#[derive(Derivative, Component)]
#[derivative(Debug)]
pub struct Player {
    #[derivative(Debug = "ignore")]
    pub actor: Arc<Mutex<dyn Actor>>,
}

impl Player {
    pub fn new(actor: impl Actor) -> Self {
        let actor = Arc::new(Mutex::new(actor));
        Self { actor }
    }
}

#[derive(Debug, Clone, Deref, DerefMut, Component, Reflect)]
#[reflect(Component)]
pub struct Table(pub [Entity; 2]);

impl Table {
    pub fn new(x: Entity, y: Entity) -> Self {
        Self([x, y])
    }
}

#[derive(Debug, Default, Clone, Resource, Reflect, Serialize, Deserialize)]
#[reflect(Resource)]
pub struct PublicState {
    pub rock: u32,
    pub paper: u32,
    pub scissors: u32,
}

impl PublicState {
    pub fn total_cards(&self) -> u32 {
        self.rock + self.paper + self.scissors
    }
}

fn setup_scene(mut commands: Commands) {
    let names = NAMES.split("\n").map(|x| x.trim()).collect_vec();
    commands.spawn_batch((0..NUM_PLAYERS).map(move |index| {
        (
            Name::new(names[index]),
            Player::new(LlmActor::new()),
            Inventory::default(),
        )
    }));
}

fn update_public_state(mut state: ResMut<PublicState>, players: Query<&Inventory, With<Player>>) {
    *state = Default::default();
    for inventory in &players {
        state.rock += inventory.rock;
        state.paper += inventory.paper;
        state.scissors += inventory.scissors;
    }
}

/// Find players that are not currently in match, and put them onto a table.
fn match_players(
    mut commands: Commands,
    mut players: Query<(Entity, &Name, &Inventory), With<Player>>,
    tables: Query<&Table, Without<Player>>,
) {
    let mut total_cards = 0;
    for (_, _, inventory) in &players {
        total_cards += inventory.rock;
        total_cards += inventory.paper;
        total_cards += inventory.scissors;
    }
    if total_cards < 2 {
        // there is only one card, cannot proceed
        return;
    }

    let mut players = players
        .iter_mut()
        .filter(|(entity, _, _)| {
            tables
                .iter()
                .map(|table| table.0)
                .collect_vec()
                .concat()
                .contains(entity)
                .not()
        })
        .filter(|(_, _, inventory)| inventory.is_alive())
        .filter(|(_, _, inventory)| !inventory.is_safe())
        .collect_vec();

    if players.len() < MIN_MATCH_PLAYERS {
        return;
    }

    fastrand::shuffle(&mut players);

    for (x, y) in players.into_iter().tuples() {
        let table = Table::new(x.0, y.0);
        let name = Name::new(format!("Table ({}, {})", x.1, y.1));
        commands.spawn((table, name));
    }
}

fn remove_players(
    mut commands: Commands,
    players: Query<(Entity, &Name, &Inventory), With<Player>>,
) {
    for (entity, name, inventory) in &players {
        if !inventory.is_alive() {
            bevy::log::info!("player dead: {name}");
            commands.entity(entity).despawn_recursive();
        }
        if inventory.is_safe() {
            bevy::log::info!("player safe: {name}");
            commands.entity(entity).despawn_recursive();
        }
    }
}

#[derive(Debug, Component)]
struct DuelTask(Task<Result<[Inventory; 2]>>);

fn start_duel(
    mut commands: Commands,
    state: Res<PublicState>,
    players: Query<(Entity, &Name, &Inventory, &Player)>,
    tables: Query<(Entity, &Table), Without<DuelTask>>,
) {
    let thread_pool = IoTaskPool::get();
    for (entity, table) in &tables {
        let (Ok(x), Ok(y)) = (players.get(table[0]), players.get(table[1])) else {
            continue;
        };
        assert!(x.2.is_alive());
        assert!(y.2.is_alive());

        let state = state.clone();
        let actors = [x.3.actor.clone(), y.3.actor.clone()];
        let data = [
            PlayerData {
                entity: x.0,
                name: x.1.clone(),
                inventory: x.2.clone(),
            },
            PlayerData {
                entity: y.0,
                name: y.1.clone(),
                inventory: y.2.clone(),
            },
        ];
        let task = thread_pool.spawn(duel(state, actors, data));
        commands.entity(entity).insert(DuelTask(task));
    }
}

fn poll_duel(
    mut commands: Commands,
    mut players: Query<&mut Inventory, With<Player>>,
    mut tables: Query<(Entity, &Table, &mut DuelTask), Without<Player>>,
) {
    for (entity, table, mut task) in &mut tables {
        if let Some(result) = block_on(future::poll_once(&mut task.0)) {
            match result {
                Ok([m, n]) => {
                    if let Ok(mut x) = players.get_mut(table[0]) {
                        *x = m;
                    }
                    if let Ok(mut y) = players.get_mut(table[1]) {
                        *y = n;
                    }
                }
                Err(err) => bevy::log::warn!("duel error: {err}"),
            }
            commands.entity(entity).despawn_recursive();
        }
    }
}

#[derive(Debug, Clone)]
pub struct PlayerData {
    pub entity: Entity,
    pub name: Name,
    pub inventory: Inventory,
}

#[derive(Debug, Clone)]
pub struct OpponentData {
    pub name: Name,
    pub star: u32,
    pub card: u32,
}

impl From<PlayerData> for OpponentData {
    fn from(value: PlayerData) -> Self {
        Self {
            name: value.name,
            star: value.inventory.star,
            card: value.inventory.num_cards(),
        }
    }
}

#[derive(Debug, Default, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Role {
    #[default]
    None,
    System(Entity),
    Assistant(Entity),
    Actor(Entity, String),
    Inner(Entity, String),
}

impl Role {
    pub fn actor(entity: Entity, name: impl AsRef<str>) -> Self {
        let name = name.as_ref().trim().to_owned();
        Self::Actor(entity, name)
    }

    pub fn inner(entity: Entity, name: impl AsRef<str>) -> Self {
        let name = name.as_ref().trim().to_owned();
        Self::Inner(entity, name)
    }
}

impl Display for Role {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Role::None => write!(f, ""),
            Role::System(_) => write!(f, "{SYSTEM_NAME}",),
            Role::Assistant(_) => write!(f, "{ASSISTANT_NAME}",),
            Role::Actor(_, name) => write!(f, "{name}"),
            Role::Inner(_, name) => write!(f, "{name} (inner voice)"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ChatKind {
    Trade(usize),
    Duel(usize),
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChatRecordId;

#[derive(Debug, Derivative, Clone, Serialize, Deserialize)]
#[derivative(PartialEq, Eq)]
pub struct ChatRecord {
    #[serde(skip)]
    pub id: uid::Id<ChatRecordId>,
    #[derivative(PartialEq = "ignore")]
    pub role: Role,
    #[derivative(PartialEq = "ignore")]
    pub content: String,
}

impl ChatRecord {
    pub fn new(role: Role, content: impl ToString) -> Self {
        let id = uid::Id::new();
        let content = content.to_string();
        Self { id, role, content }
    }
}

impl Display for ChatRecord {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.role, self.content.trim())
    }
}

#[derive(Debug, Clone, Copy)]
pub struct TradeState<'a> {
    pub this: &'a Trade,
    pub that: &'a Trade,
}

#[derive(Debug, Clone, Copy)]
pub struct StakeState<'a> {
    pub this: &'a Stake,
    pub that: &'a Stake,
}

pub trait Actor: ConditionalSend + Sync + 'static {
    /// Notify the actor about how many cards are there on the stage.
    fn notify<'a>(
        &'a mut self,
        player: &'a PlayerData,
        state: &'a PublicState,
    ) -> BoxedFuture<'a, ()>;
    /// Provide feedback to the actor (due to erroneous actions).
    fn feedback_error<'a>(
        &'a mut self,
        player: &'a PlayerData,
        text: String,
    ) -> BoxedFuture<'a, ()>;
    /// Chat with the actor.
    fn chat<'a>(
        &'a mut self,
        player: &'a PlayerData,
        opponent: &'a OpponentData,
        history: &'a [ChatRecord],
        kind: ChatKind,
    ) -> BoxedFuture<'a, Vec<ChatRecord>>;
    /// Trade with another actor.
    fn trade<'a>(
        &'a mut self,
        player: &'a PlayerData,
        opponent: &'a OpponentData,
        history: &'a [ChatRecord],
    ) -> BoxedFuture<'a, Trade>;
    /// Accept the trade or not.
    fn accept_trade<'a>(
        &'a mut self,
        player: &'a PlayerData,
        opponent: &'a OpponentData,
        history: &'a [ChatRecord],
        state: TradeState<'a>,
    ) -> BoxedFuture<'a, bool>;
    /// Feedback on accepting the trade or not.
    fn feedback_trade<'a>(
        &'a mut self,
        player: &'a PlayerData,
        state: [bool; 2],
    ) -> BoxedFuture<'a, ()>;
    /// Bet for the duel.
    fn bet<'a>(
        &'a mut self,
        player: &'a PlayerData,
        opponent: &'a OpponentData,
        history: &'a [ChatRecord],
    ) -> BoxedFuture<'a, Stake>;
    /// Accept the duel or not. If accepts, draw a card from the inventory.
    fn accept_duel<'a>(
        &'a mut self,
        player: &'a PlayerData,
        opponent: &'a OpponentData,
        history: &'a [ChatRecord],
        state: StakeState<'a>,
    ) -> BoxedFuture<'a, Option<Card>>;
}

#[derive(Debug, Default, Clone, Copy)]
pub struct DummyActor;

impl Actor for DummyActor {
    fn notify<'a>(
        &'a mut self,
        _player: &'a PlayerData,
        _state: &'a PublicState,
    ) -> BoxedFuture<'a, ()> {
        Box::pin(async move {})
    }

    fn feedback_error<'a>(
        &'a mut self,
        _player: &'a PlayerData,
        text: String,
    ) -> BoxedFuture<'a, ()> {
        Box::pin(async move { panic!("{text}") })
    }

    fn chat<'a>(
        &'a mut self,
        _player: &'a PlayerData,
        _opponent: &'a OpponentData,
        _history: &'a [ChatRecord],
        _kind: ChatKind,
    ) -> BoxedFuture<'a, Vec<ChatRecord>> {
        Box::pin(async move { vec![] })
    }

    fn trade<'a>(
        &'a mut self,
        player: &'a PlayerData,
        _opponent: &'a OpponentData,
        _history: &'a [ChatRecord],
    ) -> BoxedFuture<'a, Trade> {
        Box::pin(async move {
            let deck = [
                vec![Card::Rock; player.inventory.rock as usize],
                vec![Card::Paper; player.inventory.paper as usize],
                vec![Card::Scissors; player.inventory.scissors as usize],
            ]
            .concat();
            let card = fastrand::choice(&deck).cloned();
            match card {
                Some(Card::Rock) => Trade {
                    rock: 1,
                    ..Default::default()
                },
                Some(Card::Paper) => Trade {
                    paper: 1,
                    ..Default::default()
                },
                Some(Card::Scissors) => Trade {
                    scissors: 1,
                    ..Default::default()
                },
                None => Trade::default(),
            }
        })
    }

    fn accept_trade<'a>(
        &'a mut self,
        _player: &'a PlayerData,
        _opponent: &'a OpponentData,
        _history: &'a [ChatRecord],
        _state: TradeState<'a>,
    ) -> BoxedFuture<'a, bool> {
        Box::pin(async move { true })
    }

    fn feedback_trade<'a>(
        &'a mut self,
        _player: &'a PlayerData,
        _state: [bool; 2],
    ) -> BoxedFuture<'a, ()> {
        Box::pin(async move {})
    }

    fn bet<'a>(
        &'a mut self,
        _player: &'a PlayerData,
        _opponent: &'a OpponentData,
        _history: &'a [ChatRecord],
    ) -> BoxedFuture<'a, Stake> {
        Box::pin(async move { Default::default() })
    }

    fn accept_duel<'a>(
        &'a mut self,
        player: &'a PlayerData,
        _opponent: &'a OpponentData,
        _history: &'a [ChatRecord],
        _state: StakeState<'a>,
    ) -> BoxedFuture<'a, Option<Card>> {
        Box::pin(async move {
            let deck = [
                vec![Card::Rock; player.inventory.rock as usize],
                vec![Card::Paper; player.inventory.paper as usize],
                vec![Card::Scissors; player.inventory.scissors as usize],
            ]
            .concat();
            fastrand::choice(&deck).cloned()
        })
    }
}

pub async fn duel(
    state: PublicState,
    [a0, a1]: [Arc<Mutex<dyn Actor>>; 2],
    [mut p0, mut p1]: [PlayerData; 2],
) -> Result<[Inventory; 2]> {
    let (mut a0, mut a1) = join!(a0.lock(), a1.lock());

    // step 1: notify both players about public state
    join!(a0.notify(&p0, &state), a1.notify(&p1, &state));

    // step 2: players chat before trade
    let mut history: Vec<ChatRecord> = vec![];

    // remove opponent's private history
    let observe = |data: &PlayerData, history: &[ChatRecord]| {
        history
            .iter()
            .filter(|x| match x.role.clone() {
                Role::Assistant(entity) | Role::Inner(entity, _) => entity == data.entity,
                _ => true,
            })
            .cloned()
            .collect_vec()
    };

    for round in 0..NUM_CHAT_ROUNDS {
        let h0 = observe(&p0, &history);
        let q0 = p1.clone().into();
        let r0 = round * 2;
        let mut records = a0.chat(&p0, &q0, &h0, ChatKind::Trade(r0)).await;
        history.append(&mut records);

        let h1 = observe(&p1, &history);
        let q1 = p0.clone().into();
        let r1 = r0 + 1;
        let mut records = a1.chat(&p1, &q1, &h1, ChatKind::Trade(r1)).await;
        history.append(&mut records);
    }

    // step 3: players trade
    let (t0, t1) = {
        let (Some((t0, x0)), Some((t1, x1))) = join!(
            async {
                let mut round = 0;
                loop {
                    if round > MAX_TRAIL_ROUNDS {
                        break None;
                    }
                    round += 1;

                    let h0 = observe(&p0, &history);
                    let q0 = p1.clone().into();
                    let trade = a0.trade(&p0, &q0, &h0).await;
                    let inventory = match p0.inventory.split_trade(&trade) {
                        Ok(inventory) => inventory,
                        Err(err) => {
                            a0.feedback_error(&p0, format!("Error: {err}")).await;
                            continue;
                        }
                    };

                    break Some((trade, inventory));
                }
            },
            async {
                let mut round = 0;
                loop {
                    if round > MAX_TRAIL_ROUNDS {
                        break None;
                    }
                    round += 1;

                    let h1 = observe(&p1, &history);
                    let q1 = p0.clone().into();
                    let trade = a1.trade(&p1, &q1, &h1).await;
                    let inventory = match p1.inventory.split_trade(&trade) {
                        Ok(inventory) => inventory,
                        Err(err) => {
                            a1.feedback_error(&p1, format!("Error: {err}")).await;
                            continue;
                        }
                    };

                    break Some((trade, inventory));
                }
            }
        ) else {
            bail!("trade failed too many times");
        };

        // success, update inventories
        p0.inventory = x0;
        p1.inventory = x1;
        (t0, t1)
    };

    // step 4: players agree on the trade
    {
        let q0 = p1.clone().into();
        let q1 = p0.clone().into();
        match join!(
            a0.accept_trade(
                &p0,
                &q0,
                &history,
                TradeState {
                    this: &t0,
                    that: &t1
                }
            ),
            a1.accept_trade(
                &p1,
                &q1,
                &history,
                TradeState {
                    this: &t1,
                    that: &t0
                }
            )
        ) {
            (true, true) => {
                // players do reach an agreement, perform the trade
                p0.inventory.apply_trade(&t1);
                p1.inventory.apply_trade(&t0);
                join!(
                    a0.feedback_trade(&p0, [true, true]),
                    a1.feedback_trade(&p1, [true, true])
                );
            }
            (u0, u1) => {
                // players do not reach an agreement, rewind
                p0.inventory.apply_trade(&t0);
                p1.inventory.apply_trade(&t1);
                join!(
                    a0.feedback_trade(&p0, [u0, u1]),
                    a1.feedback_trade(&p1, [u1, u0])
                );
            }
        }
    }

    // check if we can proceed to duel
    if [&p0, &p1].iter().any(|x| !x.inventory.can_duel()) {
        return Ok([p0, p1].map(|x| x.inventory));
    }

    // step 5: player chat before duel
    let mut history: Vec<ChatRecord> = vec![];

    for round in 0..NUM_CHAT_ROUNDS {
        let h0 = observe(&p0, &history);
        let q0 = p1.clone().into();
        let r0 = round * 2;
        let mut records = a0.chat(&p0, &q0, &h0, ChatKind::Duel(r0)).await;
        history.append(&mut records);

        let h1 = observe(&p1, &history);
        let q1 = p0.clone().into();
        let r1 = r0 + 1;
        let mut records = a1.chat(&p1, &q1, &h1, ChatKind::Duel(r1)).await;
        history.append(&mut records);
    }

    // step 6: players bet
    let (s0, s1) = {
        let (Some((s0, x0)), Some((s1, x1))) = join!(
            async {
                let mut round = 0;
                loop {
                    if round > MAX_TRAIL_ROUNDS {
                        break None;
                    }
                    round += 1;

                    let h0 = observe(&p0, &history);
                    let q0 = p1.clone().into();
                    let stake = a0.bet(&p0, &q0, &h0).await;
                    let inventory = match p0.inventory.split_stake(&stake) {
                        Ok(inventory) => inventory,
                        Err(err) => {
                            a0.feedback_error(&p0, format!("Error: {err}")).await;
                            continue;
                        }
                    };

                    break Some((stake, inventory));
                }
            },
            async {
                let mut round = 0;
                loop {
                    if round > MAX_TRAIL_ROUNDS {
                        break None;
                    }
                    round += 1;

                    let h1 = observe(&p0, &history);
                    let q1 = p0.clone().into();
                    let stake = a1.bet(&p1, &q1, &h1).await;
                    let inventory = match p1.inventory.split_stake(&stake) {
                        Ok(inventory) => inventory,
                        Err(err) => {
                            a1.feedback_error(&p1, format!("Error: {err}")).await;
                            continue;
                        }
                    };

                    break Some((stake, inventory));
                }
            }
        ) else {
            bail!("bet failed too many times");
        };

        // success, update inventories
        p0.inventory = x0;
        p1.inventory = x1;
        (s0, s1)
    };

    // step 7: players agree on the duel
    let mut round = 0;
    let cards = loop {
        if round > MAX_TRAIL_ROUNDS {
            bail!("duel failed too many times");
        }
        round += 1;

        let q0 = p1.clone().into();
        let q1 = p0.clone().into();
        let cards = join!(
            a0.accept_duel(
                &p0,
                &q0,
                &history,
                StakeState {
                    this: &s0,
                    that: &s1
                }
            ),
            a1.accept_duel(
                &p1,
                &q1,
                &history,
                StakeState {
                    this: &s1,
                    that: &s0
                }
            )
        );

        if let (Some(lhs), Some(rhs)) = cards {
            let x0 = match p0.inventory.split_duel(lhs) {
                Ok(inventory) => inventory,
                Err(err) => {
                    a0.feedback_error(&p0, format!("Error: {err}")).await;
                    continue;
                }
            };
            let x1 = match p1.inventory.split_duel(rhs) {
                Ok(inventory) => inventory,
                Err(err) => {
                    a1.feedback_error(&p1, format!("Error: {err}")).await;
                    continue;
                }
            };

            // success, update inventories
            p0.inventory = x0;
            p1.inventory = x1;
        }

        break cards;
    };

    match cards {
        (Some(lhs), Some(rhs)) => match lhs.compare(rhs) {
            Some(index) => {
                let stake = s0 + s1;
                [&mut p0, &mut p1][index].inventory.apply_stake(&stake);
            }
            None => {
                p0.inventory.apply_stake(&s0);
                p1.inventory.apply_stake(&s1);
            }
        },
        _ => {
            p0.inventory.apply_stake(&s0);
            p1.inventory.apply_stake(&s1);
        }
    }

    Ok([p0, p1].map(|x| x.inventory))
}
