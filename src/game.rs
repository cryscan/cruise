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

pub const NUM_PLAYERS: usize = 64;
pub const MIN_MATCH_PLAYERS: usize = 8;
pub const NUM_CHAT_ROUNDS: usize = 4;
pub const MAX_TRAIL_ROUNDS: usize = 3;

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
                (update_public_state, match_players, start_duel, poll_duel),
            );
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Card {
    Rock,
    Paper,
    Scissors,
}

#[derive(Debug, Derivative, Clone, Component, Reflect)]
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

#[derive(Debug, Default, Clone, Reflect, Serialize, Deserialize)]
#[reflect(Default)]
pub struct Trade {
    pub star: u32,
    pub coin: u32,
    pub rock: u32,
    pub paper: u32,
    pub scissors: u32,
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

#[derive(Debug, Derivative, Clone, Reflect, Serialize, Deserialize)]
#[derivative(Default)]
#[reflect(Default)]
pub struct Stake {
    #[derivative(Default(value = "1"))]
    pub star: u32,
    pub coin: u32,
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
    pub fn is_alive(&self) -> bool {
        self.star > 0
    }

    pub fn num_cards(&self) -> u32 {
        self.rock + self.paper + self.scissors
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

    pub fn apply_trade(&self, trade: &Trade) -> Self {
        Self {
            star: self.star + trade.star,
            coin: self.coin + trade.coin,
            rock: self.rock + trade.rock,
            paper: self.paper + trade.paper,
            scissors: self.scissors + trade.scissors,
        }
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

    pub fn apply_stake(&self, stake: &Stake) -> Self {
        Self {
            star: self.star + stake.star,
            coin: self.coin + stake.coin,
            ..self.clone()
        }
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

#[derive(Debug, Default, Clone, Resource, Reflect)]
#[reflect(Resource)]
pub struct PublicState {
    pub rock: u32,
    pub paper: u32,
    pub scissors: u32,
}

fn setup_scene(mut commands: Commands) {
    let names = NAMES.split("\n").map(|x| x.trim()).collect_vec();
    commands.spawn_batch((0..NUM_PLAYERS).map(move |index| {
        (
            Name::new(names[index]),
            Player::new(DummyActor),
            Inventory::default(),
        )
    }));
}

fn update_public_state(mut state: ResMut<PublicState>, players: Query<&Inventory, With<Player>>) {
    let mut x = PublicState::default();
    for inventory in &players {
        x.rock += inventory.rock;
        x.paper += inventory.paper;
        x.scissors += inventory.scissors;
    }
    *state = x;
}

/// Find players that are not currently in match, and put them onto a table.
fn match_players(
    mut commands: Commands,
    mut players: Query<(Entity, &Name, &Inventory), With<Player>>,
    tables: Query<&Table, Without<Player>>,
) {
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

#[derive(Debug, Clone, Reflect)]
pub struct PlayerData {
    pub entity: Entity,
    pub name: Name,
    pub inventory: Inventory,
}

#[derive(Debug, Clone, Reflect)]
pub enum Role {
    System(Entity),
    Actor(Entity, Name),
    Inner(Entity, Name),
}

impl Display for Role {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Role::System(_) => write!(f, "System"),
            Role::Actor(_, name) => writeln!(f, "{name}"),
            Role::Inner(_, name) => writeln!(f, "{name} (Thinks)"),
        }
    }
}

#[derive(Debug, Clone, Reflect)]
pub struct ChatRecord {
    pub role: Role,
    pub content: String,
}

#[derive(Debug, Clone, Copy, Reflect)]
pub struct TradeState<'a> {
    pub mine: &'a Trade,
    pub others: &'a Trade,
}

#[derive(Debug, Clone, Copy, Reflect)]
pub struct StakeState<'a> {
    pub mine: &'a Stake,
    pub others: &'a Stake,
}

pub trait Actor: ConditionalSend + Sync + 'static {
    /// Notify the actor about how many cards are there on the stage.
    fn notify<'a>(&'a mut self, state: &'a PublicState) -> BoxedFuture<'a, ()>;
    /// Provide feedback to the actor (due to erroneous actions).
    fn feedback<'a>(&'a mut self, text: String) -> BoxedFuture<'a, ()>;
    /// Chat with the actor.
    fn chat<'a>(
        &'a mut self,
        data: &'a PlayerData,
        history: &'a [ChatRecord],
        round: usize,
    ) -> BoxedFuture<'a, Vec<ChatRecord>>;
    /// Trade with another actor.
    fn trade<'a>(
        &'a mut self,
        data: &'a PlayerData,
        history: &'a [ChatRecord],
    ) -> BoxedFuture<'a, Trade>;
    /// Accept the trade or not.
    fn accept_trade<'a>(
        &'a mut self,
        data: &'a PlayerData,
        history: &'a [ChatRecord],
        state: TradeState<'a>,
    ) -> BoxedFuture<'a, bool>;
    /// Raise the stake of the duel.
    fn raise_stake<'a>(
        &'a mut self,
        data: &'a PlayerData,
        history: &'a [ChatRecord],
    ) -> BoxedFuture<'a, Stake>;
    /// Accept the duel or not. If accepts, draw a card from the inventory.
    fn accept_duel<'a>(
        &'a mut self,
        data: &'a PlayerData,
        history: &'a [ChatRecord],
        state: StakeState<'a>,
    ) -> BoxedFuture<'a, Option<Card>>;
}

#[derive(Debug, Default, Clone, Copy, Reflect)]
#[reflect(Default)]
pub struct DummyActor;

impl Actor for DummyActor {
    fn notify<'a>(&'a mut self, _state: &'a PublicState) -> BoxedFuture<'a, ()> {
        Box::pin(async move {})
    }

    fn feedback<'a>(&'a mut self, _text: String) -> BoxedFuture<'a, ()> {
        Box::pin(async move {})
    }

    fn chat<'a>(
        &'a mut self,
        _data: &'a PlayerData,
        _history: &'a [ChatRecord],
        _round: usize,
    ) -> BoxedFuture<'a, Vec<ChatRecord>> {
        Box::pin(async move { vec![] })
    }

    fn trade<'a>(
        &'a mut self,
        _data: &'a PlayerData,
        _history: &'a [ChatRecord],
    ) -> BoxedFuture<'a, Trade> {
        Box::pin(async move { Default::default() })
    }

    fn accept_trade<'a>(
        &'a mut self,
        _data: &'a PlayerData,
        _history: &'a [ChatRecord],
        _state: TradeState<'a>,
    ) -> BoxedFuture<'a, bool> {
        Box::pin(async move { true })
    }

    fn raise_stake<'a>(
        &'a mut self,
        _data: &'a PlayerData,
        _history: &'a [ChatRecord],
    ) -> BoxedFuture<'a, Stake> {
        Box::pin(async move { Default::default() })
    }

    fn accept_duel<'a>(
        &'a mut self,
        _data: &'a PlayerData,
        _history: &'a [ChatRecord],
        _state: StakeState<'a>,
    ) -> BoxedFuture<'a, Option<Card>> {
        Box::pin(async move {
            match fastrand::u8(0..4) {
                0 => Some(Card::Rock),
                1 => Some(Card::Paper),
                2 => Some(Card::Scissors),
                _ => None,
            }
        })
    }
}

pub async fn duel(
    state: PublicState,
    actors: [Arc<Mutex<dyn Actor>>; 2],
    mut data: [PlayerData; 2],
) -> Result<[Inventory; 2]> {
    let mut actors = join!(actors[0].lock(), actors[1].lock());

    // step 1: notify both players about public state
    join!(actors.0.notify(&state), actors.1.notify(&state));

    // step 2: players chat before trade
    let mut history: Vec<ChatRecord> = vec![];

    // remove opponent's private history
    let observe = |history: &[ChatRecord], index: usize| {
        history
            .iter()
            .filter(|x| match x.role.clone() {
                Role::System(entity) | Role::Inner(entity, _) => entity == data[index].entity,
                Role::Actor(_, _) => true,
            })
            .cloned()
            .collect_vec()
    };

    for round in 0..NUM_CHAT_ROUNDS {
        let observation = observe(&history, 0);
        let mut records = actors.0.chat(&data[0], &observation, round).await;
        history.append(&mut records);

        let observation = observe(&history, 1);
        let mut records = actors.1.chat(&data[1], &observation, round).await;
        history.append(&mut records);
    }

    // step 3: players trade
    let mut round = 0;
    let trades = loop {
        if round > MAX_TRAIL_ROUNDS {
            bail!("trade failed too many times");
        }
        round += 1;

        let trades = join!(
            actors.0.trade(&data[0], &history),
            actors.1.trade(&data[1], &history)
        );
        let x0 = match data[0].inventory.split_trade(&trades.0) {
            Ok(inventory) => inventory,
            Err(err) => {
                actors.0.feedback(format!("Error: {err}")).await;
                continue;
            }
        };
        let x1 = match data[1].inventory.split_trade(&trades.1) {
            Ok(inventory) => inventory,
            Err(err) => {
                actors.1.feedback(format!("Error: {err}")).await;
                continue;
            }
        };

        // success, update inventories
        data[0].inventory = x0;
        data[1].inventory = x1;
        break trades;
    };

    // step 4: players agree on the trade
    let agreements = join!(
        actors.0.accept_trade(
            &data[0],
            &history,
            TradeState {
                mine: &trades.0,
                others: &trades.1
            }
        ),
        actors.1.accept_trade(
            &data[1],
            &history,
            TradeState {
                mine: &trades.1,
                others: &trades.0
            }
        )
    );
    match agreements {
        (true, true) => {
            // players do reach an agreement, perform the trade
            data[0].inventory.apply_trade(&trades.1);
            data[1].inventory.apply_trade(&trades.0);
        }
        _ => {
            // players do not reach an agreement, rewind
            data[0].inventory.apply_trade(&trades.0);
            data[1].inventory.apply_trade(&trades.1);
        }
    }

    Ok(data.map(|x| x.inventory))
}
