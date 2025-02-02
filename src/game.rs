use std::{
    fmt::Display,
    ops::{Add, Not},
    time::Duration,
};

use anyhow::Result;
use async_std::task::block_on;
use bevy::{
    prelude::*,
    tasks::{futures_lite::future, IoTaskPool, Task},
};
use derivative::Derivative;
use itertools::Itertools;
use serde::{Deserialize, Serialize};
use thiserror::Error;

pub const NUM_PLAYERS: usize = 64;
pub const MIN_MATCH_PLAYERS: usize = 8;

const NAMES: &str = include_str!("names.txt");

#[derive(Debug, Default)]
pub struct GamePlugin;

impl Plugin for GamePlugin {
    fn build(&self, app: &mut App) {
        app.register_type::<Inventory>()
            .register_type::<Player>()
            .register_type::<Table>()
            .add_systems(Startup, setup_scene)
            .add_systems(Update, (match_players, start_duel, end_duel));
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

#[derive(Debug, Error)]
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

#[derive(Debug, Error)]
pub enum StakeError {
    #[error("cannot take out {1} star(s) since you only have {0} star(s)")]
    Star(u32, u32),
    #[error("cannot take out {1} coin(s) since you only have {0} coin(s)")]
    Coin(u32, u32),
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
}

#[derive(Debug, Default, Clone, Copy, Component, Reflect)]
#[reflect(Component)]
pub struct Player;

#[derive(Debug, Clone, Deref, DerefMut, Component, Reflect)]
#[reflect(Component)]
pub struct Table(pub [Entity; 2]);

impl Table {
    pub fn new(x: Entity, y: Entity) -> Self {
        Self([x, y])
    }
}

fn setup_scene(mut commands: Commands) {
    let names = NAMES.split("\n").map(|x| x.trim()).collect_vec();
    commands.spawn_batch(
        (0..NUM_PLAYERS).map(move |index| (Name::new(names[index]), Player, Inventory::default())),
    );
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
    players: Query<(Entity, &Name, &Inventory), With<Player>>,
    tables: Query<(Entity, &Table), Without<DuelTask>>,
) {
    let thread_pool = IoTaskPool::get();
    for (entity, table) in &tables {
        let (Ok(x), Ok(y)) = (players.get(table[0]), players.get(table[1])) else {
            continue;
        };
        assert!(x.2.is_alive());
        assert!(y.2.is_alive());

        let players = [(x.1.clone(), x.2.clone()), (y.1.clone(), y.2.clone())];
        let task = thread_pool.spawn(duel(players));
        commands.entity(entity).insert(DuelTask(task));
    }
}

fn end_duel(
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
pub enum Role {
    System,
    Actor(String),
    Inner(String),
}

impl Display for Role {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Role::System => write!(f, "System"),
            Role::Actor(name) => writeln!(f, "{name}"),
            Role::Inner(name) => writeln!(f, "{name} (Thinks)"),
        }
    }
}

#[derive(Debug, Clone, Reflect)]
pub struct ChatRecord {
    pub role: Role,
    pub content: String,
}

#[derive(Debug, Default, Clone, Reflect)]
pub struct GameState {
    pub rock: u32,
    pub paper: u32,
    pub scissors: u32,
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

pub trait Actor {
    /// Notify the actor about how many cards are there on the stage.
    fn notify(&mut self, state: &GameState);
    /// Chat with the actor.
    fn chat(&mut self, inventory: &Inventory, history: &[ChatRecord]) -> ChatRecord;
    /// Trade with another actor.
    fn trade(&mut self, inventory: &Inventory, history: &[ChatRecord]) -> Trade;
    /// If accepting the trade.
    fn accept_trade(
        &mut self,
        inventory: &Inventory,
        history: &[ChatRecord],
        state: TradeState<'_>,
    ) -> bool;
    /// Raise the stake of the duel.
    fn raise_stake(&mut self, inventory: &Inventory, history: &[ChatRecord]) -> Stake;
    /// If accepting the duel.
    fn accept_duel(
        &mut self,
        inventory: &Inventory,
        history: &[ChatRecord],
        state: StakeState<'_>,
    ) -> bool;
}

pub async fn duel(players: [(Name, Inventory); 2]) -> Result<[Inventory; 2]> {
    async_std::task::sleep(Duration::from_secs_f32(fastrand::f32() * 10.0)).await;

    let [(n0, x0), (n1, x1)] = players;

    bevy::log::info!("{n0}, {n1}");
    Ok([x0, x1])
}
