use std::time::Duration;

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

#[derive(Debug, Derivative, Clone, Reflect, Serialize, Deserialize)]
#[derivative(Default)]
#[reflect(Default)]
pub struct Stake {
    #[derivative(Default(value = "1"))]
    pub star: u32,
    pub coin: u32,
    pub rock: u32,
    pub paper: u32,
    pub scissors: u32,
}

impl std::ops::Add<Stake> for Stake {
    type Output = Self;

    fn add(self, rhs: Stake) -> Self::Output {
        Stake {
            star: self.star + rhs.star,
            coin: self.coin + rhs.coin,
            rock: self.rock + rhs.rock,
            paper: self.paper + rhs.paper,
            scissors: self.scissors + rhs.scissors,
        }
    }
}

#[derive(Debug, Error)]
pub enum StakeError {
    #[error("you cannot take out {1} star(s) because you only have {0} star(s)")]
    Star(u32, u32),
    #[error("you cannot take out {1} coin(s) because you only have {0} coin(s)")]
    Coin(u32, u32),
    #[error("you cannot take out {1} rock card(s) because you only have {0} rock card(s)")]
    Rock(u32, u32),
    #[error("you cannot take out {1} paper card(s) because you only have {0} paper card(s)")]
    Paper(u32, u32),
    #[error("you cannot take out {1} scissors card(s) because you only have {0} scissors card(s)")]
    Scissors(u32, u32),
}

impl Inventory {
    pub fn take(&self, stake: &Stake) -> Result<Self, StakeError> {
        match (self, stake) {
            (x, y) if x.star < y.star => Err(StakeError::Star(x.star, y.star)),
            (x, y) if x.coin < y.coin => Err(StakeError::Coin(x.star, y.star)),
            (x, y) if x.rock < y.rock => Err(StakeError::Rock(x.star, y.star)),
            (x, y) if x.paper < y.paper => Err(StakeError::Paper(x.star, y.star)),
            (x, y) if x.scissors < y.scissors => Err(StakeError::Scissors(x.star, y.star)),
            (x, y) => Ok(Self {
                star: x.star - y.star,
                coin: x.coin - y.coin,
                rock: x.rock - y.rock,
                paper: x.paper - y.paper,
                scissors: x.scissors - y.scissors,
            }),
        }
    }

    pub fn receive(&self, stake: Stake) -> Self {
        Self {
            star: self.star + stake.star,
            coin: self.coin + stake.coin,
            rock: self.rock + stake.rock,
            paper: self.paper + stake.paper,
            scissors: self.scissors + stake.scissors,
        }
    }
}

#[derive(Debug, Default, Clone, Copy, Component, Reflect)]
#[reflect(Component)]
pub struct Player;

#[derive(Debug, Clone, Deref, DerefMut, Component, Reflect)]
#[reflect(Component)]
pub struct Table(pub [(Entity, Stake); 2]);

impl Table {
    pub fn new(x: Entity, y: Entity) -> Self {
        Self([(x, Stake::default()), (y, Stake::default())])
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
    mut players: Query<(Entity, &Name, &mut Inventory), With<Player>>,
    tables: Query<&Table, Without<Player>>,
) {
    let mut players = players
        .iter_mut()
        .filter(|(entity, _, _)| {
            !tables
                .iter()
                .map(|table| table.0.clone().map(|x| x.0))
                .collect_vec()
                .concat()
                .contains(entity)
        })
        .collect_vec();

    if players.len() < MIN_MATCH_PLAYERS {
        return;
    }

    fastrand::shuffle(&mut players);

    for (mut x, mut y) in players.into_iter().tuples() {
        let table = Table::new(x.0, y.0);

        let (Ok(m), Ok(n)) = (x.2.take(&table[0].1), y.2.take(&table[1].1)) else {
            continue;
        };
        *x.2 = m;
        *y.2 = n;

        commands.spawn((table, Name::new(format!("Table ({}, {})", x.1, y.1))));
    }
}

#[derive(Debug, Component)]
struct DuelTask(Task<Result<[Inventory; 2]>>);

async fn duel(_table: Table, players: [(Name, Inventory); 2]) -> Result<[Inventory; 2]> {
    async_std::task::sleep(Duration::from_secs_f32(fastrand::f32() * 10.0)).await;

    let [(n0, x0), (n1, x1)] = players;

    bevy::log::info!("{n0}, {n1}");
    Ok([x0, x1])
}

fn start_duel(
    mut commands: Commands,
    players: Query<(Entity, &Name, &Inventory), With<Player>>,
    tables: Query<(Entity, &Table), Without<DuelTask>>,
) {
    let thread_pool = IoTaskPool::get();
    for (entity, table) in &tables {
        let (Ok(x), Ok(y)) = (players.get(table[0].0), players.get(table[1].0)) else {
            continue;
        };
        let table = table.clone();
        let players = [(x.1.clone(), x.2.clone()), (y.1.clone(), y.2.clone())];
        let task = thread_pool.spawn(duel(table, players));
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
                    let Ok(mut x) = players.get_mut(table[0].0) else {
                        continue;
                    };
                    *x = m;

                    let Ok(mut y) = players.get_mut(table[1].0) else {
                        continue;
                    };
                    *y = n;
                }
                Err(err) => bevy::log::warn!("duel error: {err}"),
            }
            commands.entity(entity).despawn_recursive();
        }
    }
}
