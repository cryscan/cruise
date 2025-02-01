use bevy::prelude::*;
use derivative::Derivative;
use itertools::Itertools;
use serde::{Deserialize, Serialize};
use thiserror::Error;

pub const NUM_PLAYERS: usize = 64;

const NAMES: &str = include_str!("names.txt");

#[derive(Debug, Default)]
pub struct GamePlugin;

impl Plugin for GamePlugin {
    fn build(&self, app: &mut App) {
        app.register_type::<Inventory>()
            .register_type::<Player>()
            .register_type::<Table>()
            .add_systems(Startup, setup_scene)
            .add_systems(Update, match_players);
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

#[derive(Debug, Clone, Component, Reflect)]
#[reflect(Component)]
pub struct Table {
    pub state: [(Entity, Stake); 2],
    pub turn: usize,
}

impl Table {
    pub fn new(x: Entity, y: Entity) -> Self {
        Self {
            state: [(x, Stake::default()), (y, Stake::default())],
            turn: 0,
        }
    }
}

fn setup_scene(mut commands: Commands) {
    let names = NAMES.split("\n").collect_vec();
    commands.spawn_batch(
        (0..NUM_PLAYERS).map(move |index| (Name::new(names[index]), Player, Inventory::default())),
    );
}

fn match_players(
    mut commands: Commands,
    mut players: Query<(Entity, &Name, &mut Inventory), With<Player>>,
    tables: Query<&Table, Without<Player>>,
) {
    // find players that are not currently in match
    let players = players.iter_mut().filter(|(entity, _, _)| {
        !tables
            .iter()
            .map(|table| table.state.clone().map(|x| x.0))
            .collect_vec()
            .concat()
            .contains(entity)
    });
    for (mut x, mut y) in players.tuples() {
        let table = Table::new(x.0, y.0);

        let (Ok(m), Ok(n)) = (x.2.take(&table.state[0].1), y.2.take(&table.state[1].1)) else {
            continue;
        };
        *x.2 = m;
        *y.2 = n;

        commands.spawn((table, Name::new(format!("Table ({}, {})", x.1, y.1))));
    }
}
